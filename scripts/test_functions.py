"""Function-level test suite: cheap, CPU-only, no model loads, no network, no GPU.

Covers the pure logic and file handling across the project — output layout and migration,
characters (refs, sub-characters, favorites, library import/export), cleanup, settings,
Piper model registry and helpers, the qualify / slicer / edit audio code, inflation sampling,
spoken-form text normalisation, the import staging flow, vc helpers (with the model call
faked), the model manager's eviction logic, rvc DSP helpers, auth middleware, and a
no-stale-references guard. Model-backed behaviour lives in test_zero_shot.py.

Everything runs in a temporary working directory, so the real outputs/, models/ and config/
are never touched.

  .venv/bin/python scripts/test_functions.py            # all
  .venv/bin/python scripts/test_functions.py -v         # one line per test
  .venv/bin/python scripts/test_functions.py TestPaths TestLibrary
  .venv/bin/python scripts/test_functions.py -k split   # name filter
Exit status is non-zero if anything fails.
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

# ── sandbox: modules bind relative paths ("outputs/…", "models/…", "faily/data/…") at import ──
SANDBOX = Path(tempfile.mkdtemp(prefix="faily_functest_"))
os.environ["FAILY_CONFIG_DIR"] = str(SANDBOX / "config")
os.environ.pop("FAILY_PASSWORD", None)
sys.path.insert(0, str(ROOT))
(SANDBOX / "faily").mkdir()
(SANDBOX / "faily" / "data").symlink_to(ROOT / "faily" / "data")      # the script bank is read relative to cwd
os.chdir(SANDBOX)

import numpy as np                                    # noqa: E402
import soundfile as sf                                # noqa: E402

import faily                                          # noqa: E402,F401  (sets HF_HOME etc.)
from faily.core import characters as C                # noqa: E402
from faily.core import cleanup as CL                  # noqa: E402
from faily.core import paths as P                     # noqa: E402
from faily.core import piper_models as PM             # noqa: E402
from faily.core import settings as S                  # noqa: E402

SR = 16000
OUT = Path("outputs")


# ── helpers ───────────────────────────────────────────────────────────────────

def speechish(seconds: float, sr: int = SR, level: float = 0.3, pauses=(), seed: int = 0) -> np.ndarray:
    """A voiced-sounding signal: a 150 Hz harmonic stack, syllable-rate modulated, over a quiet
    noise floor. `pauses` = [(start_s, end_s)] silences cut into it."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 150 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = sum(np.sin(k * phase) / k for k in range(1, 8))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t) ** 2
    x = level * sig * env / 2.0 + 0.0005 * rng.standard_normal(len(t))
    for a, b in pauses:
        x[int(a * sr):int(b * sr)] = 0.0005 * rng.standard_normal(max(int(b * sr) - int(a * sr), 0))
    return x.astype("float32")


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


def reset_outputs():
    shutil.rmtree(OUT, ignore_errors=True)
    for f in Path(os.environ["FAILY_CONFIG_DIR"]).glob("*"):         # keep the dir: modules hold paths into it
        f.unlink()
    P.ensure_layout()


def make_char(name: str, *, refs: int = 0, seconds: float = 3.0, inflated: int = 0, parent: str | None = None,
              level: float = 0.3) -> dict:
    """A saved character with a primary reference plus `refs` live and `inflated` generated clips."""
    tmp = SANDBOX / "_fixtures"
    base = write_wav(tmp / f"{name}_primary.wav", speechish(seconds, level=level))
    if parent:
        cfg = C.save_sub_character(name, parent, "xtts_v2", 0.7, 1.0)
    else:
        cfg = C.save_character(name, base, f"{name} primary line.")
    for i in range(refs):
        clip = write_wav(tmp / f"{name}_r{i}.wav", speechish(seconds, level=level, seed=i + 1))
        C.add_ref_clip(name, clip, f"{name} live line {i}.")
    for i in range(inflated):
        clip = write_wav(tmp / f"{name}_i{i}.wav", speechish(seconds, level=level, seed=100 + i))
        C.add_ref_clip(name, clip, f"{name} inflated line {i}.", source="inflated", category="neutral")
    return C.get_character(name) or cfg


class Sandboxed(unittest.TestCase):
    def setUp(self):
        reset_outputs()


# ═════════════════════════════════════════════════════════════════════════════
class TestPaths(Sandboxed):
    def test_layout_has_exactly_three_folders(self):
        self.assertEqual(sorted(p.name for p in OUT.iterdir()), ["characters", "foley", "misc"])

    def test_history_dir_saved_character(self):
        make_char("alice")
        self.assertEqual(P.history_dir("alice"), P.CHARACTERS_DIR / "alice" / "history")

    def test_history_dir_unknown_goes_to_misc(self):
        self.assertEqual(P.history_dir("nobody"), P.MISC_DIR)
        self.assertEqual(P.history_dir(None), P.MISC_DIR)

    def test_history_dir_ignores_staging_characters(self):
        d = P.CHARACTERS_DIR / "_stage_x"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        self.assertEqual(P.history_dir("_stage_x"), P.MISC_DIR)

    def test_history_dir_via_reference_owner(self):
        make_char("alice", refs=1)
        ref = P.CHARACTERS_DIR / "alice" / "refs" / "ref_001.wav"
        self.assertEqual(P.history_dir("unsaved name", ref), P.CHARACTERS_DIR / "alice" / "history")
        self.assertEqual(P.history_dir(None, SANDBOX / "elsewhere.wav"), P.MISC_DIR)

    def test_owner_of(self):
        make_char("alice")
        self.assertEqual(P.owner_of(P.CHARACTERS_DIR / "alice" / "ref.wav"), "alice")
        self.assertIsNone(P.owner_of(P.MISC_DIR / "x.wav"))
        self.assertIsNone(P.owner_of(None))

    def test_owner_from_filename_prefers_longest_prefix(self):
        names = ["e-girl", "e-girl-two"]
        self.assertEqual(P._owner_from_filename("e_girl_two_dark_can_001", names), "e-girl-two")
        self.assertEqual(P._owner_from_filename("piper_e-girl_m003_Hello", names), "e-girl")
        self.assertEqual(P._owner_from_filename("speak_e-girl_line", names), "e-girl")
        self.assertIsNone(P._owner_from_filename("random_clip_001", names))

    def test_migrate_legacy_layout_and_idempotent(self):
        make_char("e-girl")
        for rel in ["vc/e_girl_dark_can_001.wav", "vc/piper_e-girl_Hello.wav", "vc/other_clip_001.wav",
                    "vc/refs/up.wav", "vc/_import_tmp/a.wav", "tts/t.wav", "sfx/s.wav", "mix/m.wav",
                    "edit/_preview_1.wav", "base_voice_samples/lessac.wav", "_model_try/e-girl_m001.wav",
                    "_experiments/ab/x.onnx"]:
            write_wav(OUT / rel, speechish(0.2)) if rel.endswith(".wav") else (OUT / rel).parent.mkdir(parents=True, exist_ok=True) or (OUT / rel).write_text("x")
        moved = P.migrate_legacy_layout()
        self.assertGreater(moved, 0)
        chars = P.CHARACTERS_DIR / "e-girl"
        self.assertTrue((chars / "history" / "e_girl_dark_can_001.wav").exists())
        self.assertTrue((chars / "history" / "piper_e-girl_Hello.wav").exists())
        self.assertTrue((chars / "model_try" / "e-girl_m001.wav").exists())
        self.assertTrue((P.MISC_DIR / "other_clip_001.wav").exists())
        self.assertTrue((P.UPLOADS_DIR / "up.wav").exists())
        self.assertTrue((P.FOLEY_DIR / "s.wav").exists())
        self.assertTrue((P.MIX_DIR / "m.wav").exists())
        self.assertTrue((P.MISC_DIR / "t.wav").exists())
        self.assertTrue((P.MISC_DIR / "base_voice_samples" / "lessac.wav").exists())
        self.assertTrue((P.MISC_DIR / "_experiments").is_dir())
        self.assertTrue((P.TMP_DIR / "_import_tmp").exists())
        self.assertEqual(sorted(p.name for p in OUT.iterdir()), ["characters", "foley", "misc"])
        self.assertEqual(P.migrate_legacy_layout(), 0)

    def test_migrate_never_overwrites(self):
        make_char("alice")
        hist = P.CHARACTERS_DIR / "alice" / "history"
        write_wav(hist / "alice_clip_001.wav", speechish(0.2))
        write_wav(OUT / "vc" / "alice_clip_001.wav", speechish(0.3))
        P.migrate_legacy_layout()
        self.assertEqual(len(list(hist.glob("*.wav"))), 2)


# ═════════════════════════════════════════════════════════════════════════════
class TestCharacters(Sandboxed):
    def test_save_and_get(self):
        cfg = make_char("alice")
        self.assertEqual(cfg["name"], "alice")
        self.assertEqual([c["name"] for c in C.list_characters()], ["alice"])
        self.assertTrue((C.CHARACTERS_DIR / "alice" / cfg["ref_audio"]).exists())
        self.assertIsNone(C.get_character("ghost"))

    def test_list_hides_staging_and_broken(self):
        make_char("alice")
        (C.CHARACTERS_DIR / "_stage_1").mkdir()
        (C.CHARACTERS_DIR / "_stage_1" / "config.json").write_text("{}")
        (C.CHARACTERS_DIR / "broken").mkdir()
        (C.CHARACTERS_DIR / "broken" / "config.json").write_text("not json")
        self.assertEqual([c["name"] for c in C.list_characters()], ["alice"])

    def test_add_ref_clip_numbers_and_tags(self):
        make_char("alice", refs=2, inflated=1)
        cfg = C.get_character("alice")
        self.assertEqual([rc["file"] for rc in cfg["ref_clips"]], ["refs/ref_001.wav", "refs/ref_002.wav", "refs/ref_003.wav"])
        self.assertEqual(cfg["ref_clips"][2]["source"], "inflated")
        with self.assertRaises(FileNotFoundError):
            C.add_ref_clip("ghost", SANDBOX / "x.wav")

    def test_chain_order_and_live_only(self):
        make_char("alice", refs=2, inflated=2)
        chain = C.get_ref_chain("alice")
        self.assertEqual(len(chain), 5)
        self.assertEqual(chain[0]["file_key"], None)             # primary first
        self.assertEqual(len(C.get_ref_chain("alice", live_only=True)), 3)

    def test_exclude_and_include_excluded(self):
        make_char("alice", refs=2)
        C.set_ref_clip_excluded("alice", "refs/ref_001.wav", True)
        self.assertEqual(len(C.get_ref_chain("alice")), 2)
        self.assertEqual(len(C.get_ref_chain("alice", include_excluded=True)), 3)
        C.set_ref_clip_excluded("alice", "refs/ref_001.wav", False)
        self.assertEqual(len(C.get_ref_chain("alice")), 3)

    def test_sub_character_inherits_parent_chain(self):
        make_char("alice", refs=1)
        make_char("alice-angry", parent="alice")
        chain = C.get_ref_chain("alice-angry")
        self.assertEqual(len(chain), 2)
        self.assertTrue(all(e["owner"] == "alice" for e in chain))
        self.assertEqual(C.get_ref_path("alice-angry"), C.get_ref_path("alice"))

    def test_chain_survives_parent_cycle(self):
        make_char("a")
        make_char("b", parent="a")
        cfg = json.loads(C._cfg("a").read_text()); cfg["parent"] = "b"
        C._cfg("a").write_text(json.dumps(cfg))
        self.assertTrue(C.get_ref_chain("a") is not None)         # no infinite loop

    def test_remove_ref_clip_deletes_file(self):
        make_char("alice", refs=1)
        f = C.CHARACTERS_DIR / "alice" / "refs" / "ref_001.wav"
        self.assertTrue(f.exists())
        C.remove_ref_clip("alice", "refs/ref_001.wav")
        self.assertFalse(f.exists())
        self.assertEqual(C.get_character("alice")["ref_clips"], [])

    def test_update_ref_clip_rename_and_transcript(self):
        make_char("alice", refs=1)
        C.update_ref_clip("alice", "refs/ref_001.wav", new_stem="renamed", new_transcript="new words")
        rc = C.get_character("alice")["ref_clips"][0]
        self.assertEqual(rc["file"], "refs/renamed.wav")
        self.assertEqual(rc["transcript"], "new words")
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "refs" / "renamed.wav").exists())

    def test_rename_ref_audio(self):
        make_char("alice")
        new = C.rename_ref_audio("alice", "voice")
        self.assertEqual(new.name, "voice.wav")
        self.assertEqual(C.get_character("alice")["ref_audio"], "voice.wav")
        with self.assertRaises(FileExistsError):
            make_char("bob")
            write_wav(C.CHARACTERS_DIR / "bob" / "taken.wav", speechish(0.2))
            C.rename_ref_audio("bob", "taken")

    def test_metadata_only_touches_safe_fields(self):
        make_char("alice")
        cfg = C.update_character_metadata("alice", {"transcript": "hi", "name": "hacked", "ref_audio": "x", "speed": 1.2})
        self.assertEqual(cfg["transcript"], "hi")
        self.assertEqual(cfg["name"], "alice")
        self.assertNotEqual(cfg["ref_audio"], "x")
        with self.assertRaises(FileNotFoundError):
            C.update_character_metadata("ghost", {})

    def test_favorites_and_clips(self):
        make_char("alice")
        clip = write_wav(SANDBOX / "gen.wav", speechish(0.5))
        C.add_clip_to_favorites("alice", clip)
        C.add_clip_to_character("alice", clip)
        self.assertEqual([p.name for p in C.list_character_favorites("alice")], ["gen.wav"])
        self.assertEqual([p.name for p in C.list_character_clips("alice")], ["gen.wav"])
        new = C.rename_character_file("alice", "favorites", "gen.wav", "best")
        self.assertEqual(new.name, "best.wav")
        with self.assertRaises(FileNotFoundError):
            C.rename_character_file("alice", "favorites", "gen.wav", "x")
        self.assertEqual(C.list_character_favorites("ghost"), [])

    def test_delete_character(self):
        make_char("alice")
        C.delete_character("alice")
        self.assertIsNone(C.get_character("alice"))
        C.delete_character("alice")                                # idempotent

    def test_set_models(self):
        make_char("alice")
        self.assertEqual(C.set_piper_model("alice", "x/model.onnx")["piper_model"], "x/model.onnx")
        self.assertEqual(C.set_rvc_model("alice", "x/m.npz")["rvc_model"], "x/m.npz")

    def test_migrate_legacy_tags(self):
        make_char("alice", refs=2)
        cfg = C.get_character("alice")
        cfg["ref_clips"][0]["source"] = "buffer"
        C._cfg("alice").write_text(json.dumps(cfg))
        self.assertEqual(C.migrate_legacy_tags(), 1)
        self.assertEqual(C.get_character("alice")["ref_clips"][0]["source"], "inflated")
        self.assertEqual(C.migrate_legacy_tags(), 0)

    def test_clip_quality_issues(self):
        natural = speechish(3.0, pauses=[(1.0, 1.4)])              # real speech has pauses to measure the noise floor from
        self.assertEqual(C.clip_quality_issues(write_wav(SANDBOX / "ok.wav", natural)), [])
        self.assertTrue(any("short" in i for i in C.clip_quality_issues(write_wav(SANDBOX / "s.wav", speechish(0.4)))))
        self.assertIn("clipping", C.clip_quality_issues(write_wav(SANDBOX / "c.wav", np.clip(speechish(3.0) * 6, -1, 1))))
        self.assertIn("very quiet", C.clip_quality_issues(write_wav(SANDBOX / "q.wav", speechish(3.0, level=0.004))))
        noisy = speechish(3.0, pauses=[(1.0, 1.4)]) + 0.05 * np.random.default_rng(1).standard_normal(3 * SR).astype("float32")
        self.assertTrue(any("noisy" in i for i in C.clip_quality_issues(write_wav(SANDBOX / "n.wav", noisy))))

    def test_split_long_ref_clip(self):
        make_char("alice")
        x = speechish(24.0, pauses=[(6.0, 6.5), (12.0, 12.6), (18.0, 18.5)])
        long_clip = write_wav(SANDBOX / "long.wav", x)
        C.add_ref_clip("alice", long_clip, "One sentence here. Two sentences there. Three is the charm. And four to finish.",
                       source="inflated", category="neutral")
        n = C.split_long_ref_clip("alice", "refs/ref_001.wav", 6.0)
        self.assertGreaterEqual(n, 2)
        clips = C.get_character("alice")["ref_clips"]
        self.assertEqual(len(clips), n)
        self.assertTrue(all(rc["source"] == "inflated" and rc["transcript"] for rc in clips))
        self.assertTrue(all((C.CHARACTERS_DIR / "alice" / rc["file"]).exists() for rc in clips))

    def test_split_needs_two_sentences(self):
        make_char("alice")
        C.add_ref_clip("alice", write_wav(SANDBOX / "l.wav", speechish(20.0)), "Just one long sentence")
        self.assertEqual(C.split_long_ref_clip("alice", "refs/ref_001.wav", 6.0), 0)
        with self.assertRaises(ValueError):
            C.split_long_ref_clip("alice", "refs/nope.wav", 6.0)


class TestBuildRefAudio(Sandboxed):
    def test_single_clip_returned_directly(self):
        make_char("alice", seconds=4.0)
        with C.build_ref_audio("alice") as (ref, text):
            self.assertEqual(ref, C.get_ref_path("alice"))
            self.assertIn("primary", text)

    def test_concatenates_and_cleans_up_temp(self):
        make_char("alice", refs=3, seconds=4.0)
        with C.build_ref_audio("alice") as (ref, text):
            self.assertTrue(ref.exists())
            self.assertAlmostEqual(sf.info(str(ref)).duration, 16.0, delta=0.2)
            self.assertEqual(len(text.split(" line ")), 4)
            tmp = ref
        self.assertFalse(tmp.exists())

    def test_default_cap_bounds_total_duration(self):
        make_char("alice", refs=30, seconds=4.0)                  # 124 s of audio
        with C.build_ref_audio("alice") as (ref, _):
            self.assertLessEqual(sf.info(str(ref)).duration, C.DEFAULT_REF_SECONDS + 0.2)
            self.assertGreater(sf.info(str(ref)).duration, 20)

    def test_explicit_cap_and_f5_cap(self):
        make_char("alice", refs=10, seconds=4.0)
        with C.build_ref_audio("alice", max_duration=C.F5_REF_SECONDS) as (ref, _):
            self.assertLessEqual(sf.info(str(ref)).duration, C.F5_REF_SECONDS + 0.2)

    def test_overlong_clips_are_skipped_not_truncated(self):
        make_char("alice", seconds=18.0)                          # primary is 18 s: over an 11 s cap
        for i in range(3):
            C.add_ref_clip("alice", write_wav(SANDBOX / f"s{i}.wav", speechish(4.0, seed=i)), f"short {i}.")
        with C.build_ref_audio("alice", max_duration=11.0) as (ref, text):
            self.assertLessEqual(sf.info(str(ref)).duration, 11.2)
            self.assertNotIn("primary", text)                     # transcript matches the audio actually used

    def test_single_enormous_clip_is_trimmed_and_transcript_dropped(self):
        make_char("alice", seconds=200.0)
        with C.build_ref_audio("alice", max_duration=11.0) as (ref, text):
            self.assertLessEqual(sf.info(str(ref)).duration, 11.2)
            self.assertEqual(text, "")
            tmp = ref
        self.assertFalse(tmp.exists())

    def test_require_transcript_drops_untranscribed(self):
        make_char("alice", refs=1)
        C.add_ref_clip("alice", write_wav(SANDBOX / "n.wav", speechish(2.0)), "")
        with C.build_ref_audio("alice", require_transcript=True, max_duration=0) as (ref, text):
            self.assertNotIn("  ", text)
            self.assertAlmostEqual(sf.info(str(ref)).duration, 6.0, delta=0.2)   # the untranscribed 2 s clip left out

    def test_mixed_sample_rates_are_matched(self):
        make_char("alice")
        C.add_ref_clip("alice", write_wav(SANDBOX / "hi.wav", speechish(2.0, sr=24000, seed=3), sr=24000), "hi.")
        with C.build_ref_audio("alice") as (ref, _):
            info = sf.info(str(ref))
            self.assertEqual(info.samplerate, SR)
            self.assertAlmostEqual(info.duration, 5.0, delta=0.2)

    def test_no_clips_yields_none(self):
        make_char("alice")
        cfg = C.get_character("alice"); cfg.pop("ref_audio"); C._cfg("alice").write_text(json.dumps(cfg))
        with C.build_ref_audio("alice") as (ref, text):
            self.assertIsNone(ref)
            self.assertEqual(text, "")


# ═════════════════════════════════════════════════════════════════════════════
class TestLibrary(Sandboxed):
    def setUp(self):
        super().setUp()
        self.other = SANDBOX / "other_machine" / "outputs"
        shutil.rmtree(self.other.parent, ignore_errors=True)
        lib = self.other / "characters"
        lib.mkdir(parents=True)
        # build a library elsewhere by pointing the module at it for a moment
        with mock.patch.object(C, "CHARACTERS_DIR", lib):
            make_char("alice", refs=1)
            make_char("alice-angry", parent="alice")
            make_char("bob")
        (lib / "alice" / "models" / "m001").mkdir(parents=True)
        (lib / "alice" / "models" / "m001" / "model.onnx").write_text("x")
        (lib / "alice" / "piper_train").mkdir()
        (lib / "alice" / "piper_train" / "ckpt").write_text("x")
        cfg = json.loads((lib / "alice" / "config.json").read_text())
        cfg["piper_model"] = "C:\\old\\outputs\\characters\\alice\\models\\m001\\model.onnx"
        (lib / "alice" / "config.json").write_text(json.dumps(cfg))

    def test_resolve_accepts_outputs_or_characters_folder(self):
        self.assertEqual(C.resolve_library(self.other), self.other / "characters")
        self.assertEqual(C.resolve_library(self.other / "characters"), self.other / "characters")
        with self.assertRaises(FileNotFoundError):
            C.resolve_library(SANDBOX / "nope")

    def test_scan(self):
        found = {c["name"]: c for c in C.scan_library(self.other)}
        self.assertEqual(sorted(found), ["alice", "alice-angry", "bob"])
        self.assertTrue(found["alice"]["has_model"])
        self.assertEqual(found["alice-angry"]["parent"], "alice")
        self.assertFalse(found["bob"]["here"])
        make_char("bob")
        self.assertTrue({c["name"]: c for c in C.scan_library(self.other)}["bob"]["here"])

    def test_import_brings_ancestors_and_repoints_models(self):
        res = C.import_characters(self.other, ["alice-angry"])
        self.assertEqual(res["imported"], ["alice", "alice-angry"])
        cfg = C.get_character("alice")
        self.assertEqual(cfg["piper_model"], "outputs/characters/alice/models/m001/model.onnx")
        self.assertTrue(Path(cfg["piper_model"]).exists())

    def test_import_skips_existing_unless_overwrite(self):
        make_char("bob")
        self.assertEqual(C.import_characters(self.other, ["bob"]), {"imported": [], "skipped": ["bob"]})
        (C.CHARACTERS_DIR / "bob" / "marker").write_text("x")
        self.assertEqual(C.import_characters(self.other, ["bob"], overwrite=True)["imported"], ["bob"])
        self.assertFalse((C.CHARACTERS_DIR / "bob" / "marker").exists())

    def test_import_can_leave_training_data_behind(self):
        C.import_characters(self.other, ["alice"], include_training=False)
        self.assertFalse((C.CHARACTERS_DIR / "alice" / "piper_train").exists())
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "models").exists())
        shutil.rmtree(C.CHARACTERS_DIR / "alice")
        C.import_characters(self.other, ["alice"])
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "piper_train").exists())

    def test_import_from_self_refused(self):
        with self.assertRaises(ValueError):
            C.import_characters(C.CHARACTERS_DIR, ["x"])

    def test_export_roundtrip(self):
        make_char("carol", refs=1)
        dest = SANDBOX / "backup"
        self.assertEqual(C.export_characters(dest, ["carol"])["exported"], ["carol"])
        self.assertEqual(C.export_characters(dest, ["carol"])["skipped"], ["carol"])
        self.assertTrue((dest / "carol" / "config.json").exists())
        self.assertEqual([c["name"] for c in C.scan_library(dest)], ["carol"])
        with self.assertRaises(ValueError):
            C.export_characters(C.CHARACTERS_DIR, ["carol"])

    def test_repoint_models_variants(self):
        cfg = {"piper_model": "/mnt/x/outputs/characters/old/models/m002/model.onnx", "rvc_model": "weird/path.npz"}
        C._repoint_models(cfg, "old", "new")
        self.assertEqual(cfg["piper_model"], "outputs/characters/new/models/m002/model.onnx")
        self.assertEqual(cfg["rvc_model"], "weird/path.npz")        # no recognisable marker: left alone


# ═════════════════════════════════════════════════════════════════════════════
class TestCleanup(Sandboxed):
    def _populate(self):
        make_char("alice", refs=1)
        d = C.CHARACTERS_DIR / "alice"
        write_wav(d / "history" / "h.wav", speechish(1.0))
        write_wav(d / "favorites" / "f.wav", speechish(1.0))
        write_wav(d / "model_try" / "t.wav", speechish(1.0))
        write_wav(d / "refs_trash" / "x.wav", speechish(1.0))
        write_wav(d / "refs_original" / "o.wav", speechish(1.0))
        (d / "piper_train").mkdir(); (d / "piper_train" / "c").write_bytes(b"x" * 1000)
        for mid in ("m001", "m002"):
            (d / "models" / mid).mkdir(parents=True)
            (d / "models" / mid / "model.onnx").write_bytes(b"x" * 100)
        C.set_piper_model("alice", str(d / "models" / "m002" / "model.onnx"))
        write_wav(P.MISC_DIR / "loose.wav", speechish(1.0))
        write_wav(P.TMP_DIR / "s" / "a.wav", speechish(1.0))
        write_wav(P.FOLEY_DIR / "sfx.wav", speechish(1.0))

    def test_open_library_validation(self):
        with self.assertRaises(FileNotFoundError):
            CL.open_library(SANDBOX / "ghost")
        empty = SANDBOX / "empty"; empty.mkdir()
        with self.assertRaises(ValueError):
            CL.open_library(empty)
        make_char("alice")
        self.assertIsNotNone(CL.open_library(OUT).characters)
        self.assertIsNotNone(CL.open_library(OUT / "characters").characters)
        self.assertIsNone(CL.open_library(OUT / "characters").foley)

    def test_scan_reports_sizes(self):
        self._populate()
        by = {c["key"]: c for c in CL.scan(OUT)}
        self.assertGreater(by["history"]["bytes"], 0)
        self.assertEqual(by["old_models"]["items"], 1)              # m001 only: m002 is active
        self.assertEqual(by["training"]["bytes"], 1000)
        self.assertEqual(by["foley"]["items"], 1)
        self.assertEqual(by["stale"]["items"], 0)

    def test_clean_history_keeps_favorites_and_refs(self):
        self._populate()
        res = CL.clean(OUT, ["history", "model_try", "tmp"])
        d = C.CHARACTERS_DIR / "alice"
        self.assertFalse((d / "history").exists())
        self.assertFalse((d / "model_try").exists())
        self.assertFalse(P.TMP_DIR.exists())
        self.assertTrue((d / "favorites" / "f.wav").exists())
        self.assertTrue((d / "refs" / "ref_001.wav").exists())
        self.assertGreater(res["freed"], 0)

    def test_unused_models_never_removes_active(self):
        self._populate()
        CL.clean(OUT, ["old_models"])
        d = C.CHARACTERS_DIR / "alice" / "models"
        self.assertTrue((d / "m002").exists())
        self.assertFalse((d / "m001").exists())

    def test_active_training_is_protected(self):
        self._populate()
        from faily.modules import piper
        piper._write_status(C.CHARACTERS_DIR / "alice", state="training")
        CL.clean(OUT, ["training", "old_models"])
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "piper_train").exists())
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "models" / "m001").exists())

    def test_misc_and_foley(self):
        self._populate()
        CL.clean(OUT, ["misc", "foley"])
        self.assertFalse((P.MISC_DIR / "loose.wav").exists())
        self.assertEqual(list(P.FOLEY_DIR.iterdir()), [])

    def test_unknown_keys_do_nothing_and_human(self):
        self._populate()
        self.assertEqual(CL.clean(OUT, ["bogus"]), {"freed": 0, "removed": 0})
        self.assertEqual(CL.human(500), "500 B")
        self.assertEqual(CL.human(2048), "2.0 KB")
        self.assertEqual(CL.human(5 * 2**30), "5.0 GB")


# ═════════════════════════════════════════════════════════════════════════════
class TestSettings(Sandboxed):
    def test_defaults(self):
        self.assertEqual(S.get_piper_epochs(), 1000)
        self.assertEqual(S.get_zero_shot_reference(), "all")
        self.assertEqual(S.get_piper_train_data(), "mixed")
        self.assertAlmostEqual(S.get_piper_learning_rate(), 5e-5)

    def test_roundtrip(self):
        S.set_piper_epochs(42); S.set_piper_learning_rate(1e-4); S.set_zero_shot_reference("live")
        S.set_default_base_voice("lessac"); S.set_piper_train_data("recordings")
        S.set_piper_match("rhythm", "reference")
        self.assertEqual(S.get_piper_epochs(), 42)
        self.assertAlmostEqual(S.get_piper_learning_rate(), 1e-4)
        self.assertEqual(S.get_zero_shot_reference(), "live")
        self.assertEqual(S.get_default_base_voice(), "lessac")
        self.assertEqual(S.get_piper_train_data(), "recordings")
        self.assertEqual(S.get_piper_match()["rhythm"], "reference")

    def test_corrupt_file_falls_back_to_defaults(self):
        S._SETTINGS_FILE.write_text("{{{")
        self.assertEqual(S.load_settings()["piper_epochs"], 1000)

    def test_legacy_padding_keys_migrate(self):
        S._SETTINGS_FILE.write_text(json.dumps({"piper_use_padding": False}))
        s = S.load_settings()
        self.assertEqual(s["piper_train_data"], "recordings")
        S._SETTINGS_FILE.write_text(json.dumps({"piper_use_padding": True}))
        self.assertEqual(S.load_settings()["piper_train_data"], "mixed")

    def test_hf_token(self):
        from faily.core import hf_token as H
        self.assertFalse(H.has_hf_token())
        H.save_hf_token("hf_abc")
        self.assertTrue(H.has_hf_token())
        self.assertEqual(H.get_hf_token(), "hf_abc")
        H.clear_hf_token()
        self.assertIsNone(H.get_hf_token())


# ═════════════════════════════════════════════════════════════════════════════
class TestPiperModels(Sandboxed):
    def _fake_model(self, tag="a"):
        d = SANDBOX / f"train_{tag}"; d.mkdir(exist_ok=True)
        (d / "m.onnx").write_text(tag); (d / "m.onnx.json").write_text("{}")
        (d / "m.ckpt").write_text("ckpt" + tag)
        return d / "m.onnx", d / "m.ckpt"

    def test_save_activate_list(self):
        make_char("alice")
        onnx, ckpt = self._fake_model("1")
        m = PM.save_model("alice", onnx=onnx, ckpt=ckpt, meta={"epoch": 10})
        self.assertEqual(m["id"], "m001")
        self.assertTrue(m["active"])
        self.assertEqual(m["name"], "Model 1")
        self.assertIsNotNone(m["ckpt"])
        self.assertEqual(PM.active_model_id("alice"), "m001")
        onnx2, _ = self._fake_model("2")
        m2 = PM.save_model("alice", onnx=onnx2, ckpt=None, meta={"epoch": 20}, activate=False)
        self.assertEqual(m2["id"], "m002")
        self.assertFalse(m2["active"])
        self.assertEqual([x["id"] for x in PM.list_models("alice")], ["m002", "m001"])   # newest first

    def test_set_active_and_missing(self):
        make_char("alice")
        PM.save_model("alice", onnx=self._fake_model("1")[0], ckpt=None, meta={})
        PM.save_model("alice", onnx=self._fake_model("2")[0], ckpt=None, meta={})
        PM.set_active("alice", "m001")
        self.assertEqual(PM.active_model_id("alice"), "m001")
        with self.assertRaises(FileNotFoundError):
            PM.set_active("alice", "m099")

    def test_delete_active_reassigns(self):
        make_char("alice")
        PM.save_model("alice", onnx=self._fake_model("1")[0], ckpt=None, meta={})
        PM.save_model("alice", onnx=self._fake_model("2")[0], ckpt=None, meta={})
        PM.delete_model("alice", "m002")
        self.assertEqual(PM.active_model_id("alice"), "m001")
        PM.delete_model("alice", "m001")
        self.assertIsNone(PM.active_model_id("alice"))
        self.assertEqual(C.get_character("alice")["piper_model"], "")

    def test_replace_keeps_name_and_activity(self):
        make_char("alice")
        PM.save_model("alice", onnx=self._fake_model("1")[0], ckpt=None, meta={"name": "Keeper"})
        new = PM.save_model("alice", onnx=self._fake_model("2")[0], ckpt=None, meta={}, activate=False, replace_id="m001")
        self.assertEqual(new["name"], "Keeper")
        self.assertTrue(new["active"])
        self.assertEqual([m["id"] for m in PM.list_models("alice")], [new["id"]])

    def test_update_meta_and_delete_checkpoint(self):
        make_char("alice")
        onnx, ckpt = self._fake_model("1")
        PM.save_model("alice", onnx=onnx, ckpt=ckpt, meta={})
        PM.update_meta("alice", "m001", name="Renamed", note="x")
        self.assertEqual(PM.get_model("alice", "m001")["name"], "Renamed")
        PM.delete_checkpoint("alice", "m001")
        self.assertIsNone(PM.get_model("alice", "m001")["ckpt"])
        self.assertIsNone(PM.get_model("alice", "m404"))

    def test_failed_save_leaves_nothing_behind(self):
        make_char("alice")
        with self.assertRaises(FileNotFoundError):
            PM.save_model("alice", onnx=SANDBOX / "missing.onnx", ckpt=None, meta={})
        self.assertEqual(PM.list_models("alice"), [])
        self.assertEqual(list(PM.models_root("alice").glob("*")), [])


class TestPiperHelpers(Sandboxed):
    def setUp(self):
        super().setUp()
        from faily.modules import piper
        self.piper = piper

    def test_match_aspects_and_frozen_modules(self):
        p = self.piper
        self.assertEqual(set(p.DEFAULT_MATCH), set(p.MATCH_ASPECTS))
        self.assertEqual(set(S._DEFAULTS[f"piper_match_{a}"] for a in p.MATCH_ASPECTS), set(p.DEFAULT_MATCH.values()))
        all_ref = {a: "reference" for a in p.MATCH_ASPECTS}
        self.assertEqual(p.frozen_modules(all_ref), "")
        frozen = p.frozen_modules({**all_ref, "rhythm": "base"}).split(",")
        self.assertEqual(sorted(frozen), sorted(p.MATCH_ASPECTS["rhythm"]["modules"]))
        with self.assertRaises(ValueError):
            p.frozen_modules({a: "base" for a in p.MATCH_ASPECTS})

    def test_filter_training_clips(self):
        chain = [{"source": ""}, {"source": "inflated"}, {"source": "import"}]
        f = self.piper.filter_training_clips
        self.assertEqual(len(f(chain, "mixed")), 3)
        self.assertEqual(len(f(chain, "recordings")), 2)
        self.assertEqual([c["source"] for c in f(chain, "inflated")], ["inflated"])

    def test_sentences(self):
        self.assertEqual(self.piper._sentences("Hi there. How are you? Fine!\nNext line"),
                         ["Hi there.", "How are you?", "Fine!", "Next line"])
        self.assertEqual(self.piper._sentences("   "), [])

    def test_bad_take_detection(self):
        sr = 22050
        bt = self.piper._bad_take
        good = speechish(2.0, sr=sr)
        self.assertIsNone(bt(good, sr, 30, 1.0))
        self.assertEqual(bt(np.zeros(100, dtype="float32"), sr, 30, 1.0), "empty")
        self.assertIn("runaway", bt(speechish(30.0, sr=sr), sr, 10, 1.0))
        noise = (0.3 * np.random.default_rng(0).standard_normal(sr * 2)).astype("float32")
        self.assertEqual(bt(noise, sr, 40, 1.0), "white-noise burst")
        clipped = np.clip(speechish(2.0, sr=sr) * 20, -1, 1)
        self.assertEqual(bt(clipped.astype("float32"), sr, 40, 1.0), "clipping")

    def test_training_status_roundtrip_and_stale_detection(self):
        make_char("alice")
        d = C.CHARACTERS_DIR / "alice"
        self.assertIsNone(self.piper.read_training_status(d))
        self.piper._write_status(d, state="training", epoch=3)
        st = self.piper.read_training_status(d)
        self.assertEqual((st["state"], st["epoch"]), ("training", 3))
        self.assertEqual(self.piper.training_in_progress(C.CHARACTERS_DIR), "alice")
        # a run recorded by a previous server process is reported as interrupted
        raw = json.loads(self.piper._status_path(d).read_text()); raw["server"] = "old-process"
        self.piper._status_path(d).write_text(json.dumps(raw))
        self.assertEqual(self.piper.read_training_status(d)["state"], "interrupted")
        self.assertIsNone(self.piper.training_in_progress(C.CHARACTERS_DIR))

    def test_request_stop(self):
        make_char("alice")
        d = C.CHARACTERS_DIR / "alice"
        with mock.patch.object(self.piper, "stop_patch_applied", return_value=False):
            self.assertFalse(self.piper.request_stop(d))
        with mock.patch.object(self.piper, "stop_patch_applied", return_value=True):
            self.assertTrue(self.piper.request_stop(d))
        self.assertTrue((d / "piper_train" / "STOP_REQUEST").exists())
        self.assertTrue(self.piper.read_training_status(d)["stopping"])

    def test_list_candidates_sorted_by_epoch(self):
        make_char("alice")
        d = C.CHARACTERS_DIR / "alice"
        for e in (120, 7, 55):
            (d / "piper_candidates").mkdir(exist_ok=True)
            (d / "piper_candidates" / f"epoch_{e:05d}.onnx").write_text("x")
        (d / "piper_candidates" / "notes.txt").write_text("x")
        self.assertEqual([e for e, _ in self.piper.list_candidates(d)], [7, 55, 120])

    def test_read_loss_history_empty(self):
        make_char("alice")
        h = self.piper.read_loss_history(C.CHARACTERS_DIR / "alice")
        self.assertIsInstance(h, dict)


# ═════════════════════════════════════════════════════════════════════════════
class TestQualify(Sandboxed):
    def setUp(self):
        super().setUp()
        from faily.modules import qualify
        self.Q = qualify

    def test_normalize_words(self):
        n = self.Q.normalize_words
        self.assertEqual(n("Um, I'm gonna go-to the store."), n("i'm going to go to the store"))
        self.assertEqual(n("Hmm... uh, okay."), ["okay"] if "okay" in n("okay") else n("okay"))
        self.assertEqual(n("It’s 5 p.m."), n("it's 5pm"))

    def test_has_filler(self):
        self.assertTrue(self.Q.has_filler("well, um, yes"))
        self.assertTrue(self.Q.has_filler("mhm"))
        self.assertFalse(self.Q.has_filler("a clean sentence"))

    def test_word_error_rate(self):
        w = self.Q.word_error_rate
        self.assertEqual(w(["a", "b", "c"], ["a", "b", "c"]), 0.0)
        self.assertAlmostEqual(w(["a", "b", "c", "d"], ["a", "x", "c", "d"]), 0.25)
        self.assertEqual(w([], []), 0.0)
        self.assertEqual(w([], ["a"]), 1.0)
        self.assertEqual(w(["a", "b"], []), 1.0)

    def test_count_and_diff_words(self):
        self.assertEqual(self.Q.count_differences("Hello there friend", "hello there, friend"), 0)
        self.assertEqual(self.Q.count_differences("hello there friend", "hello big friend"), 1)
        d = self.Q.diff_words("hello there friend", "hello big new friend")
        self.assertEqual([w for w, _ in d], ["hello", "big", "new", "friend"])
        self.assertEqual(d[0][1], "same")
        self.assertIn(d[1][1], ("added", "changed"))

    def test_process_audio_gain_trim_fade_mute(self):
        x = speechish(2.0)
        y = self.Q.process_audio(x, SR, gain_db=6.0)
        self.assertAlmostEqual(float(np.abs(y).max() / np.abs(x).max()), 2.0, delta=0.05)
        self.assertEqual(len(self.Q.process_audio(x, SR, trim_start=0.5, trim_end=1.5)), SR)
        self.assertTrue(np.abs(self.Q.process_audio(x * 5, SR, gain_db=20)).max() <= 1.0)
        f = self.Q.process_audio(np.ones(SR, dtype="float32"), SR, fade=(0.1, 0.1))
        self.assertLess(f[0], 0.01); self.assertLess(f[-1], 0.01); self.assertAlmostEqual(float(f[SR // 2]), 1.0, places=3)
        m = self.Q.process_audio(np.ones(SR, dtype="float32"), SR, mute=((0.4, 0.6),))
        self.assertLess(abs(float(m[SR // 2])), 0.01)
        self.assertAlmostEqual(float(m[int(0.1 * SR)]), 1.0, places=3)
        self.assertEqual(len(m), SR)                                # muting never shifts timing

    def test_speech_level_tracks_gain(self):
        a = self.Q.speech_rms_db(speechish(3.0, level=0.1), SR)
        b = self.Q.speech_rms_db(speechish(3.0, level=0.2), SR)
        self.assertAlmostEqual(b - a, 6.0, delta=1.0)

    def test_analyze_volume(self):
        ok = write_wav(SANDBOX / "ok.wav", speechish(3.0, level=0.15))
        level = self.Q.speech_rms_db(speechish(3.0, level=0.15), SR)
        self.assertTrue(self.Q.analyze_volume(ok, level)["ok"])
        r = self.Q.analyze_volume(ok, level + 9.0)                  # set is much louder than this clip
        self.assertFalse(r["ok"]); self.assertGreater(r["gain_db"], 0)
        r = self.Q.analyze_volume(ok, level - 9.0)
        self.assertFalse(r["ok"]); self.assertLess(r["gain_db"], 0)
        clipped = write_wav(SANDBOX / "c.wav", np.clip(speechish(3.0) * 8, -1, 1))
        self.assertTrue(any("clipping" in i for i in self.Q.analyze_volume(clipped, -20)["issues"]))

    def test_analyze_trim(self):
        x = np.concatenate([np.zeros(int(1.5 * SR), "float32") + 0.0005, speechish(2.0), np.zeros(int(1.5 * SR), "float32") + 0.0005])
        r = self.Q.analyze_trim(write_wav(SANDBOX / "t.wav", x))
        self.assertFalse(r["ok"])
        self.assertGreater(r["trim_start"], 0.8); self.assertLess(r["trim_end"], 4.5)
        tight = self.Q.analyze_trim(write_wav(SANDBOX / "tight.wav", speechish(2.0)))
        self.assertTrue(tight["ok"])
        self.assertTrue(self.Q.analyze_trim(write_wav(SANDBOX / "tiny.wav", speechish(0.3)))["ok"])

    def test_analyze_transcript_with_fake_asr(self):
        wav = write_wav(SANDBOX / "a.wav", speechish(2.0))
        same = self.Q.analyze_transcript(wav, "the quick brown fox", lambda x, sr: "The quick brown fox.")
        diff = self.Q.analyze_transcript(wav, "the quick brown fox", lambda x, sr: "a slow red dog")
        self.assertTrue(same is None or same.get("ok", True))
        self.assertTrue(diff is not None and not diff.get("ok", False))

    def test_own_clips_skip_inherited(self):
        make_char("alice", refs=1)
        make_char("kid", parent="alice")
        self.assertEqual(len(self.Q.own_clips("alice")), 2)
        self.assertEqual(self.Q.own_clips("kid"), [])

    def test_apply_audio_fix_backs_up_original(self):
        make_char("alice", refs=1)
        clip = self.Q.own_clips("alice")[1]
        before = sf.info(str(clip.audio)).duration
        self.Q.apply_audio_fix("alice", clip, gain_db=-6.0, trim_start=0.5)
        self.assertAlmostEqual(sf.info(str(clip.audio)).duration, before - 0.5, delta=0.01)
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "refs_original" / clip.audio.name).exists())
        self.assertIs(self.Q.apply_audio_fix("alice", clip), clip)            # no-op

    def test_set_transcript_logs_change(self):
        make_char("alice", refs=1)
        clip = self.Q.own_clips("alice")[1]
        self.Q.set_transcript("alice", clip, "  new   words  ")
        self.assertEqual(C.get_character("alice")["ref_clips"][0]["transcript"], "new words")
        log = json.loads((C.CHARACTERS_DIR / "alice" / "refs_original" / "transcript_changes.json").read_text())
        self.assertEqual(log[0]["after"], "new words")
        prim = self.Q.own_clips("alice")[0]
        self.Q.set_transcript("alice", prim, "primary now")
        self.assertEqual(C.get_character("alice")["transcript"], "primary now")

    def test_delete_clip_goes_to_trash_and_primary_is_protected(self):
        make_char("alice", refs=1)
        clips = self.Q.own_clips("alice")
        self.assertFalse(self.Q.delete_clip("alice", clips[0]))
        self.assertTrue(self.Q.delete_clip("alice", clips[1]))
        self.assertTrue((C.CHARACTERS_DIR / "alice" / "refs_trash" / clips[1].audio.name).exists())
        self.assertEqual(C.get_character("alice")["ref_clips"], [])

    def test_delete_inflated_clips(self):
        make_char("alice", refs=1, inflated=3)
        self.assertEqual(self.Q.delete_inflated_clips("alice"), 3)
        self.assertEqual(len(C.get_character("alice")["ref_clips"]), 1)
        self.assertEqual(self.Q.delete_inflated_clips("alice"), 0)


# ═════════════════════════════════════════════════════════════════════════════
class TestSlicerAndEdit(Sandboxed):
    def test_presets(self):
        from faily.modules import slicer
        for target, lo, hi in slicer.PRESETS.values():
            self.assertTrue(lo < target < hi)

    def test_find_slices_respects_bounds(self):
        from faily.modules import slicer
        pauses = [(i * 4.0 + 3.4, i * 4.0 + 4.0) for i in range(6)]
        x = speechish(24.0, pauses=pauses)
        sl = slicer.find_slices(x, SR, target=7.0, min_len=2.5, max_len=12.0)
        self.assertGreaterEqual(len(sl), 2)
        self.assertTrue(all(s.duration <= 12.5 for s in sl))
        self.assertTrue(all(a.end <= b.start + 0.01 for a, b in zip(sl, sl[1:])))
        self.assertEqual(slicer.find_slices(np.zeros(SR, "float32"), SR), [])
        self.assertEqual(slicer.find_slices(speechish(0.05), SR), [])

    def test_unbroken_speech_is_force_cut(self):
        from faily.modules import slicer
        dips = [(i * 0.8 + 0.70, i * 0.8 + 0.80) for i in range(37)]       # word gaps (0.1 s) shorter than a pause
        sl = slicer.find_slices(speechish(30.0, pauses=dips), SR, max_len=12.0)
        self.assertGreaterEqual(len(sl), 3)
        self.assertTrue(all(s.duration <= 12.5 for s in sl))

    def test_dense_speech_with_almost_no_silence_still_slices(self):
        from faily.modules import slicer
        sl = slicer.find_slices(speechish(30.0), SR, max_len=12.0)         # flat and continuous: no quiet frames at all
        self.assertGreaterEqual(len(sl), 3)
        self.assertTrue(all(s.duration <= 12.5 for s in sl))

    def test_split_file_writes_16bit_mono(self):
        from faily.modules import slicer
        src = write_wav(SANDBOX / "long.wav", speechish(20.0, pauses=[(5.0, 5.7), (9.5, 10.3), (14.0, 14.8), (18.0, 19.0)]))
        out = slicer.split_file(src, SANDBOX / "slices", "medium")
        self.assertGreaterEqual(len(out), 2)
        for wav, sl in out:
            info = sf.info(str(wav))
            self.assertEqual((info.channels, info.subtype), (1, "PCM_16"))
            self.assertAlmostEqual(info.duration, sl.duration, delta=0.1)

    def test_audio_info_and_ensure_stereo(self):
        from faily.modules import edit
        p = write_wav(SANDBOX / "m.wav", speechish(1.0))
        self.assertEqual(edit.audio_info(p)["channels"], 1)
        edit.ensure_stereo(p)
        self.assertEqual(edit.audio_info(p)["channels"], 2)
        edit.ensure_stereo(p)                                        # no-op when already stereo
        self.assertEqual(edit.audio_info(p)["channels"], 2)

    def test_find_silence_splits(self):
        from faily.modules import edit
        x = speechish(12.0, pauses=[(4.0, 4.5), (8.0, 8.5)])
        cuts = edit.find_silence_splits(x, SR, 2)
        self.assertEqual(len(cuts), 2)
        self.assertTrue(4.0 * SR <= cuts[0] <= 4.5 * SR and 8.0 * SR <= cuts[1] <= 8.5 * SR)
        self.assertEqual(edit.find_silence_splits(speechish(12.0), SR, 2), [])      # no gaps: no cuts
        self.assertEqual(edit.find_silence_splits(x, SR, 0), [])

    def test_apply_edits(self):
        from faily.modules import edit
        src = write_wav(SANDBOX / "src.wav", speechish(3.0))
        out = SANDBOX / "o.wav"
        edit.apply_edits(src, out, volume_db=-6.0)
        a, b = sf.read(str(src))[0], sf.read(str(out))[0]
        self.assertAlmostEqual(float(np.abs(b).max() / np.abs(a).max()), 0.5, delta=0.05)
        edit.apply_edits(src, out, trim_start=0.5, trim_end=0.5)
        self.assertAlmostEqual(sf.info(str(out)).duration, 2.0, delta=0.02)
        edit.apply_edits(src, out, speed=2.0)
        self.assertAlmostEqual(sf.info(str(out)).duration, 1.5, delta=0.1)
        edit.apply_edits(src, out, stereo=True)
        self.assertEqual(sf.info(str(out)).channels, 2)
        edit.apply_edits(out, SANDBOX / "mono.wav", mono=True)
        self.assertEqual(sf.info(str(SANDBOX / "mono.wav")).channels, 1)
        edit.apply_edits(src, out, highpass=True, highpass_hz=200.0)
        self.assertTrue(np.isfinite(sf.read(str(out))[0]).all())

    def test_apply_edits_pitch_silence_denoise_run(self):
        from faily.modules import edit
        padded = np.concatenate([np.zeros(SR, "float32"), speechish(1.0), np.zeros(SR, "float32")])
        src = write_wav(SANDBOX / "p.wav", padded)
        out = SANDBOX / "o2.wav"
        edit.apply_edits(src, out, trim_silence=True)
        self.assertLess(sf.info(str(out)).duration, 2.0)
        edit.apply_edits(src, out, pitch_semitones=2.0)
        self.assertGreater(sf.info(str(out)).duration, 1.5)
        edit.apply_edits(src, out, denoise=True, denoise_strength=0.5)
        self.assertTrue(np.isfinite(sf.read(str(out))[0]).all())

    def test_mix_tracks(self):
        from faily.modules import edit
        a = write_wav(SANDBOX / "a.wav", speechish(2.0))
        b = write_wav(SANDBOX / "b.wav", speechish(1.0, sr=8000, seed=2), sr=8000)
        out = edit.mix_tracks([
            {"path": a, "vol": 1.0, "muted": False, "offset": 0.0},
            {"path": b, "vol": 0.5, "muted": False, "offset": 1.5},
            {"path": a, "vol": 1.0, "muted": True, "offset": 0.0},
        ], SANDBOX / "mix.wav")
        info = sf.info(str(out))
        self.assertAlmostEqual(info.duration, 2.5, delta=0.1)
        self.assertEqual(info.samplerate, SR)
        data = sf.read(str(out))[0]
        self.assertLessEqual(float(np.abs(data).max()), 1.0)


# ═════════════════════════════════════════════════════════════════════════════
class TestImporter(Sandboxed):
    def setUp(self):
        super().setUp()
        from faily.modules import importer, slicer
        self.I, self.slicer = importer, slicer

    def _staged(self, n=3):
        src = write_wav(SANDBOX / "recording.wav", speechish(30.0, pauses=[(i * 5 + 4.2, i * 5 + 4.9) for i in range(6)]))
        pieces = self.slicer.split_file(src, SANDBOX / "sl", "short")
        return self.I.stage_slices(pieces[:n], src), src

    def test_stage_is_hidden_and_has_clips(self):
        stage, _ = self._staged()
        self.assertTrue(stage.startswith(self.I.STAGE_PREFIX))
        self.assertEqual(C.list_characters(), [])
        self.assertEqual(len(C.get_character(stage)["ref_clips"]), 3)

    def test_commit_new_character(self):
        stage, _ = self._staged()
        cfg = C.get_character(stage)
        for rc in cfg["ref_clips"]:
            rc["transcript"] = "hello there"
        C._cfg(stage).write_text(json.dumps(cfg))
        res = self.I.commit_stage(stage, "fresh", is_new=True)
        self.assertEqual(res, {"added": 3, "no_transcript": 0})
        self.assertIsNone(C.get_character(stage))
        fresh = C.get_character("fresh")
        self.assertEqual(len(fresh["ref_clips"]), 2)               # first clip became the primary
        self.assertIn("ref_audio", fresh)

    def test_commit_existing_and_conflicts(self):
        make_char("alice")
        stage, _ = self._staged()
        res = self.I.commit_stage(stage, "alice", is_new=False)
        self.assertEqual(res["added"], 3)
        self.assertEqual(res["no_transcript"], 3)
        self.assertEqual(len(C.get_character("alice")["ref_clips"]), 3)
        stage2, _ = self._staged()
        with self.assertRaises(FileExistsError):
            self.I.commit_stage(stage2, "alice", is_new=True)
        with self.assertRaises(FileNotFoundError):
            self.I.commit_stage(stage2, "ghost", is_new=False)

    def test_inflated_clips_stay_inflated_after_commit(self):
        from faily.modules.inflation import InflatedCandidate
        make_char("alice")
        wav = write_wav(SANDBOX / "inf.wav", speechish(2.0))
        stage = self.I.stage_inflated([InflatedCandidate(category="neutral", text="Say  this.", path=wav, issues=[])])
        self.I.commit_stage(stage, "alice", is_new=False)
        rc = C.get_character("alice")["ref_clips"][0]
        self.assertEqual((rc["source"], rc["category"], rc["transcript"]), ("inflated", "neutral", "Say this."))

    def test_move_clips_and_discard(self):
        stage, _ = self._staged()
        keys = [rc["file"] for rc in C.get_character(stage)["ref_clips"]][:1]
        other = self.I.move_clips(stage, keys)
        self.assertEqual(len(C.get_character(other)["ref_clips"]), 1)
        self.assertEqual(len(C.get_character(stage)["ref_clips"]), 2)
        self.I.discard_stage(other)
        self.assertIsNone(C.get_character(other))
        self.I.discard_stage("alice")                              # only _stage_ folders can be discarded
        make_char("alice")
        self.I.discard_stage("alice")
        self.assertIsNotNone(C.get_character("alice"))

    def test_purge_old_stages(self):
        stage, _ = self._staged()
        old = time.time() - 3 * 86400
        os.utime(C.CHARACTERS_DIR / stage / "config.json", (old, old))
        os.utime(C.CHARACTERS_DIR / stage, (old, old))
        self.I.purge_old_stages(max_age_hours=24)
        self.assertFalse((C.CHARACTERS_DIR / stage).exists())


# ═════════════════════════════════════════════════════════════════════════════
class TestInflation(Sandboxed):
    def setUp(self):
        super().setUp()
        from faily.modules import inflation
        self.inf = inflation
        self.bank = inflation.load_script_bank()

    def test_bank_matches_categories(self):
        self.assertEqual(set(self.bank), set(self.inf.CATEGORIES))
        self.assertTrue(all(len(v) > 0 for v in self.bank.values()))
        self.assertTrue(all(isinstance(t, str) and t.strip() for v in self.bank.values() for t in v))

    def test_bank_parsing_and_extra_files(self):
        d = SANDBOX / "bank"; d.mkdir()
        (d / "scripts.txt").write_text("# comment\nstray line\n## one\nA\n\nB\n# note\n## two\nC\n")
        (d / "scripts_more.txt").write_text("## one\nD\n")
        bank = self.inf.load_script_bank(d / "scripts.txt")
        self.assertEqual(bank, {"one": ["A", "B", "D"], "two": ["C"]})
        with self.assertRaises(FileNotFoundError):
            self.inf.load_script_bank(d / "missing.txt")

    def test_start_class(self):
        sc = self.inf.start_class
        cases = {"The cat": "the", "This one": "th", "Think hard": "th", "Apple pie": "vowel", "Hour of need": "vowel",
                 "Mary had": "sonorant", "Whole lot": "sonorant", "Knife edge": "sonorant", "One day": "sonorant",
                 "Use it": "sonorant", "Sam said": "fricative", "Photo": "fricative", "Cell phone": "fricative",
                 "Dog days": "stop", "Quick": "stop", "42 reasons": "sonorant", "": "sonorant"}
        for text, want in cases.items():
            self.assertEqual(sc(text), want, text)

    def test_allocate_counts(self):
        a = self.inf._allocate_counts
        self.assertEqual(sum(a({"x": 1, "y": 1, "z": 1}, 100).values()), 100)
        self.assertEqual(a({"x": 3, "y": 1}, 8), {"x": 6, "y": 2})
        self.assertNotIn("z", a({"x": 1, "z": 0}, 5))
        self.assertEqual(a({"x": 0}, 5), {})
        self.assertEqual(a({"x": 1}, 0), {})

    def test_balanced_order_is_a_permutation(self):
        import random
        texts = [f"{w} line {i}" for i, w in enumerate(["The", "Apple", "Mary", "Sam", "Dog", "This"] * 10)]
        out = self.inf._balanced_order(texts, random.Random(1))
        self.assertEqual(sorted(out), sorted(texts))
        head = [self.inf.start_class(t) for t in out[:12]]
        self.assertGreaterEqual(len(set(head)), 4)                  # openings are mixed, not clustered

    def test_sample_lines_exact_distinct_and_excluded(self):
        import random
        weights = {k: 1.0 for k in self.inf.CATEGORIES}
        lines = self.inf.sample_lines(self.bank, weights, 60, random.Random(3))
        self.assertEqual(len(lines), 60)
        keys = [self.inf._norm(l.text) for l in lines]
        self.assertEqual(len(set(keys)), 60)
        again = self.inf.sample_lines(self.bank, weights, 60, random.Random(4), exclude=set(keys))
        self.assertFalse(set(keys) & {self.inf._norm(l.text) for l in again})

    def test_sample_lines_zero_weight_excluded_and_shortfall_redistributed(self):
        import random
        tiny = {"a": ["a1", "a2"], "b": [f"b{i}" for i in range(20)], "c": ["c1", "c2", "c3"]}
        out = self.inf.sample_lines(tiny, {"a": 1, "b": 1, "c": 0}, 10, random.Random(0))
        self.assertEqual(len(out), 10)
        self.assertNotIn("c", {l.category for l in out})
        self.assertEqual(sum(1 for l in out if l.category == "a"), 2)     # a ran out; b made up the rest
        all_out = self.inf.sample_lines(tiny, {"a": 1, "b": 1, "c": 1}, 999, random.Random(0))
        self.assertEqual(len(all_out), 25)                                # can't exceed what exists

    def test_sample_lines_drops_cross_category_duplicates(self):
        import random
        dup = {"a": ["Same line", "Other"], "b": ["same   LINE", "Third"]}
        out = self.inf.sample_lines(dup, {"a": 1, "b": 1}, 10, random.Random(0))
        self.assertEqual(len(out), 3)

    def test_used_inflated_lines_walks_ancestors(self):
        make_char("alice", inflated=2)
        make_char("kid", parent="alice")
        used = self.inf.used_inflated_lines("kid")
        self.assertEqual(used, {self.inf._norm("alice inflated line 0."), self.inf._norm("alice inflated line 1.")})
        self.assertEqual(self.inf.used_inflated_lines("ghost"), set())

    def test_generate_batch_with_fake_generator(self):
        make_char("alice", refs=1)
        gen_calls = []

        def fake_generate(text, ref, progress, out_dir, backend, p1, p2, ref_text, char_name):
            gen_calls.append((text, backend, char_name))
            if text == "boom":
                raise RuntimeError("synthetic failure")
            return write_wav(Path(out_dir) / f"{len(gen_calls)}.wav", speechish(2.0))

        lines = [self.inf.ScriptLine("neutral", t) for t in ("one", "boom", "three")]
        errors, progress = [], []
        with mock.patch.object(self.inf, "generate", fake_generate):
            cands = asyncio.run(self.inf.generate_batch(
                "alice", lines, "xtts_v2", 0.7, 1.0, SANDBOX / "stg", [False],
                lambda i, n, l: progress.append((i, n)), lambda i, l, e: errors.append(str(e))))
        self.assertEqual([c.text for c in cands], ["one", "three"])
        self.assertEqual(errors, ["synthetic failure"])
        self.assertEqual(progress[-1], (3, 3))
        cancel = [True]
        with mock.patch.object(self.inf, "generate", fake_generate):
            none = asyncio.run(self.inf.generate_batch("alice", lines, "xtts_v2", 0.7, 1.0, SANDBOX / "stg2", cancel,
                                                       lambda *a: None, lambda *a: None))
        self.assertEqual(none, [])


class TestSpokenForm(unittest.TestCase):
    def setUp(self):
        from faily.modules import autoqualify
        self.aq = autoqualify

    def spoken(self, s, **kw):
        return " ".join(w for w, _ in self.aq.spoken_tokens(s, **kw))

    def test_currency_percent_ordinal(self):
        self.assertEqual(self.spoken("$63.47"), "sixty three dollars and forty seven cents")
        self.assertEqual(self.spoken("$1"), "one dollar")
        self.assertEqual(self.spoken("50%"), "fifty percent")
        self.assertEqual(self.spoken("the 3rd"), "the third")
        self.assertEqual(self.spoken("$2 million"), "two million dollars")

    def test_years_numbers_fractions(self):
        self.assertEqual(self.spoken("1999"), "nineteen ninety nine")
        self.assertEqual(self.spoken("1999", year_style=False), "one thousand nine hundred ninety nine")
        self.assertEqual(self.spoken("42"), "forty two")
        self.assertEqual(self.spoken("3.5"), "three point five")
        self.assertEqual(self.spoken("-4"), "minus four")
        self.assertEqual(self.spoken("1/2"), "one half")

    def test_time_and_abbreviations(self):
        self.assertEqual(self.spoken("3:30"), "three thirty")
        self.assertEqual(self.spoken("3:05"), "three oh five")
        self.assertEqual(self.spoken("Dr. Smith"), "doctor Smith")
        self.assertEqual(self.spoken("rock & roll"), "rock and roll")

    def test_punctuation_is_carried_per_token(self):
        toks = self.aq.spoken_tokens('He said, "go!"')
        self.assertEqual([w for w, _ in toks], ["He", "said", "go"])
        self.assertEqual([p for _, p in toks], ["", ",", "!"])

    def test_phone_digits_and_hyphens(self):
        self.assertEqual(self.spoken("555-0142"), "five five five oh one four two")
        self.assertEqual(self.spoken("well-known"), "well known")

    def test_every_bank_line_tokenizes(self):
        from faily.modules.inflation import load_script_bank
        os.chdir(ROOT)
        try:
            bank = load_script_bank()
        finally:
            os.chdir(SANDBOX)
        for cat, lines in bank.items():
            for line in lines:
                toks = self.aq.spoken_tokens(line)
                self.assertTrue(toks, line)
                self.assertFalse(any(any(ch.isdigit() for ch in w) for w, _ in toks), f"digits survive in {line!r}")


class TestNoiseAudit(unittest.TestCase):
    def test_returns_info_and_finds_breath_before_speech(self):
        from faily.modules import inflation_audit as A
        rng = np.random.default_rng(0)
        hiss = 0.04 * rng.standard_normal(int(0.4 * SR)).astype("float32")
        x = np.concatenate([np.zeros(int(0.3 * SR), "float32") + 0.0003 * rng.standard_normal(int(0.3 * SR)).astype("float32"),
                            hiss, 0.0003 * rng.standard_normal(int(0.4 * SR)).astype("float32"), speechish(2.0)])
        events, info = A.find_noise_events(x.astype("float32"), SR)
        self.assertTrue({"floor_db", "speech_db", "snr_db", "voiced_start", "voiced_end"} <= set(info))
        self.assertTrue(any(e.where == "start" for e in events), events)

    def test_clean_speech_has_no_events(self):
        from faily.modules import inflation_audit as A
        events, info = A.find_noise_events(speechish(3.0, pauses=[(0.0, 0.4), (1.4, 1.8), (2.6, 3.0)]), SR)
        self.assertEqual([e for e in events if e.where != "inside"], [])
        self.assertGreater(info["snr_db"], 15)

    def test_silence_is_reported_as_no_speech(self):
        from faily.modules import inflation_audit as A
        events, info = A.find_noise_events(np.zeros(SR, "float32") + 1e-5, SR)
        self.assertTrue(info.get("no_speech"))

    def test_word_diff_helpers(self):
        from faily.modules import inflation_audit as A
        self.assertEqual(A._wer(["a", "b"], ["a", "b"]), 0.0)
        self.assertGreater(A._wer(["a", "b"], ["a", "c"]), 0.0)


# ═════════════════════════════════════════════════════════════════════════════
class TestVCHelpers(Sandboxed):
    def setUp(self):
        super().setUp()
        from faily.modules import vc
        self.vc = vc

    def test_backends_are_well_formed(self):
        self.assertEqual(list(self.vc.BACKENDS), ["xtts_v2", "f5_tts", "chatterbox"])
        for key, b in self.vc.BACKENDS.items():
            self.assertTrue({"label", "desc", "text_hint", "split_chars", "param1", "param2"} <= set(b), key)
            for p in (b["param1"], b["param2"]):
                self.assertTrue(p["min"] <= p["default"] <= p["max"], key)
                self.assertGreater(p["step"], 0)
        self.assertEqual(set(self.vc.STAGE2_BACKENDS), {"freevc", "openvoice", "seedvc"})

    def test_clip_names(self):
        d = SANDBOX / "names"; d.mkdir()
        n1 = self.vc._make_clip_name("E-Girl", "Dark Mood", "Hello world", d)
        self.assertEqual(n1, "e_girl_dark_hello_001.wav")
        (d / n1).write_bytes(b"x")
        self.assertEqual(self.vc._make_clip_name("E-Girl", "Dark Mood", "Hello world", d), "e_girl_dark_hello_002.wav")
        self.assertEqual(self.vc._make_clip_name(None, "", "", d), "clip_001.wav")
        self.assertTrue(self.vc._make_clip_name("x" * 40, "", "", d).startswith("x" * 16 + "_"))

    def test_split_text(self):
        s = self.vc._split_text_for_generation
        self.assertEqual(s("Short line.", 100), ["Short line."])
        text = "First sentence here. Second sentence there. Third one now."
        chunks = s(text, 45)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 45 for c in chunks))
        self.assertEqual(" ".join(chunks), text)
        long_word_sentence = " ".join(["word"] * 40)
        self.assertTrue(all(len(c) <= 30 for c in s(long_word_sentence, 30)))
        self.assertEqual(s("  spaced \n  out  ", 100), ["spaced out"])

    def test_ideal_max_chars(self):
        ref = write_wav(SANDBOX / "r.wav", speechish(8.0))
        self.assertEqual(self.vc._ideal_max_chars("xtts_v2", ref), self.vc.BACKENDS["xtts_v2"]["split_chars"])
        self.assertEqual(self.vc._ideal_max_chars("f5_tts", ref), int(8.0 * 15 * 2.5))
        self.assertEqual(self.vc._ideal_max_chars("f5_tts", None), self.vc.BACKENDS["f5_tts"]["split_chars"])

    def test_normalize_audio(self):
        p = write_wav(SANDBOX / "n.wav", speechish(2.0, level=0.05))
        self.vc._normalize_audio(p, -18.0)
        x = sf.read(str(p))[0]
        self.assertAlmostEqual(20 * np.log10(np.sqrt(np.mean(x ** 2))), -18.0, delta=1.0)

    def test_dispatch_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            self.vc._dispatch_generate("nope", "t", None, SANDBOX / "o.wav", 1, 1, "")

    def test_generate_routes_output_by_character(self):
        make_char("alice", refs=1)
        ref = C.get_ref_path("alice")
        fake = lambda backend, text, ref_path, out, *a: write_wav(out, speechish(1.5))
        with mock.patch.object(self.vc, "_dispatch_generate", fake):
            for_char = self.vc.generate("Hello there.", ref, None, None, "xtts_v2", char_name="alice")
            unsaved = self.vc.generate("Hello there.", SANDBOX / "x.wav", None, None, "xtts_v2", char_name="nobody")
            via_ref = self.vc.generate("Hello there.", ref, None, None, "xtts_v2")
        self.assertEqual(for_char.parent, P.CHARACTERS_DIR / "alice" / "history")
        self.assertEqual(unsaved.parent, P.MISC_DIR)
        self.assertEqual(via_ref.parent, P.CHARACTERS_DIR / "alice" / "history")
        self.assertTrue(for_char.name.startswith("alice_xtts_hello_"), for_char.name)

    def test_generate_chunks_long_text(self):
        calls = []

        def fake(backend, text, ref_path, out, *a):
            calls.append(text)
            return write_wav(out, speechish(1.2, seed=len(calls)))

        long_text = ". ".join(f"This is sentence number {i} of the long passage" for i in range(30)) + "."
        with mock.patch.object(self.vc, "_dispatch_generate", fake):
            out = self.vc.generate(long_text, SANDBOX / "r.wav", None, SANDBOX / "chunked", "xtts_v2")
        self.assertGreater(len(calls), 3)
        self.assertTrue(all(len(c) <= self.vc.BACKENDS["xtts_v2"]["split_chars"] for c in calls))
        self.assertGreater(sf.info(str(out)).duration, len(calls) * 0.8)
        self.assertEqual(list((SANDBOX / "chunked").glob("_*")), [])              # chunk scratch files cleaned up

    def test_f5_short_reference_trims_long_audio(self):
        long_ref = write_wav(SANDBOX / "long_ref.wav", speechish(18.0, pauses=[(10.0, 10.4)]))
        with mock.patch("faily.modules.asr.transcribe", return_value="trimmed words"):
            path, text, tmp = self.vc._f5_short_reference(long_ref, "full transcript of everything")
        self.assertEqual(text, "trimmed words")
        self.assertLessEqual(sf.info(str(path)).duration, 12.0)
        self.assertEqual(path, tmp)
        short = write_wav(SANDBOX / "short_ref.wav", speechish(6.0))
        self.assertEqual(self.vc._f5_short_reference(short, "keep")[1:], ("keep", None))

    def test_freevc_speaker_encoder_uses_local_copy_without_network(self):
        dest = self.vc.VC_MODELS_DIR / "coqui" / "freevc_speaker_encoder.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"0" * 1_100_000)
        with mock.patch("urllib.request.urlretrieve", side_effect=AssertionError("must not download")):
            self.assertEqual(self.vc._freevc_speaker_encoder(), str(dest.resolve()))

    def test_freevc_speaker_encoder_retries_flaky_network(self):
        dest = self.vc.VC_MODELS_DIR / "coqui" / "freevc_speaker_encoder.pt"
        dest.unlink(missing_ok=True)
        attempts = []

        def flaky(url, to):
            attempts.append(1)
            if len(attempts) < 3:
                raise OSError("dns")
            Path(to).write_bytes(b"0" * 1_100_000)

        with mock.patch("urllib.request.urlretrieve", flaky), mock.patch.object(self.vc.time, "sleep"):
            self.vc._freevc_speaker_encoder()
        self.assertEqual(len(attempts), 3)
        self.assertTrue(dest.exists())


# ═════════════════════════════════════════════════════════════════════════════
class TestModelManager(unittest.TestCase):
    def setUp(self):
        from faily.core.model_manager import ModelManager
        self.mm = ModelManager()
        self.mm.device = "cpu"

    def test_load_caches_and_unload(self):
        calls = []
        loader = lambda: calls.append(1) or object()
        a = self.mm.load("a", loader)
        self.assertIs(self.mm.load("a", loader), a)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.mm.loaded, ["a"])
        self.mm.unload("a"); self.mm.unload("a")
        self.assertEqual(self.mm.loaded, [])

    def test_unload_all(self):
        for k in "abc":
            self.mm.load(k, object)
        self.mm.unload_all()
        self.assertEqual(self.mm.loaded, [])

    def test_lru_eviction_under_vram_pressure(self):
        self.mm.device = "cuda"
        self.mm.vram_reserve_gb = 6.0
        free = [10.0]
        self.mm._free_vram_gb = lambda: free[0]
        self.mm.load("a", object); time.sleep(0.01)
        self.mm.load("b", object); time.sleep(0.01)
        self.mm.load("a", object); time.sleep(0.01)              # touch a: b is now least recently used
        free[0] = 2.0
        orig_unload = self.mm.unload
        self.mm.unload = lambda k: (orig_unload(k), free.__setitem__(0, free[0] + 5))[0]
        self.mm.load("c", object)
        self.assertNotIn("b", self.mm.loaded)
        self.assertIn("a", self.mm.loaded)
        self.assertIn("c", self.mm.loaded)

    def test_oom_retries_once_after_unloading_everything(self):
        import torch
        self.mm.load("old", object)
        attempts = []

        def loader():
            attempts.append(1)
            if len(attempts) == 1:
                raise torch.cuda.OutOfMemoryError("oom")
            return "ok"

        self.assertEqual(self.mm.load("new", loader), "ok")
        self.assertEqual(self.mm.loaded, ["new"])
        self.assertEqual(len(attempts), 2)

    def test_free_vram_is_none_off_cuda(self):
        self.assertIsNone(self.mm._free_vram_gb())


class TestRvcDsp(unittest.TestCase):
    def setUp(self):
        from faily.modules import rvc
        self.rvc = rvc

    def test_f0_tracks_pitch(self):
        t = np.arange(int(1.0 * self.rvc._SR)) / self.rvc._SR
        f0 = self.rvc._voiced_f0((0.3 * np.sin(2 * np.pi * 180 * t)).astype("float32"))
        self.assertGreater(len(f0), 10)
        self.assertAlmostEqual(float(np.median(f0)), 180.0, delta=8.0)
        self.assertEqual(len(self.rvc._voiced_f0(np.zeros(self.rvc._SR, "float32"))), 0)

    def test_envelope_helpers(self):
        a = speechish(1.5, sr=self.rvc._SR)
        env = self.rvc._mean_log_envelope(a)
        self.assertEqual(env.ndim, 1)
        out = self.rvc._envelope_transfer(a, env + 1.0, 0.5, 3)
        self.assertAlmostEqual(len(out) / len(a), 1.0, delta=0.02)        # STFT framing may pad slightly
        self.assertTrue(np.isfinite(out).all())

    def test_pitch_shift_preserves_length_roughly(self):
        a = speechish(1.0, sr=self.rvc._SR)
        self.assertAlmostEqual(len(self.rvc._shift_pitch(a, 3.0)) / len(a), 1.0, delta=0.1)


# ═════════════════════════════════════════════════════════════════════════════
class TestAppPieces(Sandboxed):
    def test_every_module_imports(self):
        import importlib
        import pkgutil
        failures = []
        for pkg_name in ("faily.core", "faily.modules", "faily.ui", "faily.ui.tabs"):
            pkg = importlib.import_module(pkg_name)
            for mod in pkgutil.iter_modules(pkg.__path__):
                try:
                    importlib.import_module(f"{pkg_name}.{mod.name}")
                except Exception as exc:                       # noqa: BLE001
                    failures.append(f"{pkg_name}.{mod.name}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [])

    def test_basic_auth_middleware(self):
        from faily.ui.app import _BasicAuth
        import base64
        sent = []

        async def inner(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def send(msg):
            sent.append(msg)

        mw = _BasicAuth(inner, "faily", "secret")
        good = base64.b64encode(b"faily:secret")

        def call(headers, scope_type="http"):
            sent.clear()
            asyncio.run(mw({"type": scope_type, "headers": headers}, None, send))
            return sent[0].get("status")

        self.assertEqual(call([]), 401)
        self.assertEqual(call([(b"authorization", b"Basic " + base64.b64encode(b"faily:wrong"))]), 401)
        self.assertEqual(call([(b"authorization", b"Basic " + good)]), 200)
        self.assertEqual(call([], scope_type="websocket"), 200)                # websockets pass through
        self.assertTrue(any(h[0] == b"www-authenticate" for h in (call([]) or [], ) and sent[0]["headers"]))

    def test_component_helpers(self):
        from faily.ui import components as comp
        p = P.CHARACTERS_DIR / "alice" / "history" / "a.wav"
        write_wav(p, speechish(0.2))
        self.assertTrue(comp.output_url(p).startswith("/outputs/characters/alice/history/a.wav?v="))
        self.assertEqual(comp.output_url(OUT / "nothere.wav"), "/outputs/nothere.wav")
        make_char("alice", refs=2, inflated=1)
        self.assertEqual(comp.ref_counts_text("alice"), "3 recordings · 1 inflated")
        S.set_zero_shot_reference("live")
        self.assertTrue(comp.ref_live_only())
        S.set_zero_shot_reference("all")
        self.assertFalse(comp.ref_live_only())

    def test_tts_and_foley_write_to_their_folders(self):
        from faily.modules import foley, tts
        self.assertEqual(foley.SFX_OUTPUT_DIR, P.FOLEY_DIR)
        self.assertEqual(tts.TTS_OUTPUT_DIR, P.MISC_DIR)


class TestRepoHygiene(unittest.TestCase):
    def test_all_python_compiles(self):
        import py_compile
        bad = []
        for f in list((ROOT / "faily").rglob("*.py")) + list((ROOT / "scripts").glob("*.py")) + [ROOT / "main.py"]:
            try:
                py_compile.compile(str(f), doraise=True, cfile=str(SANDBOX / "x.pyc"))
            except py_compile.PyCompileError as exc:
                bad.append(str(exc))
        self.assertEqual(bad, [])

    def test_no_stale_references(self):
        """Things removed from the project must not linger in code, config or docs."""
        banned = ["speecht5", "SpeechT5", "speechbrain", "outputs/vc", "outputs/tts", "outputs/sfx", "VC_OUTPUT_DIR"]
        skip = {"test_functions.py"}
        hits = []
        for f in list((ROOT / "faily").rglob("*.py")) + list((ROOT / "scripts").glob("*")) + [ROOT / "pyproject.toml", ROOT / "README.md"]:
            if f.suffix not in (".py", ".sh", ".toml", ".md") or f.name in skip:
                continue
            text = f.read_text(errors="ignore")
            hits += [f"{f.relative_to(ROOT)}: {b}" for b in banned if b in text]
        self.assertEqual(hits, [])

    def test_script_bank_has_no_duplicate_lines(self):
        seen, dups = {}, []
        for f in sorted((ROOT / "faily" / "data").glob("inflation_scripts*.txt")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    key = " ".join(line.split()).casefold()
                    if key in seen:
                        dups.append(f"{f.name}: {line!r} (also in {seen[key]})")
                    seen[key] = f.name
        self.assertEqual(dups, [])


if __name__ == "__main__":
    try:
        unittest.main(verbosity=1, argv=[sys.argv[0]] + sys.argv[1:])
    finally:
        os.chdir(ROOT)
        shutil.rmtree(SANDBOX, ignore_errors=True)
