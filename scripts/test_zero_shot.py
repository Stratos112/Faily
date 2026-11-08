"""Zero-shot backend test suite.

For every cloning backend (vc.BACKENDS) and every stage-2 voice converter
(vc.STAGE2_BACKENDS via tune_generate), generate one short line from a real
character's reference clip and check the result:
  - it generated at all (no exception)
  - the file is real audio: finite, not silent, plausible duration
  - Whisper hears the script (word error rate under a limit)

Each case runs in its own subprocess so a CUDA/native crash in one backend is
reported as a failure instead of taking the suite down.

  .venv/bin/python scripts/test_zero_shot.py                 # everything
  .venv/bin/python scripts/test_zero_shot.py xtts_v2 f5_tts  # named cases
  .venv/bin/python scripts/test_zero_shot.py --char e-girl --keep
Exit status is the number of failed cases.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT = "The quick brown fox jumps over the lazy dog near the old river bank."
LONG_TEXT = ("I went down to the market early this morning, hoping to find something nice for dinner. "
             "The vendors were just setting up, and the air smelled of fresh bread and ripe oranges. "
             "After a while I settled on some fish, a handful of herbs, and a loaf that was still warm. "
             "Walking home, I realized I had forgotten to buy any wine, so I turned around and went back.")
MAX_WER = 0.35
MIN_SEC, MAX_SEC = 1.5, 45.0


def _cases():
    sys.path.insert(0, str(ROOT))
    # BACKENDS lives in a heavy module; keep the names static so listing is cheap
    return ["xtts_v2", "f5_tts", "chatterbox",
            "long:xtts_v2", "long:f5_tts", "long:chatterbox",
            "parler", "tune:freevc", "tune:openvoice", "tune:seedvc",
            "vc:freevc", "vc:openvoice", "vc:seedvc"]


def worker(case: str, char: str, out_dir: Path) -> dict:
    """Runs inside the subprocess. Returns a result dict (also printed as JSON)."""
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    import faily  # noqa: F401  (sets HF_HOME etc.)
    from faily.core.characters import build_ref_audio, get_ref_chain, F5_REF_SECONDS

    res: dict = {"case": case}
    t0 = time.time()
    chain = get_ref_chain(char, live_only=True)
    if not chain:
        return {**res, "error": f"character {char!r} has no live reference audio"}
    base = case[5:] if case.startswith("long:") else case
    text = LONG_TEXT if case.startswith("long:") else TEXT
    res["text_len"] = len(text)
    is_f5 = base == "f5_tts"
    with build_ref_audio(char, require_transcript=is_f5, max_duration=F5_REF_SECONDS if is_f5 else None,
                         live_only=True) as (ref, ref_text):
        if case == "parler":
            from faily.modules.vc import _parler_generate_checked    # the path the app uses: takes are Whisper-checked
            path = out_dir / "parler.wav"
            _parler_generate_checked(text, path, "a calm young woman speaking clearly")
        elif case.startswith("vc:"):
            # converter alone, on a clean known source (a Piper base-voice sample): what Whisper
            # hears in the output must match what it hears in the source
            import soundfile as sf2
            from faily.modules import vc as V
            src = ROOT / "outputs/misc/base_voice_samples/lessac.wav"
            if not src.exists():
                return {**res, "error": f"missing clean source clip {src}"}
            from faily.modules.asr import transcribe as tr0
            d0, sr0 = sf2.read(str(src)); d0 = d0.mean(axis=1) if d0.ndim > 1 else d0
            res["source_text"] = tr0(d0.astype("float32"), sr0)
            path = out_dir / "converted.wav"
            {"freevc": V._freevc_convert, "openvoice": V._openvoice_convert,
             "seedvc": V._seedvc_convert}[case[3:]](src, ref, path)
        elif case.startswith("tune:"):
            from faily.modules.vc import tune_generate
            path = tune_generate(text, "a calm young woman speaking clearly", "parler", ref, None,
                                 out_dir, char_name=None, stage2_backend=case.split(":")[1])
        else:
            from faily.modules.vc import generate
            path = generate(text, ref, None, out_dir, backend=base, ref_text=ref_text)
        res["gen_seconds"] = round(time.time() - t0, 1)
        res["path"] = str(path)

    data, sr = sf.read(str(path), always_2d=False)
    mono = data.mean(axis=1) if data.ndim > 1 else data
    res["duration"] = round(len(mono) / sr, 2)
    res["finite"] = bool(np.isfinite(mono).all())
    res["rms_db"] = round(float(20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-9)), 1)
    res["peak"] = round(float(np.abs(mono).max()), 3)

    from faily.modules.asr import transcribe
    heard = transcribe(mono.astype("float32"), sr)
    res["heard"] = heard
    target = res.get("source_text", text)             # converter cases: compare to what the source says
    res["wer"] = round(_wer(target, heard), 3)
    return res


def _norm(s: str) -> list[str]:
    import re
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def _wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref), _norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(len(r), 1)


def check(r: dict) -> list[str]:
    """Reasons this result fails (empty = pass)."""
    if "error" in r:
        return [r["error"]]
    why = []
    if not r["finite"]:
        why.append("output has NaN/inf samples")
    if r["rms_db"] < -50:
        why.append(f"output is silent ({r['rms_db']} dBFS)")
    if not (MIN_SEC <= r["duration"] <= MAX_SEC):
        why.append(f"duration {r['duration']}s outside {MIN_SEC}-{MAX_SEC}s")
    if r["wer"] > MAX_WER:
        why.append(f"unintelligible: WER {r['wer']} > {MAX_WER} (heard {r['heard']!r})")
    return why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--char", default="e-girl")
    ap.add_argument("--keep", action="store_true", help="keep generated clips (printed paths)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--repeat", type=int, default=1,
                    help="runs per case (these models sample, so one run can be a bad draw); "
                         "a case passes when at least 2/3 of its runs do")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        r = worker(a.worker, a.char, Path(a.out))
        print("RESULT_JSON " + json.dumps(r))
        return

    cases = a.cases or _cases()
    out_root = Path(tempfile.mkdtemp(prefix="faily_zs_")) if a.keep else None
    failed = 0
    for case in cases:
        passes = 0
        print(f"\n── {case} ", "─" * (50 - len(case)), flush=True)
        for run in range(a.repeat):
            out = (out_root or Path(tempfile.mkdtemp(prefix="faily_zs_"))) / f"{case.replace(':', '_')}_{run}"
            out.mkdir(parents=True, exist_ok=True)
            try:
                p = subprocess.run([sys.executable, __file__, "--worker", case, "--char", a.char, "--out", str(out)],
                                   capture_output=True, text=True, timeout=a.timeout, cwd=ROOT)
                line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
                if line is None:
                    r = {"error": f"worker died (exit {p.returncode}): " + (p.stderr.strip().splitlines() or ["no output"])[-1]}
                    tail = "\n".join(p.stderr.strip().splitlines()[-25:])
                else:
                    r, tail = json.loads(line[12:]), ""
            except subprocess.TimeoutExpired:
                r, tail = {"error": f"timed out after {a.timeout}s"}, ""
            why = check(r)
            passes += not why
            tag = f" (run {run + 1}/{a.repeat})" if a.repeat > 1 else ""
            if a.repeat == 1 or why:
                print(("FAIL  " if why else "PASS  ") + case + tag)
                for k in ("gen_seconds", "duration", "rms_db", "wer", "heard"):
                    if k in r:
                        print(f"      {k}: {r[k]}")
                for w in why:
                    print("      ✗", w)
                if tail:
                    print("      --- stderr tail ---\n" + "\n".join("      " + l for l in tail.splitlines()))
                if a.keep and "path" in r:
                    print("      clip:", r["path"])
            else:
                print(f"PASS  {case}{tag}  wer {r.get('wer')}")
        ok = passes * 3 >= a.repeat * 2
        if a.repeat > 1:
            print(f"{'PASS' if ok else 'FAIL'}  {case}: {passes}/{a.repeat} runs")
        failed += not ok
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    sys.exit(failed)


if __name__ == "__main__":
    main()
