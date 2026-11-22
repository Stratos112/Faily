# Handoff note — written 2026-11-21 by Claude, for the next Claude session

The user rebuilt the Docker container after this note was written and will ask you to read it.
Delete or update this file once its open items are done (CLAUDE.md points here).

## 1. Where we left off

Working on Piper fine-tuning of character **eg2** (Amy base voice) — the user's complaint:
**"the timbre is there, but the words sound like she's doing a terrible impression of somebody else."**
The inflated clips themselves sound great to the user (that is not the issue, they said so explicitly).
I can't hear audio — everything below is from the measurements.

## 2. Code changes made before the rebuild

Committed by the user as **963cb14** ("better graphs. history output. pronunciation every checkpoint…"):
- `faily/ui/training_charts.py`: loss parts split into one graph each (own linear scale, faint per-epoch +
  bold 5-epoch avg), listed under LOSS in the left graph list; `describe()` notes frozen dp on the duration graph.
- `faily/ui/tabs/piper_training_dialog.py`: PROGRESS / CHECKPOINTS / LOG tabs; log tailed from a file.
- `faily/modules/piper.py`: run log file `outputs/characters/<name>/piper_training.log` (`start_log`/`append_log`,
  restarted per run); `_CandidateExporter.score()` scores pronunciation (Whisper, GPU) LIVE as each checkpoint is
  exported, base voice scored at run start; `_measure_and_score` only fills gaps; `training_history()` includes `run`;
  `train()` calls `_write_report` when a run stops (finished / stopped / failed, not on cancellation).
- `faily/modules/training_report.py` (new): writes `outputs/characters/<name>/training_report/` — summary.txt,
  epochs.csv, history.json, status.json, log.txt, overview.png + one PNG per graph. Replaced each run.

**Uncommitted at the time of writing** (tell the user if they still show in `git status`):
- `faily/ui/tabs/piper_training_dialog.py`: fix for "The current slot cannot be determined" crash — each checkpoint
  row's tooltip is created with the row (`score_lbl.tip`) and only `set_text()` later; `show_error` in the
  background preview task wrapped in `with candidates_col:`.
- `faily/modules/asr.py`: logging filter drops transformers' per-call "Both max_new_tokens and max_length" warning.
- `Dockerfile`: the piper layer now copies `scripts/patch_piper_train_*.py` by glob (the hand list was missing
  unique_validation / keep_checkpoints / loss_parts — a clean build would have failed). This busts the piper-venv
  layer once, so the rebuild after this note is a long one.
- `scripts/test_functions.py`: regression tests for the above. Suite was green (276 tests) before the rebuild.

Rolled back (do NOT reintroduce without asking): a piper_train "epoch probe" patch that rendered test sentences
every epoch. The user said the per-epoch graphs weren't needed.

Why the rebuild: the dev server (`python3 main.py`, started 2026-11-20 06:40) predated all of the above, so the
last runs wrote no log and no training_report.

## 3. First things to check after the rebuild
1. ~~Test suite~~ DONE 2026-11-21: 277 OK in the devcontainer (environment re-checked and written into CLAUDE.md "Dev environment"; torchcodec removed from the dev .venv, coqui import shim added, XTTS zero-shot passes). Then (same day, user-approved): Faily's own Dockerfile/compose/entrypoint/.dockerignore deleted (Faily runs directly in the shared devcontainer), port 7842 forwarded in `~/projects/.devcontainer/devcontainer.json`, piper_venv rebuilt on `.python/` 3.11 (old one in `piper_venv.old-sys311/`), README + CLAUDE.md rewritten. Still uncommitted at the user's request.
2. The training dialog renders its 3 tabs in a real browser (never visually verified — no browser here).
3. After the next run: `outputs/characters/eg2/piper_training.log` and `outputs/characters/eg2/training_report/`
   exist. Read `summary.txt` + `overview.png` there when the user asks what a run did.

## 4. Findings so far (eg2)

Runs (models m002–m005 were later deleted by the user; m001 and m006 remain):
- m002/m003/m004: EVERYTHING, timbre=reference, rhythm+pronunciation=base (dp,enc_p frozen), LR 1e-4,
  60 + 20 + 20 epochs. Timbre learned by ~epoch 2010; afterwards nothing measurable changed; val loss flat
  (~41-42) from ~2050 while train loss kept falling.
- m006 (latest): all three = reference, LR 1e-5, 20 epochs resumed from m005, 1113 clips (103 real + 1010 inflated,
  49.6 min). Over those 20 epochs: gen loss 31.6 → 33, val flat ~36, every measure flat. 1e-5 for 20 epochs does nothing.
  m006 measures: pronunciation 96.5–99.5 %, voice match 93.5 % (ceiling 99.8), words/s **4.1**, pitch 245 Hz,
  pitch range **2.1–2.5 st**, brightness 1260–1350 Hz.

**The key finding — measured with voice_metrics on the active clips:**

| group | clips | words/s | pitch | pitch range | brightness |
|---|---|---|---|---|---|
| her REAL recordings | 103 (~10 min) | 2.1 | 272 Hz | 10.6 st | 1358 Hz |
| inflated — ALL f5_tts | 1010 (~39 min) | 3.7 | 244 Hz | 3.4 st | 1537 Hz |
| m006 | — | 4.1 | 245 Hz | 2.1–2.5 st | ~1300 Hz |

So the model learned **F5-TTS's reading delivery** (fast, lower, flat) in her timbre — F5 is 91 % of the data. That
matches "impression of somebody else". Whisper pronunciation can't see it (it only scores intelligibility). Caveats:
her clips are conversational (pauses lower words/s), pyin may overstate range on expressive audio — but the gap is large.
Also: the graphs' "your clips" reference line is a sample of ALL training clips, i.e. ~F5, so it hid this.

## 5. Recommended configurations (given to the user, not yet run)

**A — two-stage (first pick): timbre from everything, delivery from her recordings only**
1. Fresh from Amy (NOT from m006 — that lineage has 180 epochs of F5 delivery): EVERYTHING,
   timbre=reference, rhythm=base, pronunciation=base, LR 1e-4, ~40 epochs.
2. Resume that model: RECORDINGS ONLY, all three = reference, LR 5e-5, ~60 epochs.
   Watch speaking rate fall toward ~2–2.5 and pitch range rise toward ~10 st. If pronunciation drops below
   ~90 % and keeps falling, take the checkpoint before the drop, or redo stage 2 with pronunciation=base.

**B — cleanest test of the explanation:** fresh from Amy, RECORDINGS ONLY, timbre+rhythm=reference,
pronunciation=base, LR 1e-4, ~100 epochs. Timbre may be rougher (only ~10 min).

**C — for future inflation:** measure a small batch per backend against her recordings before generating
a big one (Chatterbox with exaggeration, XTTS, conversation/everyday categories); keep the closest.

**Quick check, no training:** synthesize with m006 at length_scale ≈ 1.4 (her pace). `piper.infer()` already takes
`length_scale`; whether the UI exposes it was NOT checked yet.

## 6. Open offers the user hasn't answered yet
- Make the graphs' "your clips" reference line come from REAL recordings only when a character has any
  (`piper._write_reference_metrics` / `training_history`), so delivery gaps like the F5 one show up immediately.
- Expose a length-scale (speed) control for Piper playback if the UI lacks one.

How I measured section 4's table (re-run to compare new batches):
```python
from collections import defaultdict; from pathlib import Path
from faily.core.characters import get_ref_chain; from faily.modules import voice_metrics as VM
chain = [c for c in get_ref_chain("eg2", include_excluded=True) if c.get("active", True)]
g = defaultdict(list)
for c in chain: g["REAL" if c.get("source") != "inflated" else f"inflated:{c.get('backend')}"].append((Path(c["audio"]), c.get("transcript", "")))
for k, clips in g.items(): print(k, len(clips), {x: VM.measure_clips(clips, n=40).get(x) for x in ("wps", "pitch", "pitch_range", "centroid")})
```

## 7. 2026-11-22 — Claude's own eg2 training experiments (blind test pending)
User's verdict before: m011 best, but "cotton balls in her mouth" pronunciation; rhythm fine-not-great; told NOT to clean/low-cut the samples.
Clarity metric (scratch, `$CLAUDE_JOB_DIR/tmp/clarity.py`): wav2vec2-base-960h greedy CER on 12 held-out consonant-heavy sentences (no LM, so it hears mush Whisper forgives).
Baselines: Lessac 3.6 %, F5 clips 5.4, m007 5.4, Amy 6.2, m009 11.4, m011 15.1, her REAL recordings 33.4 %.
Runs (all eg2, via piper.train exactly like the UI): bl1 m011 -> F5-only, timbre+pron ref, rhythm base, 5e-5 x30: CER 4.8 @2179, match 93.2 -> saved **m012 bl1-crisp**.
bl2 bl1 -> real-only, timbre+rhythm ref, pron base, 2e-5: CER up to 16-22, noise floor back to her room -> saved 2199 as **m013 bl2-real** (contrast).
bl3 bl1 -> real-only, ONLY rhythm (dp) unlocked, 1e-4: weights verified frozen except dp, still CER 20-26, match 85 -> not saved.
bl4 fresh Lessac, F5-only, 1e-4 x30: 4.5-5.3 words/s, CER 14-25 -> not saved. bl5 bl1 -> everything, timbre only, 1e-5 x20: CER ~7.5, match 93.4 -> saved **m014 bl5-blend**.
Conclusion so far: anything learned from her 103 real clips (any module) loses clarity; learning from the F5 clips gains it.
Blind test: `outputs/misc/bline_compare/` (letters K P R T X, 5 sentences each, raw piper.infer defaults) covers m007, m011, m012, m013, m014. KEY: `.claude/bline_key.json` — don't reveal it until the user has ranked them.
