"""
Resume (or continue) a Piper fine-tune with a stall watchdog, then export and
promote the result.

    .venv/bin/python scripts/piper_resume_supervised.py e-girl 250

Continues from the newest checkpoint under outputs/characters/<name>/piper_train
(run piper.train() once first so the preprocessed dataset exists) for N more
epochs. Does NOT re-run dataset prep and does not wipe anything.

Why a watchdog: a WSL2/Docker-Desktop GPU stall can leave piper_train spinning
at 100% CPU forever, stuck in a CUDA copy, with no output. Progress is judged by
file activity (tensorboard events / checkpoints under piper_train/), since the
trainer prints nothing useful. No activity for STALL_SECS -> kill, resume from
the newest checkpoint. Checkpoints are written every CKPT_EVERY epochs
(ModelCheckpoint keeps only the newest), which bounds the work lost per restart.
"""
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from faily.core.characters import CHARACTERS_DIR, set_piper_model
from faily.modules.piper import _python, base_voice_paths, finalize_piper_model, infer

STALL_SECS = 600
STARTUP_GRACE = 900
CKPT_EVERY = 25
MAX_RESTARTS = 6
PREVIEW = (
    "Hello, this is a test of the trained voice. Now a longer passage, to hear how it holds up over several "
    "sentences. The quick brown fox jumps over the lazy dog, and then it keeps on running through the forest, "
    "past the river, until the sun finally sets behind the distant hills."
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def newest_ckpt(train_dir: Path):
    ckpts = list(train_dir.rglob("*.ckpt"))
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None


def ckpt_epoch(p: Path) -> int:
    return int(re.search(r"epoch=(\d+)", p.name).group(1))


def last_activity(train_dir: Path) -> float:
    files = [p for p in train_dir.rglob("*") if p.is_file() and (p.suffix == ".ckpt" or "tfevents" in p.name)]
    return max((p.stat().st_mtime for p in files), default=0.0)


def run_once(train_dir: Path, ckpt: Path, target: int) -> str:
    """Returns 'done', 'failed' or 'stalled'."""
    cmd = [
        str(_python()), "-m", "piper_train",
        "--dataset-dir", str(train_dir),
        "--accelerator", "gpu", "--devices", "1",
        "--batch-size", "2", "--accumulate_grad_batches", "8",
        "--validation-split", "0.02", "--num-test-examples", "0",
        "--num_sanity_val_steps", "0",
        "--max_epochs", str(target),
        "--resume_from_checkpoint", str(ckpt),
        "--checkpoint-epochs", str(CKPT_EVERY),
        "--precision", "32",
        "--default_root_dir", str(train_dir),
    ]
    log(f"resuming from {ckpt.name} -> target epoch {target}")
    env = {**__import__("os").environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONFAULTHANDLER": "1"}
    out = open(train_dir / "supervised_trainer.log", "ab")
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    started = time.time()
    while True:
        time.sleep(30)
        rc = proc.poll()
        if rc is not None:
            return "done" if rc == 0 else "failed"
        idle = time.time() - max(last_activity(train_dir), started)
        limit = STALL_SECS if last_activity(train_dir) > started else STARTUP_GRACE
        if idle > limit:
            log(f"no activity for {idle:.0f}s — killing stalled trainer (pid {proc.pid})")
            proc.send_signal(signal.SIGABRT)  # faulthandler prints the stack into the trainer log
            time.sleep(10)
            proc.kill()
            proc.wait()
            return "stalled"


def main(name: str, more_epochs: int):
    char_dir = CHARACTERS_DIR / name
    train_dir = char_dir / "piper_train"
    start = newest_ckpt(train_dir)
    if start is None:
        sys.exit(f"no checkpoint under {train_dir} — run piper.train() first")
    # Lightning stores the epoch that just finished; resume begins at +1.
    target = -(-(ckpt_epoch(start) + 1 + more_epochs) // CKPT_EVERY) * CKPT_EVERY
    restarts = 0
    while True:
        ckpt = newest_ckpt(train_dir)
        if ckpt_epoch(ckpt) + 1 >= target:
            break
        result = run_once(train_dir, ckpt, target)
        log(f"trainer: {result}")
        if result == "done":
            break
        restarts += 1
        if restarts > MAX_RESTARTS:
            sys.exit(f"giving up after {MAX_RESTARTS} restarts; newest checkpoint: {newest_ckpt(train_dir)}")
    ckpt = newest_ckpt(train_dir)
    epoch = ckpt_epoch(ckpt)
    log(f"training finished at {ckpt.name}; exporting")
    cand = char_dir / "piper_candidates"
    shutil.rmtree(cand, ignore_errors=True)
    cand.mkdir()
    onnx = cand / f"epoch_{epoch:05d}.onnx"
    subprocess.run([str(_python()), "-m", "piper_train.export_onnx", str(ckpt), str(onnx)], check=True)
    shutil.copy2(base_voice_paths("lessac")[1], onnx.with_suffix(".onnx.json"))
    infer(PREVIEW, onnx, char_dir / "piper_preview_long.wav")
    final = finalize_piper_model(char_dir, onnx)
    set_piper_model(name, str(final))
    log(f"FINALIZED {final} (epoch {epoch}, {restarts} restart(s))")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
