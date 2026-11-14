"""
Converts a base-voice checkpoint saved by the newer Piper training code
(piper1-gpl, Lightning 2.x) into one our piper_train (Lightning 1.7) can
resume from.

rhasspy retrained several voices (amy, ryan, hfc_female) with that code,
"with MRD and SDP", and replaced their checkpoints in rhasspy/piper-checkpoints.
The old ones are gone (404), and they had a grainy, "bit-crushed" vocoder
tone that every model fine-tuned from them inherited; the new ones are clean
(the published preview .onnx was always the new voice, so auditions sounded
fine while training started from the old one).

The generator and the period discriminator are the same shapes as ours. What
has to go:
  - model_mrd.*: the multi-resolution discriminator our piper_train doesn't have
  - optimizer / scheduler states: their discriminator optimizer covers the MRD
    params too, so it can't be loaded. Empty lists are what Lightning 1.7 wants
    (the keys must exist); FAILY_PIPER_LR then sets the rate on fresh optimizers
  - loops / callbacks: Lightning 2 progress state, replaced by a minimal
    Lightning-1.7 fit-loop state holding just the epoch. Without one, 1.7's
    "pre-1.6 checkpoint" fallback sets epoch_progress.completed but not
    .processed, and FitLoop.on_run_start copies processed (0) back over it:
    the run restarts at epoch 0 and trains toward the absolute max_epochs
    (thousands of epochs). global_step does come through from its own field

Usage (needs torch, so run it with the piper venv):
    piper_venv/bin/python scripts/convert_piper_checkpoint.py SRC.ckpt DEST.ckpt
"""
import sys

DROP_PREFIXES = ("model_mrd.",)
DROP_HPARAMS = ("_instantiator",)


def is_lightning2(ckpt: dict) -> bool:
    return str(ckpt.get("pytorch-lightning_version", "0")).split(".")[0] not in ("0", "1")


def convert(ckpt: dict) -> dict:
    """The resumable Lightning-1.x form of a piper1-gpl checkpoint (plain dicts in and out)."""
    epoch = {k: ckpt["epoch"] for k in ("ready", "started", "processed", "completed")}
    return {
        "epoch": ckpt["epoch"],
        "loops": {"fit_loop": {"epoch_progress": {"total": dict(epoch), "current": dict(epoch)}}},
        "global_step": ckpt["global_step"],
        "pytorch-lightning_version": "1.7.7",
        "state_dict": {k: v for k, v in ckpt["state_dict"].items() if not k.startswith(DROP_PREFIXES)},
        "optimizer_states": [],
        "lr_schedulers": [],
        "hparams_name": ckpt.get("hparams_name", "kwargs"),
        "hyper_parameters": {k: v for k, v in ckpt.get("hyper_parameters", {}).items() if k not in DROP_HPARAMS},
    }


def main(src: str, dest: str) -> None:
    import torch
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not is_lightning2(ckpt):
        raise SystemExit(f"{src} is already a Lightning {ckpt.get('pytorch-lightning_version')} checkpoint")
    out = convert(ckpt)
    dropped = len(ckpt["state_dict"]) - len(out["state_dict"])
    torch.save(out, dest)
    print(f"Converted {src} -> {dest} (epoch {out['epoch']}, dropped {dropped} MRD tensors)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
