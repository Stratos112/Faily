"""
Patches piper_train/vits/lightning.py to log each part of the generator loss.

piper_train only logs the sum (loss_gen_all = adversarial + feature matching + mel +
duration + KL). With parts of the model frozen, some of those terms can't go down at
all, so the sum flattens while the rest is still improving (or getting worse) and
the chart can't tell which. This logs every term on its own:
  loss_mel  how far the output's mel spectrum is from the clip's (x c_mel = 45)
  loss_kl   how well the text side's prior matches the audio's latent (x c_kl)
  loss_dur  duration predictor error
  loss_fm   feature matching against the discriminator's internal features
  loss_gen  adversarial: how well the output fools the discriminator
read by faily.modules.piper.read_loss_history as the "parts" of the loss.

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_loss_parts.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

lightning_py = PIPER_TRAIN_DIR / "vits" / "lightning.py"
text = lightning_py.read_text(encoding="utf-8")

MARKER = "FAILY_LOSS_PARTS"
OLD = '            self.log("loss_gen_all", loss_gen_all)\n'
NEW = OLD + (
    f"            # {MARKER}: each term on its own (see scripts/patch_piper_train_loss_parts.py)\n"
    '            self.log("loss_mel", loss_mel)\n'
    '            self.log("loss_kl", loss_kl)\n'
    '            self.log("loss_dur", loss_dur)\n'
    '            self.log("loss_fm", loss_fm)\n'
    '            self.log("loss_gen", loss_gen)\n'
)

if MARKER in text:
    print("loss-parts patch: already applied")
else:
    if text.count(OLD) != 1:
        raise RuntimeError(
            f"Could not find expected {OLD!r} in {lightning_py} — "
            "piper-train's lightning.py may have changed shape."
        )
    lightning_py.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"loss-parts patch: applied to {lightning_py}")
