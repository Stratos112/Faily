"""
Patches piper_train/vits/lightning.py's VitsModel._load_datasets so the
validation set never shares a transcript with the training set.

Stock piper_train splits with a blind random_split, so when the dataset holds
the same sentence more than once (padding lines reused across runs, repeated
takes of one line) a validation clip can have a twin in the training set and
val_loss then measures memorisation instead of generalisation. This replaces
the split with one that groups utterances by normalised text (case- and
whitespace-insensitive) and assigns each GROUP wholly to validation or wholly
to training. The validation size still targets int(n * validation_split) and is
never empty (an empty val dataloader crashes this Lightning version); groups
are drawn at random, so with no duplicates it behaves like the stock split.
If every group is larger than the target, the smallest group is used.

Idempotent — safe to run every setup.

Run after installing piper-train:
    piper_venv/bin/python scripts/patch_piper_train_unique_validation.py
"""
import importlib.util
import pathlib

_spec = importlib.util.find_spec("piper_train")
if _spec is None:
    raise RuntimeError("piper_train is not installed in the current Python environment")
PIPER_TRAIN_DIR = pathlib.Path(_spec.submodule_search_locations[0])

lightning_py = PIPER_TRAIN_DIR / "vits" / "lightning.py"
text = lightning_py.read_text(encoding="utf-8")

MARKER = "FAILY_UNIQUE_VALIDATION"

OLD = """        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )
"""
NEW = """        # FAILY_UNIQUE_VALIDATION: split by transcript group, not by clip, so no
        # validation sentence also occurs in the training set.
        import random as _random
        from torch.utils.data import Subset as _Subset

        groups = {}
        for _i, _utt in enumerate(full_dataset.utterances):
            _key = " ".join((_utt.text or f"\\0{_i}").split()).casefold()
            groups.setdefault(_key, []).append(_i)
        _order = list(groups.values())
        _random.shuffle(_order)
        val_idx = []
        for _g in _order:
            if len(val_idx) + len(_g) <= valid_set_size:
                val_idx.extend(_g)
        if not val_idx and valid_set_size > 0 and _order:
            val_idx = list(min(_order, key=len))
        _in_val = set(val_idx)
        rest = [i for i in range(len(full_dataset)) if i not in _in_val]
        _random.shuffle(rest)
        test_idx = rest[:num_test_examples]
        train_idx = rest[num_test_examples:]
        # test examples are drawn from training clips, so keep their text out of
        # the validation set too (already guaranteed: groups never straddle).
        self._train_dataset = _Subset(full_dataset, train_idx)
        self._test_dataset = _Subset(full_dataset, test_idx)
        self._val_dataset = _Subset(full_dataset, val_idx)
        _LOGGER.info(
            "Unique-text split: %s train / %s val / %s test (%s distinct texts of %s clips)",
            len(train_idx), len(val_idx), len(test_idx), len(groups), len(full_dataset),
        )
"""

if MARKER in text:
    print("unique-validation split patch: already applied")
elif OLD not in text:
    raise RuntimeError(
        f"Could not find expected random_split block in {lightning_py} — "
        "piper-train's lightning.py may have changed shape."
    )
else:
    lightning_py.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("unique-validation split patch: applied")
