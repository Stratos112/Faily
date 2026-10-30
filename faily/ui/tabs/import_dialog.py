"""CLONE tab: "split & qualify" import. Slices a recording at its pauses, then hands the
slices to the qualify review (qualify_dialog) with Whisper's transcripts
pre-filled, and finally adds the survivors to a new or existing character."""
import shutil
import uuid
from pathlib import Path
from faily.core.paths import TMP_DIR

import soundfile as sf
from nicegui import run as ni_run, ui

from faily.core.characters import get_character, list_characters
from faily.modules import autoqualify as AQ
from faily.modules.importer import stage_slices, move_clips, commit_stage, discard_stage
from faily.modules.slicer import PRESETS, split_file
from faily.ui.components import show_error
from faily.ui.tabs.qualify_dialog import open_qualify_dialog

_TMP = TMP_DIR / "slices"
_LBL = "text-[#444] font-mono text-[10px] tracking-widest"


def open_import_dialog(
    source: Path, default_name: str, host, on_added=lambda: None,
    fixed_target: str | None = None, delete_source: bool = False,
    remove_original=None,
) -> None:
    """host: a long-lived element to create the review dialog under (this dialog is
    closed before the review opens, and a dialog nested in a closed one would vanish).
    fixed_target: add the slices to this existing character and don't ask where.
    delete_source: `source` is a temporary upload — remove it once it has been sliced
    (or the dialog is cancelled).
    remove_original: a function that takes the long clip out of its character — when given,
    the dialog offers to run it once the slices have been added (used when splitting a clip
    that is already a character's reference)."""
    try:
        info = sf.info(str(source))
        length = f"{info.duration:.0f} s" if info.duration < 120 else f"{info.duration / 60:.1f} min"
    except Exception:
        length = "?"
    bases = [c["name"] for c in list_characters() if "parent" not in c]

    with ui.dialog().props("persistent").classes("w-full") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full max-w-xl gap-3 p-5"
    ):
        ui.label(f"SPLIT & QUALIFY  ·  {source.name}  ({length})").classes("text-white font-mono text-xs tracking-widest")
        ui.label(
            "Splits the recording at its pauses into clip-sized slices, then checks every slice — "
            "volume, stray silence and noise, and a transcript suggested by Whisper that you only "
            "adjust. Nothing is added to a character until you confirm at the end."
        ).classes("text-[#888] font-mono text-[11px] leading-snug")

        ui.label(f"ADDING TO  {fixed_target}" if fixed_target else "ADD THE CLIPS TO").classes(_LBL)
        target_kind = ui.radio(
            {"new": "A new character", "existing": "An existing character"},
            value="new" if default_name or not bases else "existing",
        ).props("dense color=amber")
        name_in = ui.input(label="New character name", value=default_name).props("outlined dark dense").classes(
            "w-full font-mono"
        )
        existing_sel = ui.select(options=bases, value=bases[0] if bases else None, label="Character").props(
            "outlined dark dense"
        ).classes("w-full font-mono")
        existing_sel.set_visibility(False)

        def _sync():
            name_in.set_visibility(target_kind.value == "new")
            existing_sel.set_visibility(target_kind.value == "existing")
        target_kind.on_value_change(lambda _: _sync())
        _sync()
        if fixed_target:
            for el in (target_kind, name_in, existing_sel):
                el.set_visibility(False)

        ui.label("HOW TO CHECK THE SLICES").classes(_LBL)
        check_mode = ui.radio(
            {"hand": "By hand — step through every slice",
             "auto_review": "Auto-qualify, add the clean ones, then review the flagged by hand",
             "auto_only": "Auto-qualify and add only the ones that pass"},
            value="hand",
        ).props("dense color=amber")
        ui.label(
            "Auto-qualify has Whisper write each slice's transcript, aligns it to the audio and cuts the "
            "slice to its words. A slice it isn't sure about (alignment doubt, a sound between words, "
            "nothing heard) counts as flagged. Transcripts added this way come from Whisper and have not "
            "been read by a person."
        ).classes("text-[#555] font-mono text-[10px] leading-snug")
        remove_cb = None
        if remove_original is not None:
            remove_cb = ui.checkbox(
                "Remove the original long clip once its slices are added (kept in refs_trash/)", value=True,
            ).props("dense color=amber").classes("font-mono text-[11px] text-[#888]")

        ui.label("SLICE LENGTH").classes(_LBL)
        preset = ui.toggle(
            {"short": "Short  ≈4 s", "medium": "Medium  ≈7 s", "long": "Long  ≈11 s"}, value="medium",
        ).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
        ui.label("Shorter slices are easier to transcribe exactly; longer ones carry more rhythm.").classes(
            "text-[#555] font-mono text-[10px]"
        )

        status = ui.label("").classes("text-amber-400 font-mono text-[11px]")
        with ui.row().classes("w-full justify-end gap-2"):
            cancel_btn = ui.button("Cancel", on_click=lambda: (_drop_source(), dlg.close())).props(
                "flat dense color=grey"
            )
            go_btn = ui.button("SPLIT & CHECK", icon="content_cut", on_click=lambda: _go()).props(
                "color=amber unelevated dense"
            ).classes("font-mono tracking-widest")

    def _drop_source():
        if delete_source:
            source.unlink(missing_ok=True)

    async def _go():
        is_new = target_kind.value == "new" and not fixed_target
        target = fixed_target or ((name_in.value or "").strip() if is_new else existing_sel.value)
        if not target:
            ui.notify("Choose a character name first", type="warning")
            return
        if is_new and get_character(target) is not None:
            ui.notify(f"“{target}” already exists — pick another name or add to it as an existing character",
                      type="warning", timeout=5000)
            return
        go_btn.disable(); cancel_btn.disable()
        status.set_text("Splitting at pauses…")
        tmp = _TMP / uuid.uuid4().hex[:8]
        try:
            files = await ni_run.io_bound(split_file, source, tmp, preset.value)
            if not files:
                status.set_text("")
                ui.notify("Couldn't find any speech to slice in that file", type="warning")
                go_btn.enable(); cancel_btn.enable()
                return
            stage = await ni_run.io_bound(stage_slices, files, source)
        except Exception as exc:
            status.set_text("")
            show_error(exc)
            go_btn.enable(); cancel_btn.enable()
            return
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        _drop_source()
        dur = [s.duration for _, s in files]
        mode = check_mode.value
        state = {"added": 0, "is_new": is_new}

        def _finish():
            """Everything the person asked for is done: remove the original if that was asked."""
            if state["added"] and remove_cb is not None and remove_cb.value and remove_original is not None:
                try:
                    remove_original()
                except Exception as exc:
                    show_error(exc)
            if state["added"]:
                on_added()

        def _committed(result: dict):
            state["added"] += result["added"]
            msg = f"Added {result['added']} clip{'s' if result['added'] != 1 else ''} to {target}"
            if result["no_transcript"]:
                msg += f" — {result['no_transcript']} still need a transcript"
            ui.notify(msg, type="positive", timeout=6000)
            _finish()

        if mode == "hand":
            ui.notify(f"{len(files)} slices ({min(dur):.1f}–{max(dur):.1f} s) — checking each now", timeout=3500)
            dlg.close()
            with host:
                open_qualify_dialog(stage, staging={"target": target, "is_new": is_new, "on_commit": _committed,
                                                    "on_discard": lambda: None})
            return

        # ── auto-qualify: transcribe + align + cut every slice, add the clean ones ──
        prog = {"i": 0, "n": len(files)}
        timer = ui.timer(0.3, lambda: status.set_text(
            f"Transcribing and aligning slice {prog['i'] + 1}/{prog['n']}"
            + ("  (the first one loads the models — ~30 s)" if prog["i"] == 0 else "")))
        try:
            results = await ni_run.io_bound(
                AQ.auto_qualify_stage, stage, lambda i, n, name: prog.update(i=i, n=n))
        except Exception as exc:
            timer.cancel()
            status.set_text("")
            show_error(exc)
            discard_stage(stage)
            go_btn.enable(); cancel_btn.enable()
            return
        timer.cancel()
        ok_keys = [k for k, r in results.items() if r.status == "ok"]
        flagged = [k for k, r in results.items() if r.status != "ok"]
        try:
            if ok_keys:
                ok_stage = move_clips(stage, ok_keys)
                res = commit_stage(ok_stage, target, state["is_new"])
                state["added"] += res["added"]
                state["is_new"] = False
        except Exception as exc:
            show_error(exc)
            discard_stage(stage)
            dlg.close()
            return
        dlg.close()
        summary = f"Auto-qualify: {len(ok_keys)} of {len(results)} slices passed and were added to {target}"
        if mode == "auto_review" and flagged:
            ui.notify(summary + f" — reviewing the {len(flagged)} flagged by hand", type="positive", timeout=6000)
            on_added()

            def _flagged_committed(result: dict):
                state["added"] += result["added"]
                ui.notify(f"Added {result['added']} reviewed clip{'s' if result['added'] != 1 else ''} to {target}",
                          type="positive", timeout=5000)
                _finish()

            with host:
                open_qualify_dialog(stage, staging={"target": target, "is_new": state["is_new"], "noun": "slices",
                                                    "on_commit": _flagged_committed, "on_discard": _finish})
        else:
            discard_stage(stage)
            ui.notify(summary + (f"; {len(flagged)} flagged slice{'s' if len(flagged) != 1 else ''} dropped"
                                 if flagged else ""), type="positive", timeout=6000)
            _finish()

    dlg.open()
