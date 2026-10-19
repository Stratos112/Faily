"""CLONE tab: "split & clean" import. Slices a recording at its pauses, then hands the
slices to the spell-check review (cleanup_dialog) with Whisper's transcripts
pre-filled, and finally adds the survivors to a new or existing character."""
import shutil
import uuid
from pathlib import Path

import soundfile as sf
from nicegui import run as ni_run, ui

from faily.core.characters import get_character, list_characters
from faily.modules.importer import stage_slices
from faily.modules.slicer import PRESETS, split_file
from faily.ui.components import show_error
from faily.ui.tabs.cleanup_dialog import open_cleanup_dialog

_TMP = Path("outputs/vc/_slices_tmp")
_LBL = "text-[#444] font-mono text-[10px] tracking-widest"


def open_import_dialog(source: Path, default_name: str, host, on_added=lambda: None) -> None:
    """host: a long-lived element to create the review dialog under (this dialog is
    closed before the review opens, and a dialog nested in a closed one would vanish)."""
    try:
        info = sf.info(str(source))
        length = f"{info.duration:.0f} s" if info.duration < 120 else f"{info.duration / 60:.1f} min"
    except Exception:
        length = "?"
    bases = [c["name"] for c in list_characters() if "parent" not in c]

    with ui.dialog().props("persistent").classes("w-full") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full max-w-xl gap-3 p-5"
    ):
        ui.label(f"SPLIT & CLEAN  ·  {source.name}  ({length})").classes("text-white font-mono text-xs tracking-widest")
        ui.label(
            "Splits the recording at its pauses into clip-sized slices, then checks every slice — "
            "volume, stray silence and noise, and a transcript suggested by Whisper that you only "
            "adjust. Nothing is added to a character until you confirm at the end."
        ).classes("text-[#888] font-mono text-[11px] leading-snug")

        ui.label("ADD THE CLIPS TO").classes(_LBL)
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

        ui.label("SLICE LENGTH").classes(_LBL)
        preset = ui.toggle(
            {"short": "Short  ≈4 s", "medium": "Medium  ≈7 s", "long": "Long  ≈11 s"}, value="medium",
        ).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
        ui.label("Shorter slices are easier to transcribe exactly; longer ones carry more rhythm.").classes(
            "text-[#555] font-mono text-[10px]"
        )

        status = ui.label("").classes("text-amber-400 font-mono text-[11px]")
        with ui.row().classes("w-full justify-end gap-2"):
            cancel_btn = ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
            go_btn = ui.button("SPLIT & CHECK", icon="content_cut", on_click=lambda: _go()).props(
                "color=amber unelevated dense"
            ).classes("font-mono tracking-widest")

    async def _go():
        is_new = target_kind.value == "new"
        target = (name_in.value or "").strip() if is_new else existing_sel.value
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
        dur = [s.duration for _, s in files]
        ui.notify(f"{len(files)} slices ({min(dur):.1f}–{max(dur):.1f} s) — checking each now", timeout=3500)
        dlg.close()

        def _committed(result: dict):
            msg = f"Added {result['added']} clip{'s' if result['added'] != 1 else ''} to {target}"
            if result["no_transcript"]:
                msg += f" — {result['no_transcript']} still need a transcript"
            ui.notify(msg, type="positive", timeout=6000)
            on_added()

        with host:
            open_cleanup_dialog(stage, staging={"target": target, "is_new": is_new, "on_commit": _committed})

    dlg.open()
