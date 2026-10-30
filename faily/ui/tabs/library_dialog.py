"""LIBRARY dialog: bring characters in from another folder (another machine, an old
backup) or copy them out to one. Copies; nothing in the source is changed."""
from nicegui import run as ni_run, ui

from faily.core.characters import (
    CHARACTERS_DIR, scan_library, import_characters, export_characters, list_characters,
)
from faily.ui.components import section_label, show_error

_DIM = "text-[#444] font-mono text-[10px] tracking-wide"


def open_library_dialog(on_change):
    """on_change(): called after characters were imported so the lists refresh."""
    picks: dict[str, bool] = {}

    with ui.dialog() as dlg, ui.card().classes("bg-[#111] border border-[#2a2a2a] min-w-[520px] gap-3"):
        ui.label("CHARACTER LIBRARY").classes("text-amber-400 font-mono text-xs tracking-widest")
        ui.label(f"Active library: {CHARACTERS_DIR.resolve()}").classes(_DIM + " break-all")

        with ui.tabs().props("dense").classes("w-full") as tabs:
            t_in = ui.tab("LOAD", icon="download")
            t_out = ui.tab("SAVE", icon="upload")

        with ui.tab_panels(tabs, value=t_in).classes("w-full bg-transparent"):

            # ── LOAD: another folder -> here ─────────────────────────────────
            with ui.tab_panel(t_in).classes("gap-2 p-0 flex flex-col"):
                section_label("SOURCE FOLDER")
                ui.label(
                    "A characters folder, or an outputs folder that contains one. "
                    "In Docker the folder must be mounted into the container."
                ).classes(_DIM)
                src = ui.input(placeholder="/path/to/outputs/characters").props("outlined dark dense").classes("w-full")
                list_col = ui.column().classes("w-full gap-1 max-h-72 overflow-y-auto")
                opts_row = ui.column().classes("w-full gap-1")

                def _scan():
                    list_col.clear()
                    picks.clear()
                    try:
                        found = scan_library(src.value.strip())
                    except Exception as exc:
                        with list_col:
                            ui.label(str(exc)).classes("text-red-400 font-mono text-[10px]")
                        return
                    with list_col:
                        if not found:
                            ui.label("No characters in that folder.").classes(_DIM)
                        for c in found:
                            picks[c["name"]] = not c["here"]
                            tag = []
                            if c["parent"]:
                                tag.append(f"↳ {c['parent']}")
                            tag.append(f"{c['n_refs']} refs")
                            if c["has_model"]:
                                tag.append("trained")
                            if c["here"]:
                                tag.append("already here")
                            with ui.row().classes("w-full items-center gap-2"):
                                ui.checkbox(c["name"], value=picks[c["name"]],
                                            on_change=lambda e, n=c["name"]: picks.__setitem__(n, e.value)
                                            ).props("dense color=amber").classes("font-mono text-[11px]")
                                ui.label("  ·  ".join(tag)).classes(_DIM)

                async def _import():
                    names = [n for n, on in picks.items() if on]
                    if not names:
                        ui.notify("Nothing selected", type="warning"); return
                    try:
                        res = await ni_run.io_bound(
                            import_characters, src.value.strip(), names,
                            overwrite=ow_in.value, include_training=train_in.value,
                        )
                    except Exception as exc:
                        show_error(exc); return
                    on_change()
                    msg = f"imported {len(res['imported'])}"
                    if res["skipped"]:
                        msg += f", skipped {len(res['skipped'])} already here"
                    ui.notify(msg, type="positive", timeout=3000)
                    dlg.close()

                with opts_row:
                    train_in = ui.checkbox("Include training data (piper_train / piper_dataset — large)", value=True
                                           ).props("dense color=amber").classes("font-mono text-[10px]")
                    ow_in = ui.checkbox("Replace characters that already exist here", value=False
                                        ).props("dense color=red").classes("font-mono text-[10px]")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("SCAN", icon="search", on_click=_scan).props("flat dense color=grey").classes("font-mono text-[10px]")
                    ui.button("IMPORT SELECTED", icon="download", on_click=_import).props(
                        "color=amber unelevated dense").classes("font-mono text-[10px]")

            # ── SAVE: here -> another folder ─────────────────────────────────
            with ui.tab_panel(t_out).classes("gap-2 p-0 flex flex-col"):
                section_label("DESTINATION FOLDER")
                dest = ui.input(placeholder="/path/to/backup/characters").props("outlined dark dense").classes("w-full")
                out_names = [c["name"] for c in list_characters()]
                sel = ui.select(out_names, multiple=True, value=[], label="Characters").props(
                    "outlined dark dense use-chips").classes("w-full")
                ui.label("Parents of variants are included automatically.").classes(_DIM)
                train_out = ui.checkbox("Include training data (large)", value=True
                                        ).props("dense color=amber").classes("font-mono text-[10px]")
                ow_out = ui.checkbox("Replace characters already in the destination", value=False
                                     ).props("dense color=red").classes("font-mono text-[10px]")

                async def _export():
                    if not dest.value.strip() or not sel.value:
                        ui.notify("Pick characters and a destination", type="warning"); return
                    try:
                        res = await ni_run.io_bound(
                            export_characters, dest.value.strip(), list(sel.value),
                            overwrite=ow_out.value, include_training=train_out.value,
                        )
                    except Exception as exc:
                        show_error(exc); return
                    msg = f"saved {len(res['exported'])}"
                    if res["skipped"]:
                        msg += f", skipped {len(res['skipped'])} already there"
                    ui.notify(msg, type="positive", timeout=3000)

                with ui.row().classes("w-full justify-end"):
                    ui.button("SAVE TO FOLDER", icon="upload", on_click=_export).props(
                        "color=amber unelevated dense").classes("font-mono text-[10px]")

        with ui.row().classes("w-full justify-end"):
            ui.button("Close", on_click=dlg.close).props("flat dense color=grey")
    dlg.open()
