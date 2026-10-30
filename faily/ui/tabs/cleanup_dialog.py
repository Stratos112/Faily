"""CLEAN UP dialog: tick what to remove from an outputs folder, see how much space
it frees. Works on the active outputs folder or any other one you point at."""
from nicegui import run as ni_run, ui

from faily.core import cleanup as CL
from faily.core.paths import OUTPUTS
from faily.ui.components import section_label, show_error

_DIM = "text-[#444] font-mono text-[10px] tracking-wide"


def open_cleanup_dialog(on_change):
    """on_change(): called after something was deleted so lists refresh."""
    ticked: dict[str, bool] = {}
    sizes: dict[str, int] = {}

    with ui.dialog() as dlg, ui.card().classes("bg-[#111] border border-[#2a2a2a] min-w-[560px] gap-3"):
        ui.label("CLEAN UP").classes("text-amber-400 font-mono text-xs tracking-widest")
        section_label("FOLDER")
        ui.label("The outputs folder to clean — this one by default. References, inflated clips, "
                 "favorites, character settings and each character's active model are never touched."
                 ).classes(_DIM)
        folder = ui.input(value=str(OUTPUTS.resolve())).props("outlined dark dense").classes("w-full")
        rows = ui.column().classes("w-full gap-1 max-h-96 overflow-y-auto")
        total_lbl = ui.label("").classes("text-amber-400 font-mono text-[11px]")

        def _update_total():
            n = sum(sizes[k] for k, on in ticked.items() if on)
            total_lbl.set_text(f"will free {CL.human(n)}" if n else "nothing selected")

        async def _scan():
            rows.clear(); ticked.clear(); sizes.clear()
            try:
                cats = await ni_run.io_bound(CL.scan, folder.value.strip())
            except Exception as exc:
                with rows:
                    ui.label(str(exc)).classes("text-red-400 font-mono text-[10px]")
                _update_total(); return
            with rows:
                for c in cats:
                    ticked[c["key"]] = False          # nothing pre-ticked: deleting is always deliberate
                    sizes[c["key"]] = c["bytes"]
                    empty = c["items"] == 0
                    with ui.row().classes("w-full items-start gap-2 no-wrap"):
                        cb = ui.checkbox(
                            value=False,
                            on_change=lambda e, k=c["key"]: (ticked.__setitem__(k, e.value), _update_total()),
                        ).props("dense color=amber")
                        if empty:
                            cb.disable()
                        with ui.column().classes("gap-0 flex-grow"):
                            ui.label(c["label"]).classes(
                                "font-mono text-[11px] " + ("text-[#444]" if empty else "text-[#bbb]"))
                            ui.label(c["desc"]).classes(_DIM)
                        ui.label("empty" if empty else CL.human(c["bytes"])).classes(
                            "font-mono text-[10px] text-[#666] shrink-0")
            _update_total()

        def _confirm():
            keys = [k for k, on in ticked.items() if on]
            if not keys:
                ui.notify("Nothing selected", type="warning"); return
            labels = [c[1] for c in CL.CATEGORIES if c[0] in keys]
            with ui.dialog() as cdlg, ui.card().classes("bg-[#1a1a1a] border border-[#3a1a1a] min-w-[380px] gap-3"):
                ui.label("DELETE PERMANENTLY").classes("text-red-400 font-mono text-xs tracking-widest")
                ui.label(f"From {folder.value.strip()}").classes(_DIM + " break-all")
                for l in labels:
                    ui.label(f"  •  {l}").classes("text-[#aaa] font-mono text-[10px]")
                ui.label(total_lbl.text + ". This cannot be undone.").classes("text-[#666] font-mono text-[10px]")

                async def _do():
                    cdlg.close()
                    try:
                        res = await ni_run.io_bound(CL.clean, folder.value.strip(), keys)
                    except Exception as exc:
                        show_error(exc); return
                    ui.notify(f"freed {CL.human(res['freed'])}", type="positive", timeout=3000)
                    on_change()
                    await _scan()

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=cdlg.close).props("flat dense color=grey")
                    ui.button("DELETE", on_click=_do).props("color=negative unelevated dense")
            cdlg.open()

        with ui.row().classes("w-full items-center gap-2"):
            ui.button("SCAN", icon="search", on_click=_scan).props("flat dense color=grey").classes("font-mono text-[10px]")
            ui.space()
            ui.button("Close", on_click=dlg.close).props("flat dense color=grey")
            ui.button("DELETE SELECTED", icon="delete", on_click=_confirm).props(
                "color=negative unelevated dense").classes("font-mono text-[10px]")
    dlg.open()
    ui.timer(0.05, _scan, once=True)
