"""The REFERENCES tab: a hub for a character's reference clips.

  ADD / QUALIFY / INFLATE      get more or better data in
  metrics + switchable chart   how much, how long, from which source / category / backend
  group cards                  one-click activate, deactivate, delete or generate-more per group
  clip table                   every clip, sortable and filterable, with a selection toolbar so
                               nothing has to be found by scrolling a list

Training only uses ACTIVE clips, and deleting is recoverable (refs_trash + UNDO), so trying a
different mix never means regenerating what was thrown away.
"""
import os
import tempfile
from pathlib import Path
from typing import Callable

from nicegui import run as ni_run, ui

from faily.core import refs
from faily.core.characters import (
    add_ref_clip, get_character, split_long_ref_clip, update_character_metadata, update_ref_clip,
)
from faily.core.paths import TMP_DIR
from faily.ui.components import output_url, section_label, show_error
from faily.ui.tabs.import_dialog import open_import_dialog
from faily.ui.tabs.inflation_dialog import open_inflation_dialog
from faily.ui.tabs.qualify_dialog import open_qualify_dialog

_BTN = "font-mono tracking-widest"
_DIM = "text-[#444] font-mono text-[10px] tracking-wide"
_AMBER, _GREY = "#f59e0b", "#4b5563"
_PALETTE = ["#f59e0b", "#38bdf8", "#a78bfa", "#34d399", "#fb7185", "#fbbf24", "#22d3ee", "#c084fc", "#94a3b8", "#f97316"]


def fmt_minutes(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def grid_rows(rows: list[dict], long_over: float) -> list[dict]:
    """Rows as the table shows them."""
    out = []
    for r in rows:
        flags = []
        if r["seconds"] > long_over:
            flags.append("LONG")
        if not r["transcript"].strip():
            flags.append("NO TEXT")
        out.append({
            "key": r["key"], "name": r["name"] + (" ★" if r["primary"] else ""),
            "origin": r["origin"], "category": r["category"] or "", "backend": r["backend"] or ("unknown" if r["origin"] == "inflated" else ""),
            "batch": r["batch"] or "", "seconds": round(r["seconds"], 1), "train": r["active"], "clone": r["clone"],
            "flags": " ".join(flags), "transcript": r["transcript"], "url": output_url(r["audio"]),
        })
    return out


def chart_options(data: dict, horizontal: bool, measure: str, stack: str | None) -> dict:
    """ECharts bar options for refs.breakdown() data, in the app's dark style."""
    cats = data["categories"]
    value_axis = {"type": "value", "name": refs.MEASURES[measure], "nameTextStyle": {"color": "#666"},
                  "axisLabel": {"color": "#777"}, "splitLine": {"lineStyle": {"color": "#222"}}}
    cat_axis = {"type": "category", "data": cats, "axisLabel": {"color": "#aaa", "interval": 0, "hideOverlap": True},
                "axisLine": {"lineStyle": {"color": "#333"}}}
    if horizontal:
        cat_axis["inverse"] = True
    stacked = stack is not None and len(data["series"]) > 0
    series = []
    for i, s in enumerate(data["series"]):
        item = {"type": "bar", "name": s["name"], "data": s["data"], "barMaxWidth": 34, "emphasis": {"focus": "series"}}
        if stacked:
            item["stack"] = "total"
        if stack == "state":
            item["itemStyle"] = {"color": _AMBER if s["name"] == "active" else _GREY}
        series.append(item)
    return {
        "backgroundColor": "transparent", "color": _PALETTE,
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"show": stacked, "top": 0, "textStyle": {"color": "#888"}},
        "grid": {"left": 8, "right": 18, "top": 34 if stacked else 20, "bottom": 8, "containLabel": True},
        "xAxis": value_axis if horizontal else cat_axis,
        "yAxis": cat_axis if horizontal else value_axis,
        "series": series,
    }


def build_refs_hub(name: str, *, dialog_host, rebuild: Callable[[], None], hooks: dict) -> Callable[[], None]:
    """Build the hub into the current slot. `rebuild` re-renders the whole character panel; hooks:
    edit(row), split(row), to_edit(row), open_character(name). Returns refresh()."""
    state = {"by": "origin", "measure": "minutes", "stack": "state", "horizontal": False}
    selected: list[dict] = []

    # ── data ─────────────────────────────────────────────────────────────
    def load():
        rows = refs.clip_rows(name)
        return rows, refs.summarize(rows), refs.long_threshold(rows)

    rows, stats, long_over = load()
    inherited = refs.inherited_rows(name)
    parent_name = (get_character(name) or {}).get("parent")

    # ── action bar ───────────────────────────────────────────────────────
    upload_mode = ["add"]

    async def on_upload(e):
        if upload_mode[0] == "split":
            import uuid
            upload_mode[0] = "add"
            keep = TMP_DIR / "import"
            keep.mkdir(parents=True, exist_ok=True)
            saved = keep / f"{uuid.uuid4().hex[:8]}_{Path(e.file.name).name}"
            saved.write_bytes(await e.file.read())
            with dialog_host:
                open_import_dialog(saved, Path(e.file.name).stem, dialog_host, on_added=rebuild,
                                   fixed_target=name, delete_source=True)
            return
        fd, tmp_name = tempfile.mkstemp(suffix=Path(e.file.name).suffix or ".wav")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(await e.file.read())
            add_ref_clip(name, tmp, upload_text.value.strip())
            ui.notify("Added to the reference pool", type="positive", timeout=2000)
            rebuild()
        except Exception as exc:                                    # noqa: BLE001
            show_error(exc)
        finally:
            tmp.unlink(missing_ok=True)

    with ui.row().classes("w-full items-center gap-2"):
        ui.button("QUALIFY DATA", icon="cleaning_services", on_click=lambda: open_qualify_dialog(name, rebuild)).props(
            "flat dense color=amber").classes(_BTN).tooltip(
            "Whisper-assisted review: fix volume/clipping, trim silence and noise, correct transcripts — "
            "or delete bad clips. Originals are kept.")
        ui.button("INFLATE DATA", icon="dynamic_form", on_click=lambda: open_inflation_dialog(name, rebuild)).props(
            "flat dense color=amber").classes(_BTN).tooltip(
            "Generate zero-shot clips of this voice from a diverse script bank (Chatterbox or F5-TTS) "
            "and add them here. They arrive switched ON and can be switched off or deleted any time.")
        ui.space()
        ui.label("Training uses ACTIVE clips only").classes(_DIM)

    with ui.expansion("ADD CLIPS", icon="upload_file").classes("w-full").props(
            "dense header-class='text-[#888] font-mono text-[11px] tracking-widest'"):
        ref_upload = (ui.upload(on_upload=on_upload, multiple=False, auto_upload=False)
                      .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Select clip'").classes("w-full"))
        upload_text = (ui.input(placeholder="transcript (optional)…").props("outlined dark dense")
                       .classes("w-full mt-2 font-mono text-[11px]"))
        with ui.row().classes("gap-2 mt-1"):
            ui.button("Upload", icon="cloud_upload",
                      on_click=lambda: (upload_mode.__setitem__(0, "add"), ref_upload.run_method("upload"))
                      ).props("flat dense color=amber").classes("font-mono text-[10px] tracking-widest")
            ui.button("Split & qualify…", icon="content_cut",
                      on_click=lambda: (upload_mode.__setitem__(0, "split"), ref_upload.run_method("upload"))
                      ).props("flat dense color=amber").classes("font-mono text-[10px] tracking-widest").tooltip(
                "Slice the selected recording at its pauses, check each slice (volume, silence, "
                "Whisper-suggested transcript) one by one, then add them here. Ignores the transcript box.")

    # ── metric cards ─────────────────────────────────────────────────────
    cards_row = ui.row().classes("w-full gap-2 flex-wrap mt-1")

    def render_cards():
        cards_row.clear()
        a, al = stats["active"], stats["all"]
        items = [
            ("CLIPS", f"{a['clips']}", f"of {al['clips']} active"),
            ("LENGTH", fmt_minutes(a["seconds"]), f"of {fmt_minutes(al['seconds'])} active"),
            ("AVERAGE", f"{al['avg']:.1f} s", "per clip"),
            ("RECORDINGS", f"{stats['recordings']['clips']}", f"{fmt_minutes(stats['recordings']['seconds'])} · avg {stats['recordings']['avg']:.1f} s"),
            ("INFLATED", f"{stats['inflated']['clips']}", f"{fmt_minutes(stats['inflated']['seconds'])} · avg {stats['inflated']['avg']:.1f} s"),
            ("TRANSCRIPTS", f"{stats['transcript_share']:.0%}", f"{stats['transcribed']} of {al['clips']}"),
        ]
        with cards_row:
            for title, big, small in items:
                with ui.column().classes("gap-0 px-3 py-2 rounded bg-[#111] border border-[#1e1e1e] min-w-[110px]"):
                    ui.label(title).classes("text-[#444] font-mono text-[9px] tracking-widest")
                    ui.label(big).classes("text-amber-400 font-mono text-lg leading-tight")
                    ui.label(small).classes("text-[#555] font-mono text-[9px]")
            if stats["long"]:
                with ui.column().classes("gap-0 px-3 py-2 rounded bg-[#1a1200] border border-amber-900/40 min-w-[110px] cursor-pointer").on(
                        "click", lambda: filter_to(flags="LONG", contains=True)):
                    ui.label("LONG CLIPS").classes("text-amber-700 font-mono text-[9px] tracking-widest")
                    ui.label(str(stats["long"])).classes("text-amber-400 font-mono text-lg leading-tight")
                    ui.label(f"over {stats['long_threshold']:.0f} s · click to view").classes("text-[#7a5a1a] font-mono text-[9px]")

    # ── chart ────────────────────────────────────────────────────────────
    ui.separator().classes("my-1 opacity-10")
    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
        section_label("BREAKDOWN")
        by_tg = ui.toggle(refs.DIMENSIONS, value=state["by"]).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
        measure_tg = ui.toggle(refs.MEASURES, value=state["measure"]).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
        stack_sel = ui.select({"": "no split", **{k: f"split by {v.lower()}" for k, v in refs.DIMENSIONS.items()}},
                              value=state["stack"]).props("outlined dark dense options-dense").classes("w-44")
        ui.button(icon="screen_rotation", on_click=lambda: (state.__setitem__("horizontal", not state["horizontal"]), render_chart())
                  ).props("flat dense round color=grey").tooltip("Rotate the chart")
    chart = ui.echart({}).classes("w-full").style("height: 250px")

    def render_chart():
        stack = state["stack"] or None
        if stack == state["by"]:
            stack = None
        data = refs.breakdown(rows, state["by"], state["measure"], stack)
        chart.options.clear()
        chart.options.update(chart_options(data, state["horizontal"], state["measure"], stack))
        chart.update()

    def on_chart_ctl():
        state["by"], state["measure"], state["stack"] = by_tg.value, measure_tg.value, stack_sel.value
        render_chart()

    for ctl in (by_tg, measure_tg, stack_sel):
        ctl.on_value_change(lambda _e: on_chart_ctl())

    # ── group cards ──────────────────────────────────────────────────────
    ui.separator().classes("my-1 opacity-10")
    section_label("GROUPS")
    ui.label("Switch whole groups on or off for training, delete them, or generate more like them.").classes(_DIM)
    groups_row = ui.row().classes("w-full gap-2 flex-wrap")

    def filter_to(contains: bool = False, **cols):
        model = {c: {"filterType": "text", "type": "contains" if contains else "equals", "filter": v} for c, v in cols.items()}
        grid.run_grid_method("setFilterModel", model or None)

    def group_action(g: dict, what: str):
        if what == "on":
            n = refs.set_active(name, g["keys"], True)
        elif what == "off":
            n = refs.set_active(name, g["keys"], False)
        elif what == "delete":
            confirm_delete(g["keys"], f"{g['clips']} clips in this group")
            return
        else:
            return
        ui.notify(f"{n} clip{'s' if n != 1 else ''} switched {what}", timeout=1800)
        refresh()

    def render_groups():
        groups_row.clear()
        with groups_row:
            for g in refs.groups(rows):
                is_inf = g["origin"] == "inflated"
                title = (g["category"] or "uncategorised") if is_inf else "recordings"
                sub = (g["backend"] or "") if is_inf else "your own audio"
                share = g["active_seconds"] / g["seconds"] if g["seconds"] else 0.0
                with ui.column().classes("gap-1 p-2 rounded bg-[#111] border border-[#1e1e1e] w-[210px]"):
                    with ui.row().classes("w-full items-center gap-1 no-wrap"):
                        ui.icon("science" if is_inf else "mic", size="13px").classes("text-[#555] shrink-0")
                        ui.label(title).classes("text-[#bbb] font-mono text-[11px] truncate flex-grow")
                    ui.label(sub).classes("text-[#555] font-mono text-[9px]")
                    ui.label(f"{g['clips']} clips · {fmt_minutes(g['seconds'])}").classes("text-[#777] font-mono text-[10px]")
                    ui.linear_progress(value=share, show_value=False, size="4px").props("color=amber track-color=grey-9").classes("w-full")
                    ui.label(f"{g['active_clips']} active ({share:.0%} of its audio)").classes("text-[#555] font-mono text-[9px]")
                    with ui.row().classes("gap-0 w-full"):
                        ui.button(icon="toggle_on", on_click=lambda g=g: group_action(g, "on")).props("flat dense color=amber").tooltip("Switch the group ON for training")
                        ui.button(icon="toggle_off", on_click=lambda g=g: group_action(g, "off")).props("flat dense color=grey").tooltip("Switch the group OFF (kept, just not trained on)")
                        ui.button(icon="filter_alt", on_click=lambda g=g: filter_to(
                            origin=g["origin"], **({"category": g["category"]} if g["category"] else {}),
                            **({"backend": g["backend"]} if is_inf and g["backend"] else {}))).props("flat dense color=grey").tooltip("Show these clips in the table")
                        if is_inf:
                            ui.button(icon="add_circle", on_click=lambda g=g: open_inflation_dialog(
                                name, rebuild, preset={"category": g["category"], "backend": g["backend"]})
                            ).props("flat dense color=grey").tooltip("Generate more like this")
                        ui.space()
                        ui.button(icon="delete_outline", on_click=lambda g=g: group_action(g, "delete")).props("flat dense color=negative").tooltip("Delete the group (can be undone)")
            if inherited:
                with ui.column().classes("gap-1 p-2 rounded bg-[#0d0d0d] border border-dashed border-[#222] w-[210px]"):
                    with ui.row().classes("items-center gap-1"):
                        ui.icon("link", size="13px").classes("text-[#444]")
                        ui.label("from parent").classes("text-[#777] font-mono text-[11px]")
                    ui.label(f"{len(inherited)} clips · {fmt_minutes(sum(r['seconds'] for r in inherited))}").classes("text-[#555] font-mono text-[10px]")
                    if parent_name:
                        ui.button(f"manage on {parent_name}", on_click=lambda: hooks["open_character"](parent_name)
                                  ).props("flat dense color=grey").classes("font-mono text-[9px]")

    # ── clip table + selection toolbar ───────────────────────────────────
    ui.separator().classes("my-1 opacity-10")
    section_label("CLIPS")
    toolbar = ui.row().classes("w-full items-center gap-1 flex-wrap")
    sel_label = None
    with toolbar:
        sel_label = ui.label("nothing selected").classes("text-amber-400 font-mono text-[11px] min-w-[110px]")
        b_on = ui.button("ACTIVATE", icon="toggle_on", on_click=lambda: apply_selected("on")).props("flat dense color=amber").classes(_BTN)
        b_off = ui.button("DEACTIVATE", icon="toggle_off", on_click=lambda: apply_selected("off")).props("flat dense color=grey").classes(_BTN)
        b_del = ui.button("DELETE", icon="delete_outline", on_click=lambda: apply_selected("delete")).props("flat dense color=negative").classes(_BTN)
        b_undo = ui.button("UNDO DELETE", icon="undo", on_click=lambda: undo_delete()).props("flat dense color=grey").classes(_BTN)
        ui.space()
        ui.button("SELECT SHOWN", on_click=lambda: grid.run_grid_method("selectAll", "filtered")).props("flat dense color=grey").classes("font-mono text-[10px]")
        ui.button("CLEAR", on_click=lambda: grid.run_grid_method("deselectAll")).props("flat dense color=grey").classes("font-mono text-[10px]")
        ui.button("RESET FILTERS", on_click=lambda: grid.run_grid_method("setFilterModel", None)).props("flat dense color=grey").classes("font-mono text-[10px]")
        with ui.button(icon="more_horiz").props("flat dense color=grey").tooltip("More actions"):
            with ui.menu():
                ui.menu_item("Allow selected for cloning", on_click=lambda: apply_selected("clone_on"))
                ui.menu_item("Block selected from cloning", on_click=lambda: apply_selected("clone_off"))
                ui.separator()
                ui.menu_item("Edit name / transcript… (one clip)", on_click=lambda: apply_selected("edit"))
                ui.menu_item("Split & qualify… (one clip)", on_click=lambda: apply_selected("split"))
                ui.menu_item("Send to the Edit tab (one clip)", on_click=lambda: apply_selected("to_edit"))
                ui.separator()
                ui.menu_item("Split all long clips", on_click=lambda: split_all_long())
    quick = ui.input(placeholder="search name or transcript…").props("outlined dark dense clearable").classes("w-full font-mono text-[11px]")
    quick.on_value_change(lambda e: grid.run_grid_method("setGridOption", "quickFilterText", e.value or ""))

    cell_dark = {"fontFamily": "monospace", "fontSize": "11px"}
    grid = ui.aggrid({
        "columnDefs": [
            {"field": "name", "headerName": "CLIP  (click to play)", "flex": 2, "minWidth": 160,
             "cellStyle": {**cell_dark, "cursor": "pointer"}, "tooltipField": "name"},
            {"field": "origin", "headerName": "SOURCE", "width": 100, "cellStyle": cell_dark},
            {"field": "category", "headerName": "CATEGORY", "width": 115, "cellStyle": cell_dark},
            {"field": "backend", "headerName": "BACKEND", "width": 105, "cellStyle": cell_dark},
            {"field": "batch", "headerName": "BATCH", "width": 120, "cellStyle": cell_dark},
            {"field": "seconds", "headerName": "SEC", "width": 78, "filter": "agNumberColumnFilter", "cellStyle": cell_dark},
            {"field": "train", "headerName": "TRAIN", "width": 82, ":valueFormatter": "p => p.value ? '● on' : '○ off'",
             ":cellStyle": "p => ({fontFamily:'monospace', fontSize:'11px', color: p.value ? '#f59e0b' : '#666'})"},
            {"field": "clone", "headerName": "CLONE", "width": 82, ":valueFormatter": "p => p.value ? '● ok' : '○ no'",
             ":cellStyle": "p => ({fontFamily:'monospace', fontSize:'11px', color: p.value ? '#9ca3af' : '#555'})"},
            {"field": "flags", "headerName": "FLAGS", "width": 105, "cellStyle": {**cell_dark, "color": "#d97706"}},
            {"field": "transcript", "headerName": "TRANSCRIPT (double-click to edit)", "flex": 4, "minWidth": 220, "editable": True,
             "tooltipField": "transcript", "cellStyle": {**cell_dark, "color": "#777"}},
        ],
        "defaultColDef": {"sortable": True, "filter": "agTextColumnFilter", "floatingFilter": True, "resizable": True},
        "rowData": [],
        "rowSelection": {"mode": "multiRow", "checkboxes": True, "headerCheckbox": True,
                         "enableClickSelection": False, "selectAll": "filtered"},
        ":getRowId": "p => p.data.key",
        "rowHeight": 26, "headerHeight": 28, "floatingFiltersHeight": 28, "tooltipShowDelay": 400,
        ":onCellClicked": "e => { if (e.colDef.field === 'name' && e.data) window.failyPlay(e.data.url); }",
    }, theme="balham").classes("w-full").style("height: 440px")

    async def read_selection():
        selected[:] = await grid.get_selected_rows()
        n = len(selected)
        sel_label.set_text(f"{n} selected" if n else "nothing selected")

    grid.on("selectionChanged", lambda _e: read_selection())

    async def on_cell_edit(e):
        args = e.args
        if (args.get("colDef") or {}).get("field") != "transcript":
            return
        key, text = args["data"]["key"], (args.get("newValue") or "").strip()
        try:
            if key == refs.PRIMARY:
                update_character_metadata(name, {"transcript": text})
            else:
                update_ref_clip(name, key, None, text)
            ui.notify("transcript saved", type="positive", timeout=1200)
        except Exception as exc:                                    # noqa: BLE001
            show_error(exc)

    grid.on("cellValueChanged", on_cell_edit)

    # ── actions on the selection ─────────────────────────────────────────
    def keys_of(selection: list[dict]) -> list[str]:
        return [r["key"] for r in selection]

    def confirm_delete(keys: list[str], what: str):
        real = [k for k in keys if k != refs.PRIMARY]
        if not real:
            ui.notify("The primary reference can't be deleted — deactivate or replace it instead.", type="warning")
            return
        with ui.dialog() as d, ui.card().classes("bg-[#1a1a1a] border border-[#3a1a1a] min-w-[380px] gap-3"):
            ui.label("DELETE CLIPS").classes("text-red-400 font-mono text-xs tracking-widest")
            ui.label(f"Delete {len(real)} clip{'s' if len(real) != 1 else ''} ({what})?").classes("text-white font-mono text-sm")
            ui.label("They move to refs_trash/ and can be restored with UNDO DELETE. Trained models are unaffected.").classes(_DIM)

            def go():
                d.close()
                refs.delete_clips(name, real)
                ui.notify(f"Deleted {len(real)} — UNDO DELETE brings them back", timeout=5000)
                refresh()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=d.close).props("flat dense color=grey")
                ui.button(f"DELETE {len(real)}", icon="delete", on_click=go).props("color=negative unelevated dense")
        d.open()

    def undo_delete():
        last = refs.last_deleted(name)
        if not last:
            ui.notify("Nothing to undo", timeout=1500)
            return
        n = refs.restore_deleted(name, last["id"])
        ui.notify(f"Restored {n} clip{'s' if n != 1 else ''}", type="positive", timeout=2500)
        refresh()

    async def apply_selected(action: str):
        await read_selection()
        if not selected:
            ui.notify("Select clips in the table first (checkbox, or SELECT SHOWN)", type="warning", timeout=2500)
            return
        keys = keys_of(selected)
        one = selected[0] if len(selected) == 1 else None
        if action in ("on", "off"):
            n = refs.set_active(name, keys, action == "on")
            ui.notify(f"{n} clip{'s' if n != 1 else ''} switched {action}", timeout=1800)
        elif action in ("clone_on", "clone_off"):
            n = refs.set_cloning(name, keys, action == "clone_on")
            ui.notify(f"{n} clip{'s' if n != 1 else ''} updated", timeout=1800)
        elif action == "delete":
            confirm_delete(keys, "selected in the table")
            return
        elif action in ("edit", "split", "to_edit"):
            if not one:
                ui.notify("Pick exactly one clip for this action", type="warning")
                return
            row = next((r for r in rows if r["key"] == one["key"]), None)
            if row:
                hooks[action](row)
            return
        refresh()

    async def split_all_long():
        longs = [r for r in rows if r["seconds"] > long_over and not r["primary"]]
        if not longs:
            ui.notify("No long clips", timeout=1500)
            return
        target = max(stats["all"]["avg"], 6.0)
        ui.notify(f"Splitting {len(longs)} long clip{'s' if len(longs) != 1 else ''}…", timeout=2500)
        made = failed = 0
        for r in longs:
            try:
                n = await ni_run.io_bound(split_long_ref_clip, name, r["key"], target)
            except Exception:                                       # noqa: BLE001
                n = 0
            made += n
            failed += (n == 0)
        ui.notify(f"Split into {made} clips" + (f"; {failed} couldn't be split (no clean pause)" if failed else ""),
                  type="positive" if made else "warning", timeout=5000)
        refresh()

    # ── refresh ──────────────────────────────────────────────────────────
    def refresh():
        nonlocal rows, stats, long_over
        rows, stats, long_over = load()
        render_cards()
        render_chart()
        render_groups()
        grid.options["rowData"] = grid_rows(rows, long_over)
        grid.update()
        b_undo.set_visibility(refs.last_deleted(name) is not None)
        selected.clear()
        sel_label.set_text("nothing selected")

    refresh()
    return refresh
