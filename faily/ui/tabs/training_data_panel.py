"""The activation layout on the TRAINING tab: what a run will learn from, at a glance, and a click to change it.

  gauge      active minutes against the 10 / 30 / 60 minute marks
  boxes      the groups drawn as nested boxes — recordings and inflated are the big boxes, their sources /
             backends sit inside as smaller boxes, and categories inside those as tiles. A box's size is its
             audio length; its fill is how much of it is switched on. Click any box to switch it (and
             everything inside it) on or off.

The REFERENCES hub is where clips are looked at, filtered and deleted; this is the view you train from.
"""
from typing import Callable

from nicegui import ui

from faily.core import refs
from faily.modules.profile import MIN_RECOGNIZABLE_MIN, MIN_STRONG_MIN

_DIM = "text-[#444] font-mono text-[10px] tracking-wide"
_SCALE_MIN = 60.0                        # the gauge's full width
_TRACK = "#181a1f"                       # the unfilled part of a box
_HUES = {                                # a box's colour comes from the deepest group name found here
    "recordings": "#f59e0b", "uploaded": "#f59e0b", "imported": "#fbbf24",
    "inflated": "#38bdf8", "chatterbox": "#38bdf8", "f5_tts": "#a78bfa", "xtts_v2": "#34d399", "unknown": "#94a3b8",
}
_GRAPHIC_HEIGHT = 340


def node_hue(node: dict) -> str:
    for label in reversed(node["path"]):
        if label in _HUES:
            return _HUES[label]
    return "#94a3b8"


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha:.2f})"


def fill_fraction(node: dict) -> float:
    """Share of the group's audio that is switched on (0..1)."""
    return node["active_seconds"] / node["seconds"] if node["seconds"] else 0.0


def tank_background(node: dict, direction: str = "to top") -> str:
    """A CSS background that fills the box to the switched-on share, in the group's colour."""
    pct = round(fill_fraction(node) * 100, 1)
    hue = node_hue(node)
    return f"linear-gradient({direction}, {rgba(hue, 0.62)} {pct}%, {_TRACK} {pct}%)"


def toggle_icon(node: dict) -> str:
    if node["active_clips"] == 0:
        return "check_box_outline_blank"
    return "check_box" if node["active_clips"] == node["clips"] else "indeterminate_check_box"


def box_tooltip(node: dict) -> str:
    return (f"{' › '.join(node['path']) or 'all data'} · {node['active_clips']}/{node['clips']} clips on · "
            f"{node['active_seconds'] / 60:.1f}/{node['seconds'] / 60:.1f} min  —  click to switch the whole group")


def gauge_html(active_minutes: float) -> str:
    pct = lambda m: min(m / _SCALE_MIN, 1.0) * 100
    colour = "#ef4444" if active_minutes < 3 else "#f59e0b" if active_minutes < MIN_RECOGNIZABLE_MIN else "#84cc16" if active_minutes < MIN_STRONG_MIN else "#22c55e"
    ticks = "".join(
        f'<div style="position:absolute;left:{pct(m):.1f}%;top:0;bottom:0;border-left:1px solid #555"></div>'
        f'<div style="position:absolute;left:{pct(m):.1f}%;top:20px;transform:translateX(-50%);color:#777;font:9px monospace">{m:.0f}m<br>{label}</div>'
        for m, label in ((MIN_RECOGNIZABLE_MIN, "voice"), (MIN_STRONG_MIN, "strong"), (_SCALE_MIN, "great")))
    return (f'<div style="position:relative;height:12px;background:#1b1b1b;border-radius:6px;margin:4px 0 34px">'
            f'<div style="width:{pct(active_minutes):.1f}%;height:100%;background:{colour};border-radius:6px"></div>{ticks}</div>')


def build_activation_panel(name: str, *, on_change: Callable[[], None], goto_references: Callable[[], None]) -> Callable[[], None]:
    """Build the panel into the current slot; returns refresh()."""
    readout = ui.label("").classes("text-[#bbb] font-mono text-[12px]")
    note = ui.label("").classes("text-[#555] font-mono text-[10px] leading-snug")
    gauge = ui.html("", sanitize=False).classes("w-full")
    graphic = ui.element("div").classes("w-full").style(
        f"display:flex; gap:8px; height:{_GRAPHIC_HEIGHT}px; align-items:stretch")
    ui.label("Click a box to switch it — and everything inside it — on or off for training. "
             "Box size = audio length; the coloured fill = how much is on.").classes(_DIM)
    with ui.row().classes("w-full items-center gap-2 flex-wrap mt-1"):
        ui.button("EVERYTHING", on_click=lambda: preset("all")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest")
        ui.button("RECORDINGS ONLY", on_click=lambda: preset("recordings")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest").tooltip(
            "Closest to the real voice; needs enough real audio to train on")
        ui.button("INFLATED ONLY", on_click=lambda: preset("inflated")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest").tooltip(
            "Steadiest data, but the model can't be better than the zero-shot clone it learns from")
        ui.space()
        ui.button("MANAGE IN REFERENCES", icon="folder_open", on_click=goto_references).props("flat dense color=amber").classes("font-mono text-[10px] tracking-widest")

    def changed():
        refresh()
        on_change()

    def preset(kind: str):
        rows = refs.clip_rows(name)
        rec = [r["key"] for r in rows if r["origin"] == "recording"]
        inf = [r["key"] for r in rows if r["origin"] == "inflated"]
        if kind == "all":
            refs.set_active(name, rec + inf, True)
        elif kind == "recordings":
            refs.set_active(name, rec, True)
            refs.set_active(name, inf, False)
        else:
            refs.set_active(name, inf, True)
            refs.set_active(name, rec, False)
        changed()

    def flip(node: dict):
        refs.set_active(name, node["keys"], node["active_clips"] < node["clips"])      # none/partly on -> all on; all on -> off
        changed()

    # ── the nested boxes ─────────────────────────────────────────────────
    def clickable(el, node: dict):
        el.props(f'data-node-toggle="{node["id"]}"')
        el.on("click.stop", lambda n=node: flip(n))
        el.style("cursor:pointer")
        with el:
            ui.tooltip(box_tooltip(node))

    def label_block(node: dict, big: bool):
        off = node["active_clips"] == 0
        with ui.element("div").style(f"min-width:0; opacity:{0.5 if off else 1}"):
            ui.label(node["label"]).style(
                f"font:{'600 12px' if big else '11px'} monospace; color:#f1f5f9; text-shadow:0 1px 2px #000; "
                "white-space:nowrap; overflow:hidden; text-overflow:ellipsis")
            ui.label(f"{node['seconds'] / 60:.1f} min · {node['clips']} clips").style(
                "font:9px monospace; color:#cbd5e1; text-shadow:0 1px 2px #000; white-space:nowrap; overflow:hidden; text-overflow:ellipsis")

    def draw_leaf(node: dict, grow_basis: str):
        hue = node_hue(node)
        tile = ui.element("div").style(
            f"flex:{max(node['seconds'], 1) / 60:.3f} 1 {grow_basis}; min-height:54px; min-width:80px; border-radius:6px; "
            f"padding:6px 8px; display:flex; align-items:flex-start; justify-content:space-between; gap:6px; "
            f"background:{tank_background(node)}; border:1px solid {rgba(hue, 0.45)}; transition:filter .15s")
        tile.classes("hover:brightness-125")
        clickable(tile, node)
        with tile:
            label_block(node, big=False)
            ui.icon(toggle_icon(node), size="15px").style("color:#e2e8f0; opacity:.85")

    def draw_group(node: dict, depth: int):
        """A box with a header you can click and its children inside it."""
        hue = node_hue(node)
        flex = f"flex:{max(node['seconds'], 1) / 60:.3f} 1 0" if depth else "flex:1 1 0"
        box = ui.element("div").style(
            f"{flex}; min-width:{150 if depth == 1 else 110}px; min-height:{80}px; display:flex; flex-direction:column; "
            f"border-radius:{10 if depth == 1 else 8}px; border:1px solid {rgba(hue, 0.5)}; "
            f"background:{rgba(hue, 0.07)}; overflow:hidden")
        with box:
            head = ui.element("div").style(
                f"display:flex; align-items:center; justify-content:space-between; gap:6px; padding:6px 8px; "
                f"background:{rgba(hue, 0.16)}; position:relative")
            clickable(head, node)
            with head:
                label_block(node, big=depth == 1)
                ui.icon(toggle_icon(node), size="16px").style("color:#e2e8f0; opacity:.9")
                ui.element("div").style(                                      # fill bar along the header's bottom edge
                    f"position:absolute; left:0; right:0; bottom:0; height:3px; background:{tank_background(node, 'to right')}")
            leaves_only = all(not c["children"] for c in node["children"])
            body = ui.element("div").style(
                "flex:1 1 0; min-height:0; padding:6px; gap:6px; display:flex; "
                + ("flex-wrap:wrap; align-content:stretch;" if leaves_only and depth >= 2 else "flex-direction:column;"))
            with body:
                for child in node["children"]:
                    if child["children"]:
                        draw_group(child, depth + 1)
                    else:
                        draw_leaf(child, "90px" if depth >= 2 else "0")

    def render_graphic(tree: dict):
        graphic.clear()
        with graphic:
            if not tree["children"]:
                ui.label("No reference clips yet — add some on the REFERENCES tab.").classes(_DIM)
            for node in tree["children"]:
                draw_group(node, 1) if node["children"] else draw_leaf(node, "0")

    def refresh():
        rows = refs.clip_rows(name)
        stats = refs.summarize(rows)
        a = stats["active"]
        mins = a["seconds"] / 60
        inf_share = stats["active_inflated"]["seconds"] / a["seconds"] if a["seconds"] else 0.0
        readout.set_text(f"{a['clips']} clips · {mins:.1f} min active  ({100 - inf_share * 100:.0f}% recordings / {inf_share * 100:.0f}% inflated)")
        msgs = []
        if mins < MIN_RECOGNIZABLE_MIN:
            msgs.append(f"under {MIN_RECOGNIZABLE_MIN:.0f} min — fine-tunes usually need more for a recognizable voice")
        if inf_share > 0.75 and stats["active_recordings"]["clips"]:
            msgs.append("mostly generated audio: the model can't beat the clone it learns from")
        if stats["active_recordings"]["clips"] == 0:
            msgs.append("no real recordings active")
        if stats["long"]:
            msgs.append(f"{stats['long']} clip(s) over {stats['long_threshold']:.0f} s are left out of runs to avoid out-of-memory crashes — "
                        "split them in REFERENCES")
        note.set_text("  ·  ".join(msgs))
        note.classes(replace=("text-amber-500" if msgs else "text-[#555]") + " font-mono text-[10px] leading-snug")
        gauge.set_content(gauge_html(mins))
        render_graphic(refs.group_tree(rows))

    refresh()
    return refresh
