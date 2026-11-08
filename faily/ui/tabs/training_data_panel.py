"""The activation layout on the TRAINING tab: what a run will learn from, at a glance.

  gauge      active minutes against the 10 / 30 / 60 minute marks
  treemap    every group of clips as a tile sized by audio length; amber = switched on, grey = off
  mix        a slider for the inflated share of the active audio, plus ALL / RECORDINGS ONLY presets
  group row  one chip per group to flip it on or off

The switches are the same ones the REFERENCES hub manages; this is the view you train from.
"""
from typing import Callable

from nicegui import ui

from faily.core import refs
from faily.modules.profile import MIN_RECOGNIZABLE_MIN, MIN_STRONG_MIN

_DIM = "text-[#444] font-mono text-[10px] tracking-wide"
_ON, _OFF = "#f59e0b", "#374151"
_SCALE_MIN = 60.0                        # the gauge's full width


def group_title(g: dict) -> str:
    if g["origin"] != "inflated":
        return "recordings"
    return f"{g['category'] or 'inflated'} · {g['backend'] or 'unknown'}"


def treemap_data(groups: list[dict]) -> list[dict]:
    """One tile per group, split into an ON part and an OFF part (minutes of audio)."""
    out = []
    for g in groups:
        on, off = g["active_seconds"] / 60, (g["seconds"] - g["active_seconds"]) / 60
        kids = []
        if on > 0:
            kids.append({"name": f"on  {on:.1f} min", "value": round(on, 2), "itemStyle": {"color": _ON}})
        if off > 0:
            kids.append({"name": f"off  {off:.1f} min", "value": round(off, 2), "itemStyle": {"color": _OFF}})
        if kids:
            out.append({"name": group_title(g), "children": kids})
    return out


def treemap_options(groups: list[dict]) -> dict:
    return {
        "backgroundColor": "transparent",
        "tooltip": {"formatter": "{b}"},
        "series": [{
            "type": "treemap", "roam": False, "nodeClick": False, "breadcrumb": {"show": False},
            "left": 0, "right": 0, "top": 0, "bottom": 0,
            "upperLabel": {"show": True, "height": 18, "color": "#cbd5e1", "fontFamily": "monospace", "fontSize": 10},
            "label": {"show": True, "color": "#0b0b0b", "fontFamily": "monospace", "fontSize": 10},
            "levels": [{"itemStyle": {"borderColor": "#0f0f0f", "borderWidth": 3, "gapWidth": 3}},
                       {"itemStyle": {"borderColor": "#0f0f0f", "borderWidth": 1, "gapWidth": 1}}],
            "data": treemap_data(groups),
        }],
    }


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
    tree = ui.echart({}).classes("w-full").style("height: 190px")
    chips = ui.row().classes("w-full gap-1 flex-wrap")

    with ui.row().classes("w-full items-center gap-3 mt-1 flex-wrap"):
        share_lbl = ui.label("").classes("text-amber-400 font-mono text-[11px] w-44")
        share = ui.slider(min=0, max=95, step=5, value=50).classes("w-56").props("color=amber")
        share.on_value_change(lambda e: share_lbl.set_text(f"inflated share  {int(e.value)}%"))
        share_lbl.set_text("inflated share  50%")
        ui.button("APPLY MIX", icon="tune", on_click=lambda: apply_mix()).props("flat dense color=amber").classes("font-mono text-[10px] tracking-widest").tooltip(
            "Switch on every recording, then just enough inflated clips (spread across categories) to make this "
            "share of the active audio. The rest are switched off, not deleted.")
    with ui.row().classes("w-full items-center gap-2 flex-wrap"):
        ui.button("EVERYTHING", on_click=lambda: preset("all")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest")
        ui.button("RECORDINGS ONLY", on_click=lambda: preset("recordings")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest").tooltip(
            "Closest to the real voice; needs enough real audio to train on")
        ui.button("INFLATED ONLY", on_click=lambda: preset("inflated")).props("flat dense color=grey").classes("font-mono text-[10px] tracking-widest").tooltip(
            "Steadiest data, but the model can't be better than the zero-shot clone it learns from")
        ui.space()
        ui.button("MANAGE IN REFERENCES", icon="folder_open", on_click=goto_references).props("flat dense color=amber").classes("font-mono text-[10px] tracking-widest")

    def changed(message: str | None = None):
        if message:
            ui.notify(message, timeout=1800)
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

    def apply_mix():
        res = refs.balance_active(name, share.value / 100)
        changed(f"{res['active_clips']} clips active · {res['share']:.0%} inflated")

    def flip(g: dict):
        refs.set_active(name, g["keys"], g["active_clips"] < g["clips"])      # partly/fully off -> on; fully on -> off
        changed()

    def refresh():
        rows = refs.clip_rows(name)
        stats = refs.summarize(rows)
        groups = refs.groups(rows)
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
        tree.options.clear()
        tree.options.update(treemap_options(groups))
        tree.update()
        chips.clear()
        with chips:
            for g in groups:
                on = g["active_clips"] == g["clips"]
                part = 0 < g["active_clips"] < g["clips"]
                ui.button(f"{group_title(g)}  {g['active_clips']}/{g['clips']}", on_click=lambda g=g: flip(g)).props(
                    f"dense unelevated no-caps color={'amber' if on else 'amber-9' if part else 'grey-9'} text-color={'black' if on else 'grey-5'}"
                ).classes("font-mono text-[10px]").tooltip("Click to flip this group on/off for training")

    refresh()
    return refresh
