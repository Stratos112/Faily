"""The ABOUT tab: who this character is, what it can do today, and where it needs work."""
from typing import Callable

from nicegui import ui

from faily.modules.profile import character_profile

_DIM = "text-[#444] font-mono text-[10px] tracking-wide"
_STATE = {          # capability state -> (icon, colour class)
    "ready": ("check_circle", "text-emerald-500"),
    "partial": ("adjust", "text-amber-500"),
    "missing": ("radio_button_unchecked", "text-[#444]"),
}
_LEVEL = {          # recommendation level -> (icon, colour class)
    "warn": ("warning", "text-amber-500"),
    "todo": ("flag", "text-sky-500"),
    "info": ("info_outline", "text-[#666]"),
    "ok": ("check_circle", "text-emerald-500"),
}
_ACTIONS = {"references": ("References", "folder_open"), "training": ("Training", "model_training"),
            "qualify": ("Qualify…", "cleaning_services")}


def _minutes(seconds: float) -> str:
    return f"{seconds:.0f} s" if seconds < 90 else f"{seconds / 60:.1f} min"


def _label(text: str):
    ui.label(text).classes("text-[#444] font-mono text-[10px] tracking-widest mt-3")


def build_about_panel(name: str, *, goto: Callable[[str], None], open_character: Callable[[str], None],
                      qualify: Callable[[], None], edit: Callable[[], None], delete: Callable[[], None]) -> None:
    """goto("references" | "training") switches the character's tab."""
    p = character_profile(name)
    if not p:
        ui.label("Character not found.").classes("text-red-500 font-mono text-[10px]")
        return
    st = p["stats"]

    # ── identity ─────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-baseline gap-4 flex-wrap"):
        ui.label(f"created {p['created'][:10] or '—'}").classes(_DIM)
        if p["parent"]:
            ui.label(f"variant of {p['parent']}").classes("text-[#777] font-mono text-[10px]")
        if p["is_base"]:
            ui.label(f"{p['favorites']} favorites · {p['history']} generated clips").classes(_DIM)

    # ── readiness ─────────────────────────────────────────────────────────
    _label("WHAT IT CAN DO")
    with ui.row().classes("w-full gap-2 flex-wrap"):
        for cap in p["capabilities"]:
            icon, colour = _STATE[cap["state"]]
            with ui.column().classes("gap-0 p-2 rounded bg-[#111] border border-[#1e1e1e] w-[240px]"):
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon(icon, size="14px").classes(colour)
                    ui.label(cap["label"]).classes("text-[#bbb] font-mono text-[11px]")
                ui.label(cap["detail"]).classes("text-[#666] font-mono text-[10px] leading-snug")

    # ── needs attention ───────────────────────────────────────────────────
    _label("WHERE IT NEEDS WORK")
    with ui.column().classes("w-full gap-1"):
        for rec in p["recommendations"]:
            icon, colour = _LEVEL[rec["level"]]
            with ui.row().classes("w-full items-start gap-2 no-wrap px-2 py-1 rounded bg-[#0d0d0d]"):
                ui.icon(icon, size="14px").classes(colour + " mt-0.5 shrink-0")
                ui.label(rec["text"]).classes("text-[#aaa] font-mono text-[11px] leading-snug flex-grow")
                action = rec.get("action")
                if action in _ACTIONS:
                    label, icn = _ACTIONS[action]
                    handler = qualify if action == "qualify" else (lambda a=action: goto(a))
                    ui.button(label, icon=icn, on_click=handler).props("flat dense color=amber").classes(
                        "font-mono text-[10px] shrink-0")

    # ── the voice ─────────────────────────────────────────────────────────
    m = p["model"]
    _label("VOICE MODEL")
    if m:
        with ui.row().classes("w-full gap-6 flex-wrap items-start"):
            with ui.column().classes("gap-0"):
                ui.label(m["name"]).classes("text-amber-400 font-mono text-sm")
                ui.label(f"epoch {m['epoch']} · base voice {m['base_voice'] or '—'}").classes(_DIM)
                if m["clips"]:
                    ui.label(f"trained on {m['clips']} clips · {m['minutes']} min").classes(_DIM)
                if m["count"] > 1:
                    ui.label(f"{m['count']} saved models (MODELS tab)").classes(_DIM)
            if m["accuracy"] is not None:
                with ui.column().classes("gap-1 w-56"):
                    ui.label(f"pronunciation {m['accuracy']:.0%}").classes("text-[#aaa] font-mono text-[11px]")
                    ui.linear_progress(value=m["accuracy"], show_value=False, size="6px").props(
                        "color=" + ("positive" if m["accuracy"] >= 0.9 else "amber" if m["accuracy"] >= 0.75 else "negative")
                    ).classes("w-full")
                    if m["words_per_sec"]:
                        ui.label(f"{m['words_per_sec']:.1f} words/s when speaking").classes(_DIM)
    else:
        ui.label("No Piper voice trained yet.").classes(_DIM)

    # ── data at a glance ──────────────────────────────────────────────────
    _label("DATA")
    a, al = st["active"], st["all"]
    with ui.row().classes("w-full gap-6 flex-wrap"):
        for title, value, sub in [
            ("ACTIVE", f"{a['clips']} clips", _minutes(a["seconds"])),
            ("ALL", f"{al['clips']} clips", _minutes(al["seconds"])),
            ("RECORDINGS", f"{st['recordings']['clips']}", _minutes(st["recordings"]["seconds"])),
            ("INFLATED", f"{st['inflated']['clips']}", _minutes(st["inflated"]["seconds"])),
            ("FROM PARENT", f"{p['inherited']['clips']}", _minutes(p["inherited"]["seconds"])),
        ]:
            with ui.column().classes("gap-0"):
                ui.label(title).classes("text-[#444] font-mono text-[9px] tracking-widest")
                ui.label(value).classes("text-[#bbb] font-mono text-[12px]")
                ui.label(sub).classes("text-[#555] font-mono text-[10px]")
    if st["by_backend"]:
        ui.label("inflated by backend: " + " · ".join(f"{k} {v['clips']}" for k, v in st["by_backend"].items())).classes(_DIM + " mt-1")

    if p["variants"]:
        _label("VARIANTS")
        with ui.row().classes("gap-2 flex-wrap"):
            for v in p["variants"]:
                ui.button(f"↳ {v}", on_click=lambda v=v: open_character(v)).props("flat dense color=grey").classes(
                    "font-mono text-[10px]")

    ui.separator().classes("mt-4 mb-3 opacity-20")
    with ui.row().classes("gap-2"):
        ui.button("EDIT", icon="edit", on_click=edit).props("flat color=grey")
        ui.button("DELETE", icon="delete_outline", on_click=delete).props("flat color=negative")
