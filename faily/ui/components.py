"""Shared UI building blocks."""
import shutil
import traceback
from pathlib import Path
from nicegui import ui

_LBL = "text-[#555] font-mono text-[10px] tracking-widest"
_DIM = "text-[#444] font-mono text-xs"


def section_label(text: str):
    ui.label(text).classes(_LBL)


def show_error(exc: Exception) -> None:
    text = traceback.format_exc() or str(exc)
    with ui.dialog() as dlg, ui.card().classes("bg-[#1a1a1a] border border-[#3a1a1a] max-w-2xl w-full"):
        with ui.row().classes("w-full justify-between items-center mb-2"):
            ui.label("ERROR").classes("text-red-500 font-mono text-xs tracking-widest")
            ui.button(icon="close", on_click=dlg.close).props("flat dense color=grey")
        (
            ui.textarea(value=text)
            .props("readonly outlined dark dense")
            .classes("w-full font-mono text-[11px]")
            .style("min-height:160px")
        )
        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Copy", icon="content_copy", on_click=lambda: ui.clipboard.write(text)).props("flat dense color=amber")
            ui.button("Close", on_click=dlg.close).props("flat dense color=grey")
    dlg.open()


def output_panel(output_subdir: str, get_char_name=None):
    """
    Builds the right-hand output column.

    Returns (progress_bar, model_loader, main_player, status_label,
             compare_player, compare_section, add_history_fn).
    """
    with ui.column().classes("gap-4 p-8 w-full h-full overflow-hidden"):

        # ── state indicators (mutually exclusive) ─────────────────────────
        model_loader = ui.row().classes("items-center gap-3 h-8")
        model_loader.set_visibility(False)
        with model_loader:
            ui.html(
                '<div class="faily-loader">'
                + "".join("<span></span>" * 7)
                + "</div>"
            )
            ui.label("LOADING MODEL").classes(
                "text-amber-500 font-mono text-[10px] tracking-widest"
            )

        progress_bar = (
            ui.linear_progress(value=0, size="4px")
            .props("color=amber")
            .classes("w-full rounded")
        )
        progress_bar.set_visibility(False)

        # ── current output ────────────────────────────────────────────────
        section_label("CURRENT OUTPUT")
        main_player = ui.audio("").classes("w-full rounded")
        status = ui.label("—").classes(_DIM)

        ui.separator().classes("my-1 opacity-20")

        # ── compare ───────────────────────────────────────────────────────
        compare_section = ui.column().classes("gap-2 w-full")
        compare_section.set_visibility(False)
        with compare_section:
            section_label("COMPARE")
            compare_player = ui.audio("").classes("w-full rounded")

        ui.separator().classes("my-1 opacity-20")

        # ── history ───────────────────────────────────────────────────────
        section_label("HISTORY")
        history_scroll = ui.scroll_area().classes("w-full flex-grow").style(
            "max-height:260px; min-height:80px"
        )
        history_col = history_scroll.default_slot.children  # internal list ref

        with history_scroll:
            history_col_el = ui.column().classes("w-full gap-1")

    # ── pre-populate from disk ─────────────────────────────────────────────
    output_dir = Path("outputs") / output_subdir
    existing: list[Path] = sorted(output_dir.glob("*.wav"), reverse=True) if output_dir.exists() else []

    def _add_to_char(path: Path):
        name = get_char_name() if get_char_name else None
        if not name:
            ui.notify("No character selected", type="warning")
            return
        from faily.core.characters import get_character, add_clip_to_character
        if not get_character(name):
            ui.notify(f"'{name}' is not a saved character yet", type="warning")
            return
        try:
            add_clip_to_character(name, path)
            ui.notify(f"Added to {name}", type="positive", timeout=2000)
        except Exception as exc:
            show_error(exc)

    def _fav(path: Path):
        name = get_char_name() if get_char_name else None
        if not name:
            ui.notify("No character selected", type="warning")
            return
        from faily.core.characters import get_character, add_clip_to_favorites
        if not get_character(name):
            ui.notify(f"'{name}' is not a saved character yet", type="warning")
            return
        try:
            add_clip_to_favorites(name, path)
            ui.notify(f"Added to {name} favorites", type="positive", timeout=2000)
        except Exception as exc:
            show_error(exc)

    def _download_local(path: Path):
        try:
            downloads = Path.home() / "Downloads"
            downloads.mkdir(parents=True, exist_ok=True)
            dest = downloads / path.name
            shutil.copy2(str(path), str(dest))
            ui.notify(f"Saved to Downloads/{path.name}", type="positive", timeout=3000)
        except Exception as exc:
            show_error(exc)

    def _make_row(path: Path):
        with history_col_el:
            with ui.row().classes(
                "w-full items-center gap-2 px-3 py-1 rounded "
                "hover:bg-[#1e1e1e] border border-[#1e1e1e] hover:border-[#333]"
            ):
                ui.icon("audio_file", size="14px").classes(
                    "text-amber-500 shrink-0 cursor-pointer"
                ).on("click", lambda p=path: _load_compare(p))
                ui.label(path.name).classes(
                    "text-[#888] font-mono text-[10px] truncate flex-grow cursor-pointer"
                ).on("click", lambda p=path: _load_compare(p))
                if get_char_name is not None:
                    ui.button(icon="add", on_click=lambda p=path: _add_to_char(p)).props(
                        "flat dense color=grey"
                    ).classes("shrink-0").tooltip("Add to character")
                    ui.button(icon="favorite_border", on_click=lambda p=path: _fav(p)).props(
                        "flat dense color=grey"
                    ).classes("shrink-0").tooltip("Favorite")
                ui.button(icon="file_download", on_click=lambda p=path: _download_local(p)).props(
                    "flat dense color=grey"
                ).classes("shrink-0").tooltip("Copy to Downloads")

    def _load_compare(path: Path):
        rel = path.relative_to(Path("outputs"))
        compare_player.set_source(f"/outputs/{rel.as_posix()}")
        compare_section.set_visibility(True)

    def add_to_history(path: Path):
        _make_row(path)

    for p in existing[:30]:
        _make_row(p)

    return progress_bar, model_loader, main_player, status, compare_player, compare_section, add_to_history
