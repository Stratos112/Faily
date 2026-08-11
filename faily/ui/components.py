"""Shared UI building blocks."""
import asyncio
import shutil
import traceback
from pathlib import Path
from nicegui import ui

_LBL = "text-[#555] font-mono text-[10px] tracking-widest"
_DIM = "text-[#444] font-mono text-xs"

# registered by build_edit_tab() via set_edit_callback()
_edit_fn: list = [lambda path, char_name=None: None]

# registered by build_daw_tab() via set_daw_callback()
_daw_fn: list = [lambda path, char_name=None: None]


def set_edit_callback(fn) -> None:
    _edit_fn[0] = fn


def set_daw_callback(fn) -> None:
    _daw_fn[0] = fn


def send_to_edit(path: Path, char_name: str | None = None) -> None:
    _edit_fn[0](path, char_name)


def send_to_daw(path: Path, char_name: str | None = None) -> None:
    _daw_fn[0](path, char_name)


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


def model_picker(options: dict, default: str, on_change) -> ui.column:
    """Selectable list with per-option hover tooltips.

    options: {key: {"label": str, "desc": str, "available": bool (default True)}}
    on_change(key: str) — sync or async, called on selection.
    """
    _active = [default]
    col = ui.column().classes("w-full gap-1")

    def _rebuild():
        col.clear()
        with col:
            for key, info in options.items():
                active = _active[0] == key
                available = info.get("available", True)
                border = (
                    "border-amber-500/60 bg-[#1a1500]" if active
                    else "border-[#1e1e1e] hover:border-[#333] hover:bg-[#111]"
                )
                text = (
                    "text-amber-400" if active
                    else "text-[#555]" if not available
                    else "text-[#888]"
                )
                cursor = "cursor-pointer" if available else "cursor-not-allowed"
                with ui.row().classes(
                    f"w-full items-center gap-2 px-3 py-1.5 rounded border {border} {text} {cursor}"
                ).tooltip(info.get("desc", "")) as row:
                    ui.icon(
                        "radio_button_checked" if active else "radio_button_unchecked",
                        size="13px",
                    ).classes("shrink-0")
                    ui.label(info["label"]).classes("font-mono text-[10px] flex-grow")
                    if not available:
                        ui.label("soon").classes("font-mono text-[9px] text-[#333] shrink-0")

                    if available:
                        def _click(k=key):
                            _active[0] = k
                            _rebuild()
                            result = on_change(k)
                            if asyncio.iscoroutine(result):
                                asyncio.ensure_future(result)
                        row.on("click", _click)

    _rebuild()
    return col


def output_panel(output_subdir: str, get_char_name=None, tile_output: bool = False):
    """
    Builds the right-hand output column.

    tile_output: when True, CURRENT OUTPUT renders generated candidates as a
    grid of square tiles (with the same per-clip actions as history rows)
    instead of a single wide audio bar. Use set_candidates() to populate it.

    Returns (progress_bar, model_loader, main_player, status_label,
             compare_player, compare_section, add_history_fn, set_candidates_fn).
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
        if tile_output:
            main_player = None
            status = ui.label("—").classes(_DIM)
            candidates_grid = ui.element("div").classes("w-full grid gap-3").style(
                "grid-template-columns: repeat(auto-fill, minmax(120px, 1fr))"
            )
        else:
            main_player = ui.audio("").classes("w-full rounded")
            status = ui.label("—").classes(_DIM)

        ui.separator().classes("my-1 opacity-20")

        # ── compare ───────────────────────────────────────────────────────
        compare_section = ui.column().classes("gap-2 w-full")
        compare_section.set_visibility(False)
        with compare_section:
            section_label("COMPARE")
            compare_player = ui.audio("").classes("w-full rounded")
            compare_name = ui.label("").classes("text-[#333] font-mono text-[9px]")

        ui.separator().classes("my-1 opacity-20")

        # ── history ───────────────────────────────────────────────────────
        section_label("HISTORY")
        history_scroll = ui.scroll_area().classes("w-full flex-grow").style(
            "max-height:260px; min-height:80px"
        )
        with history_scroll:
            history_col_el = ui.column().classes("w-full gap-1")

    # ── pre-populate from disk ─────────────────────────────────────────────
    output_dir = Path("outputs") / output_subdir
    existing: list[Path] = sorted(output_dir.glob("*.wav"), reverse=True) if output_dir.exists() else []

    def _open_ref_dialog(path: Path, transcript: str):
        name = get_char_name() if get_char_name else None
        if not name:
            ui.notify("No character selected", type="warning")
            return
        from faily.core.characters import get_character, add_ref_clip, list_characters
        if not get_character(name):
            ui.notify(f"'{name}' is not a saved character yet", type="warning")
            return

        all_chars = list_characters()
        options = {c["name"]: c["name"] for c in all_chars}

        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#111] border border-[#2a2a2a] min-w-[400px] gap-3"
        ):
            ui.label("ADD TO REFERENCE POOL").classes(
                "text-amber-400 font-mono text-[10px] tracking-widest"
            )
            ui.label("TRANSCRIPT").classes("text-[#444] font-mono text-[10px] tracking-widest")
            field = (
                ui.textarea(value=transcript)
                .props("outlined dark rows=3")
                .classes("w-full font-mono text-[11px]")
            )
            ui.label("CHARACTER").classes("text-[#444] font-mono text-[10px] tracking-widest")
            target_select = (
                ui.select(options=options, value=name)
                .props("outlined dark dense")
                .classes("w-full")
            )

            def _add():
                try:
                    add_ref_clip(target_select.value, path, field.value.strip())
                    dlg.close()
                    ui.notify(f"Added to {target_select.value}", type="positive", timeout=2000)
                except Exception as exc:
                    show_error(exc)

            with ui.row().classes("w-full justify-end gap-2 mt-1"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("ADD TO REFS", icon="mic", on_click=_add).props(
                    "color=amber unelevated dense"
                ).classes("font-mono text-[10px]")
        dlg.open()

    def _fav(path: Path, transcript: str = ""):
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
            # A favorite is a signal this clip came out well — surface the
            # option to also promote it into the reference pool right away,
            # instead of leaving that action to be found later.
            _open_ref_dialog(path, transcript)
        except Exception as exc:
            show_error(exc)

    def _download_local(path: Path):
        try:
            from faily.core.settings import get_download_dir
            downloads = get_download_dir()
            downloads.mkdir(parents=True, exist_ok=True)
            dest = downloads / path.name
            shutil.copy2(str(path), str(dest))
            ui.notify(f"Saved to {downloads.name}/{path.name}", type="positive", timeout=3000)
        except Exception as exc:
            show_error(exc)

    def _make_row(path: Path, transcript: str = ""):
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
                    ui.button(
                        icon="add_reaction",
                        on_click=lambda p=path, t=transcript: _open_ref_dialog(p, t),
                    ).props("flat dense color=grey").classes("shrink-0").tooltip(
                        "Add to character reference pool"
                    )
                    ui.button(
                        icon="favorite_border",
                        on_click=lambda p=path, t=transcript: _fav(p, t),
                    ).props("flat dense color=grey").classes("shrink-0").tooltip("Favorite")
                ui.button(
                    icon="queue_music",
                    on_click=lambda p=path: _daw_fn[0](
                        p, get_char_name() if get_char_name else None
                    ),
                ).props("flat dense color=grey").classes("shrink-0").tooltip("Send to DAW")
                ui.button(
                    icon="tune",
                    on_click=lambda p=path: _edit_fn[0](
                        p, get_char_name() if get_char_name else None
                    ),
                ).props("flat dense color=grey").classes("shrink-0").tooltip("Send to Edit tab")
                ui.button(icon="file_download", on_click=lambda p=path: _download_local(p)).props(
                    "flat dense color=grey"
                ).classes("shrink-0").tooltip("Copy to Downloads")

    def _load_compare(path: Path):
        rel = path.relative_to(Path("outputs"))
        compare_player.set_source(f"/outputs/{rel.as_posix()}")
        compare_name.set_text(f"▶  {path.stem}")
        compare_section.set_visibility(True)

    def add_to_history(path: Path, transcript: str = ""):
        _make_row(path, transcript)

    def _make_tile(path: Path, transcript: str = ""):
        playing = [False]
        rel = path.relative_to(Path("outputs"))
        with ui.column().classes(
            "relative aspect-square w-full items-center justify-between rounded-lg "
            "border border-[#1e1e1e] hover:border-[#333] bg-[#111] p-2 gap-1"
        ):
            audio_el = ui.audio(f"/outputs/{rel.as_posix()}").props("controls=false").classes("hidden")
            play_btn = ui.button(icon="play_arrow").props("round flat color=amber size=lg")

            def _toggle(a=audio_el, b=play_btn, st=playing):
                if st[0]:
                    a.pause(); b.props("icon=play_arrow"); st[0] = False
                else:
                    a.play(); b.props("icon=pause"); st[0] = True
            play_btn.on_click(_toggle)

            def _on_ended(b=play_btn, st=playing):
                st[0] = False
                b.props("icon=play_arrow")
            audio_el.on("ended", _on_ended)

            ui.label(path.stem).classes(
                "text-[#888] font-mono text-[9px] truncate w-full text-center"
            )
            with ui.row().classes("gap-0 justify-center flex-wrap"):
                if get_char_name is not None:
                    ui.button(
                        icon="add_reaction",
                        on_click=lambda p=path, t=transcript: _open_ref_dialog(p, t),
                    ).props("flat dense color=grey size=sm").classes("shrink-0").tooltip(
                        "Add to character reference pool"
                    )
                    ui.button(
                        icon="favorite_border",
                        on_click=lambda p=path, t=transcript: _fav(p, t),
                    ).props("flat dense color=grey size=sm").classes("shrink-0").tooltip("Favorite")
                ui.button(
                    icon="tune",
                    on_click=lambda p=path: _edit_fn[0](
                        p, get_char_name() if get_char_name else None
                    ),
                ).props("flat dense color=grey size=sm").classes("shrink-0").tooltip("Send to Edit tab")
                ui.button(icon="file_download", on_click=lambda p=path: _download_local(p)).props(
                    "flat dense color=grey size=sm"
                ).classes("shrink-0").tooltip("Copy to Downloads")

    def set_candidates(items: list[tuple[Path, str]]):
        """Populate CURRENT OUTPUT with a grid of square tiles (tile_output mode only)."""
        if not tile_output:
            return
        candidates_grid.clear()
        with candidates_grid:
            for path, transcript in items:
                _make_tile(path, transcript)

    for p in existing[:30]:
        _make_row(p)

    return (
        progress_bar, model_loader, main_player, status,
        compare_player, compare_section, add_to_history, set_candidates,
    )
