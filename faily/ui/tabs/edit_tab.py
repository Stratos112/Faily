import shutil
import time
from pathlib import Path
from nicegui import ui, run as ni_run
from faily.modules.edit import apply_edits, audio_info
from faily.core.characters import list_characters, add_ref_clip, add_clip_to_character
from faily.ui.components import section_label, show_error

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"
EDIT_DIR = Path("outputs/edit")


def build_edit_tab():
    """Returns send_to_edit(path, char_name=None) callable."""

    # ── state ────────────────────────────────────────────────────────────────
    _src:      list[Path | None] = [None]
    _src_char: list[str | None]  = [None]
    _preview:  list[Path | None] = [None]
    _channels: list[int]         = [1]

    _vol_db:       list[float] = [0.0]
    _speed:        list[float] = [1.0]
    _pitch:        list[int]   = [0]
    _trim_start:   list[float] = [0.0]
    _trim_end:     list[float] = [0.0]
    _trim_silence: list[bool]  = [False]
    _stereo:       list[bool]  = [False]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _char_opts() -> dict[str, str]:
        opts = {_NO_CHAR: _NO_CHAR}
        for c in list_characters():
            opts[c["name"]] = c["name"]
        return opts

    def _load_source(path: Path):
        _src[0] = path
        _preview[0] = None
        rel = path.relative_to(Path("outputs"))
        src_player.set_source(f"/outputs/{rel.as_posix()}")
        try:
            info = audio_info(path)
            _channels[0] = info["channels"]
            dur = f"{info['duration']:.2f}s"
            ch  = "stereo" if info["channels"] > 1 else "mono"
            src_info.set_text(f"{path.name}  ·  {dur}  ·  {info['sample_rate']} Hz  ·  {ch}")
        except Exception:
            _channels[0] = 1
            src_info.set_text(path.name)
        stereo_check.set_enabled(_channels[0] == 1)
        preview_player.set_source("")
        preview_status.set_text("—")
        save_col.set_visibility(False)

    def _update_save_ui():
        has_preview = _preview[0] is not None
        save_col.set_visibility(has_preview)
        can_overwrite = (
            has_preview and _src[0] is not None
            and str(_src[0]).startswith(str(Path("outputs").absolute()) if Path("outputs").is_absolute() else "outputs")
        )
        overwrite_row.set_visibility(can_overwrite)
        if can_overwrite and _src[0]:
            try:
                overwrite_lbl.set_text(str(_src[0].relative_to(Path("outputs"))))
            except ValueError:
                overwrite_lbl.set_text(str(_src[0]))

    async def _on_upload(e):
        EDIT_DIR.mkdir(parents=True, exist_ok=True)
        dest = EDIT_DIR / e.file.name
        dest.write_bytes(await e.file.read())
        _src_char[0] = None
        _load_source(dest)

    async def _do_preview():
        if _src[0] is None:
            ui.notify("Load a source file first", type="warning")
            return
        EDIT_DIR.mkdir(parents=True, exist_ok=True)
        out = EDIT_DIR / f"_preview_{int(time.time())}.wav"
        # clean stale previews
        for old in EDIT_DIR.glob("_preview_*.wav"):
            if old != out:
                try: old.unlink()
                except OSError: pass

        preview_btn.disable()
        preview_status.set_text("processing…")
        try:
            await ni_run.io_bound(
                apply_edits, _src[0], out,
                volume_db=_vol_db[0],
                speed=_speed[0],
                pitch_semitones=_pitch[0],
                trim_start=_trim_start[0],
                trim_end=_trim_end[0],
                trim_silence=_trim_silence[0],
                stereo=_stereo[0],
            )
            _preview[0] = out
            preview_player.set_source(f"/outputs/edit/{out.name}")
            try:
                info = audio_info(out)
                ch = "stereo" if info["channels"] > 1 else "mono"
                preview_status.set_text(f"✓  {info['duration']:.2f}s  ·  {ch}")
            except Exception:
                preview_status.set_text("✓  done")
            _update_save_ui()
        except Exception as exc:
            show_error(exc)
            preview_status.set_text("error")
        finally:
            preview_btn.enable()

    def _reset():
        _vol_db[0] = 0.0;     vol_lbl.set_text("0 dB");   vol_slider.set_value(0.0)
        _speed[0]  = 1.0;     spd_lbl.set_text("1.00×");  spd_slider.set_value(1.0)
        _pitch[0]  = 0;       pit_lbl.set_text("0 st");    pit_slider.set_value(0)
        _trim_start[0] = 0.0; trim_s.set_value(0.0)
        _trim_end[0]   = 0.0; trim_e.set_value(0.0)
        _trim_silence[0] = False; silence_chk.set_value(False)
        _stereo[0]       = False; stereo_check.set_value(False)

    def _save_as_ref():
        if _preview[0] is None:
            ui.notify("Generate a preview first", type="warning"); return
        char = save_char_sel.value
        if char == _NO_CHAR:
            ui.notify("Select a character", type="warning"); return
        try:
            add_ref_clip(char, _preview[0])
            ui.notify(f"Added as ref clip to {char}", type="positive", timeout=2000)
        except Exception as exc:
            show_error(exc)

    def _save_as_clip():
        if _preview[0] is None:
            ui.notify("Generate a preview first", type="warning"); return
        char = save_char_sel.value
        if char == _NO_CHAR:
            ui.notify("Select a character", type="warning"); return
        try:
            add_clip_to_character(char, _preview[0])
            ui.notify(f"Added as clip to {char}", type="positive", timeout=2000)
        except Exception as exc:
            show_error(exc)

    def _confirm_overwrite():
        if _preview[0] is None or _src[0] is None:
            return
        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#3a1a1a] min-w-[380px] gap-3"
        ):
            ui.label("OVERWRITE SOURCE").classes("text-red-400 font-mono text-xs tracking-widest")
            ui.label("Replace:").classes("text-[#666] font-mono text-[10px]")
            ui.label(str(_src[0])).classes("text-[#aaa] font-mono text-[10px] break-all")
            ui.label("with the processed version? This cannot be undone.").classes(
                "text-[#555] font-mono text-[10px]"
            )
            def _do():
                try:
                    shutil.copy2(str(_preview[0]), str(_src[0]))
                    dlg.close()
                    _load_source(_src[0])
                    ui.notify("Source overwritten", type="positive", timeout=2000)
                except Exception as exc:
                    show_error(exc)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("OVERWRITE", on_click=_do).props("color=negative unelevated dense")
        dlg.open()

    def _download():
        if _preview[0] is None:
            ui.notify("Generate a preview first", type="warning"); return
        try:
            import datetime
            downloads = Path.home() / "Downloads"
            downloads.mkdir(parents=True, exist_ok=True)
            stem = _src[0].stem if _src[0] else "edit"
            dest = downloads / f"{stem}_edited_{datetime.datetime.now():%H%M%S}.wav"
            shutil.copy2(str(_preview[0]), str(dest))
            ui.notify(f"Saved to Downloads/{dest.name}", type="positive", timeout=3000)
        except Exception as exc:
            show_error(exc)

    # ── layout ────────────────────────────────────────────────────────────────
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        # ── left: controls ──────────────────────────────────────────────────
        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            section_label("SOURCE")
            src_player = ui.audio("").classes("w-full rounded")
            src_info = ui.label("no file loaded").classes("text-[#444] font-mono text-[10px] tracking-wide")
            (
                ui.upload(on_upload=_on_upload, multiple=False, auto_upload=True)
                .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Upload audio'")
                .classes("w-full mt-1")
            )

            ui.separator().classes("my-1 opacity-20")

            section_label("VOLUME")
            with ui.row().classes("w-full items-center gap-3"):
                vol_lbl = ui.label("0 dB").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_vol(e):
                    _vol_db[0] = float(e.value)
                    vol_lbl.set_text(f"{e.value:+.1f} dB" if e.value != 0 else "0 dB")
                vol_slider = ui.slider(min=-20, max=20, step=0.5, value=0, on_change=_on_vol).classes(
                    "flex-grow"
                ).props("color=amber")

            section_label("SPEED  (changes pitch — tape effect)")
            with ui.row().classes("w-full items-center gap-3"):
                spd_lbl = ui.label("1.00×").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_spd(e):
                    _speed[0] = float(e.value); spd_lbl.set_text(f"{e.value:.2f}×")
                spd_slider = ui.slider(min=0.5, max=2.0, step=0.05, value=1.0, on_change=_on_spd).classes(
                    "flex-grow"
                ).props("color=amber")

            section_label("PITCH  (semitones, preserves duration)")
            with ui.row().classes("w-full items-center gap-3"):
                pit_lbl = ui.label("0 st").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_pit(e):
                    _pitch[0] = int(e.value)
                    pit_lbl.set_text(f"{int(e.value):+d} st" if e.value != 0 else "0 st")
                pit_slider = ui.slider(min=-12, max=12, step=1, value=0, on_change=_on_pit).classes(
                    "flex-grow"
                ).props("color=amber")

            ui.separator().classes("my-1 opacity-20")

            section_label("TRIM START  (seconds from beginning)")
            trim_s = ui.number(
                value=0.0, min=0.0, max=300.0, step=0.1, format="%.1f",
                on_change=lambda e: _trim_start.__setitem__(0, float(e.value or 0)),
            ).props("outlined dark dense").classes("w-full")

            section_label("TRIM END  (seconds from end)")
            trim_e = ui.number(
                value=0.0, min=0.0, max=300.0, step=0.1, format="%.1f",
                on_change=lambda e: _trim_end.__setitem__(0, float(e.value or 0)),
            ).props("outlined dark dense").classes("w-full")

            silence_chk = ui.checkbox(
                "TRIM SILENCE  (auto-detect quiet edges)",
                on_change=lambda e: _trim_silence.__setitem__(0, bool(e.value)),
            ).classes("font-mono text-[10px] text-[#555]")

            ui.separator().classes("my-1 opacity-20")

            stereo_check = ui.checkbox(
                "MAKE STEREO  (copy mono to both channels)",
                on_change=lambda e: _stereo.__setitem__(0, bool(e.value)),
            ).classes("font-mono text-[10px] text-[#555]")

            ui.space()
            with ui.row().classes("w-full gap-2"):
                preview_btn = (
                    ui.button("PREVIEW", on_click=_do_preview)
                    .classes(f"flex-1 {_BTN}").props("color=amber unelevated")
                )
                ui.button("RESET", on_click=_reset).props("flat color=grey").classes(_BTN)

        # ── right: preview + save ───────────────────────────────────────────
        with ui.column().classes("gap-4 p-8 overflow-y-auto"):

            section_label("PROCESSED OUTPUT")
            preview_player = ui.audio("").classes("w-full rounded")
            preview_status = ui.label("—").classes("text-[#444] font-mono text-xs")

            ui.separator().classes("my-2 opacity-20")

            save_col = ui.column().classes("w-full gap-4")
            save_col.set_visibility(False)

            with save_col:
                section_label("SAVE TO CHARACTER")
                save_char_sel = (
                    ui.select(options=_char_opts(), value=_NO_CHAR)
                    .props("outlined dark dense").classes("w-full")
                )
                with ui.row().classes("w-full gap-2"):
                    ui.button(
                        "ADD AS REF CLIP", icon="mic", on_click=_save_as_ref,
                    ).props("color=amber unelevated").classes(f"flex-1 {_BTN}")
                    ui.button(
                        "ADD AS CLIP", icon="library_music", on_click=_save_as_clip,
                    ).props("flat color=grey").classes(f"flex-1 {_BTN}")

                ui.separator().classes("my-1 opacity-20")

                overwrite_row = ui.column().classes("w-full gap-2")
                overwrite_row.set_visibility(False)
                with overwrite_row:
                    section_label("OVERWRITE SOURCE")
                    overwrite_lbl = ui.label("").classes(
                        "text-[#444] font-mono text-[10px] break-all"
                    )
                    ui.button(
                        "OVERWRITE  ⚠", on_click=_confirm_overwrite,
                    ).props("flat color=negative").classes(f"w-full {_BTN}")

                ui.separator().classes("my-1 opacity-20")

                ui.button(
                    "DOWNLOAD TO DOWNLOADS", icon="file_download", on_click=_download,
                ).props("flat color=grey").classes(f"w-full {_BTN}")

    # ── public callback ───────────────────────────────────────────────────────
    def send_to_edit(path: Path, char_name: str | None = None):
        _src_char[0] = char_name
        if char_name and char_name != _NO_CHAR:
            opts = _char_opts()
            if char_name in opts:
                save_char_sel.set_options(opts, value=char_name)
        _load_source(path)
        ui.notify("Loaded in Edit tab — switch to EDIT", timeout=2000)

    return send_to_edit
