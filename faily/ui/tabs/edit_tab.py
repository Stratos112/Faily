import shutil
import time
from pathlib import Path
from nicegui import ui, run as ni_run
from faily.modules.edit import apply_edits, audio_info
from faily.core.characters import (
    list_characters, get_character, add_ref_clip, add_clip_to_character,
    CHARACTERS_DIR, list_character_clips, list_character_favorites,
)
from faily.ui.components import section_label, show_error

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"
EDIT_DIR = Path("outputs/edit")

_TRIM_JS = """
<script>
function faily_trim_update(sp, ep) {
    var el = document.getElementById('faily-trim-kept');
    if (!el) return;
    el.style.left = sp.toFixed(1) + '%';
    el.style.width = Math.max(0, 100 - sp - ep).toFixed(1) + '%';
}
</script>
"""

_WF_H = 68  # waveform canvas height px
_WF_N = 500  # number of RMS columns


def _waveform_html(path=None) -> str:
    """Filled SVG waveform + trim overlay div. path=None → empty placeholder."""
    overlay = (
        '<div id="faily-trim-kept" style="position:absolute;top:0;left:0;height:100%;width:100%;'
        'background:rgba(245,158,11,0.10);border-left:2px solid rgba(245,158,11,0.45);'
        'border-right:2px solid rgba(245,158,11,0.45);pointer-events:none;"></div>'
    )
    wrap_open = (
        f'<div style="position:relative;height:{_WF_H}px;background:#0d0d0d;'
        f'border-radius:3px;overflow:hidden;border:1px solid #1a1a1a;">'
    )
    if path is None or not Path(path).exists():
        return wrap_open + overlay + '</div>'
    try:
        import numpy as np
        import soundfile as sf
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        n = min(_WF_N, max(1, len(data)))
        chunks = np.array_split(data, n)
        peaks = np.array([float(np.sqrt(np.mean(c ** 2))) if len(c) else 0.0 for c in chunks])
        mx = peaks.max()
        if mx > 0:
            peaks /= mx
        W, H = 500, _WF_H
        mid = H / 2
        margin = 4
        top_pts = [(i * W / n + W / n / 2, mid - max(1.0, peaks[i] * (mid - margin))) for i in range(n)]
        bot_pts = [(i * W / n + W / n / 2, mid + max(1.0, peaks[i] * (mid - margin))) for i in range(n)]
        poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in top_pts + list(reversed(bot_pts)))
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'preserveAspectRatio="none" width="100%" height="{H}" '
            f'style="position:absolute;top:0;left:0;">'
            f'<line x1="0" y1="{mid:.1f}" x2="{W}" y2="{mid:.1f}" '
            f'stroke="rgba(80,80,80,0.25)" stroke-width="0.5"/>'
            f'<polygon points="{poly_pts}" fill="rgba(160,120,40,0.45)"/>'
            f'</svg>'
        )
        return wrap_open + svg + overlay + '</div>'
    except Exception:
        return wrap_open + overlay + '</div>'


def build_edit_tab():
    """Returns send_to_edit(path, char_name=None) callable."""

    ui.add_head_html(_TRIM_JS)

    # ── state ────────────────────────────────────────────────────────────────
    _src:      list[Path | None] = [None]
    _src_char: list[str | None]  = [None]
    _preview:  list[Path | None] = [None]
    _channels: list[int]         = [1]
    _duration: list[float]       = [0.0]

    _vol_db:       list[float] = [0.0]
    _speed:        list[float] = [1.0]
    _pitch:        list[float] = [0.0]
    _trim_start:   list[float] = [0.0]
    _trim_end:     list[float] = [0.0]
    _trim_silence: list[bool]  = [False]
    _mono:         list[bool]  = [False]
    _stereo:       list[bool]  = [False]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _char_opts() -> dict[str, str]:
        opts = {_NO_CHAR: _NO_CHAR}
        for c in list_characters():
            opts[c["name"]] = c["name"]
        return opts

    def _update_trim_bar():
        dur = _duration[0]
        if dur <= 0:
            ui.run_javascript("faily_trim_update(0, 0)")
            return
        sp = min((_trim_start[0] / dur) * 100, 100)
        ep = min((_trim_end[0] / dur) * 100, 100)
        if sp + ep > 99:
            sp = max(0, 99 - ep)
        ui.run_javascript(f"faily_trim_update({sp:.1f}, {ep:.1f})")

    def _load_source(path: Path):
        _src[0] = path
        _preview[0] = None
        try:
            rel = path.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"
        except ValueError:
            url = f"/outputs/edit/{path.name}"
        src_player.set_source(url)
        src_now_playing.set_text(f"▶  {path.stem}")
        try:
            info = audio_info(path)
            _channels[0] = info["channels"]
            _duration[0] = info["duration"]
            dur = f"{info['duration']:.2f}s"
            ch  = "stereo" if info["channels"] > 1 else "mono"
            src_info.set_text(f"{path.name}  ·  {dur}  ·  {info['sample_rate']} Hz  ·  {ch}")
        except Exception:
            _channels[0] = 1
            _duration[0] = 0.0
            src_info.set_text(path.name)
        # dynamic mono/stereo checkbox
        _mono[0] = False
        _stereo[0] = False
        mono_check.set_value(False)
        if _channels[0] > 1:
            mono_check._props['label'] = "SUM TO MONO  (average L+R into one channel)"
        else:
            mono_check._props['label'] = "COPY TO STEREO  (duplicate to both channels)"
        mono_check.update()
        # waveform
        waveform_el.set_content(_waveform_html(path))
        # update trim sliders
        dur_max = max(_duration[0], 1.0)
        trim_s_slider.props(f"max={dur_max:.1f}")
        trim_e_slider.props(f"max={dur_max:.1f}")
        trim_s_slider.set_value(0.0)
        trim_e_slider.set_value(0.0)
        ts_lbl.set_text("0.0s")
        te_lbl.set_text("0.0s")
        _trim_start[0] = 0.0
        _trim_end[0] = 0.0
        ui.run_javascript("faily_trim_update(0, 0)")
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

    async def _do_preview():
        if _src[0] is None:
            ui.notify("Load a source file first", type="warning")
            return
        EDIT_DIR.mkdir(parents=True, exist_ok=True)
        out = EDIT_DIR / f"_preview_{int(time.time())}.wav"
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
                mono=_mono[0],
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
        _vol_db[0] = 0.0;  vol_lbl.set_text("0 dB");   vol_slider.set_value(0.0)
        _speed[0]  = 1.0;  spd_lbl.set_text("1.00×");  spd_slider.set_value(1.0)
        _pitch[0]  = 0.0;  pit_lbl.set_text("0.0 st");  pit_slider.set_value(0.0)
        _trim_start[0] = 0.0;  trim_s_slider.set_value(0.0);  ts_lbl.set_text("0.0s")
        _trim_end[0]   = 0.0;  trim_e_slider.set_value(0.0);  te_lbl.set_text("0.0s")
        _trim_silence[0] = False;  silence_chk.set_value(False)
        _mono[0]  = False;  _stereo[0] = False;  mono_check.set_value(False)
        ui.run_javascript("faily_trim_update(0, 0)")

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
            from faily.core.settings import get_download_dir
            downloads = get_download_dir()
            downloads.mkdir(parents=True, exist_ok=True)
            stem = _src[0].stem if _src[0] else "edit"
            dest = downloads / f"{stem}_edited_{datetime.datetime.now():%H%M%S}.wav"
            shutil.copy2(str(_preview[0]), str(dest))
            ui.notify(f"Saved to {downloads.name}/{dest.name}", type="positive", timeout=3000)
        except Exception as exc:
            show_error(exc)

    # ── layout ────────────────────────────────────────────────────────────────
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        # ── left: controls ──────────────────────────────────────────────────
        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            # ── character clip picker ───────────────────────────────────────
            section_label("LOAD FROM CHARACTER")
            with ui.row().classes("w-full items-center gap-2"):
                char_pick = (
                    ui.select(options=_char_opts(), value=_NO_CHAR)
                    .props("outlined dark dense")
                    .classes("flex-grow")
                )
                ui.button(
                    icon="refresh",
                    on_click=lambda: char_pick.set_options(_char_opts(), value=_NO_CHAR),
                ).props("flat dense color=grey")

            pick_list = ui.column().classes("w-full gap-0.5 max-h-40 overflow-y-auto")

            def _pick_row(path: Path, prefix: str = "•"):
                with pick_list:
                    with ui.row().classes(
                        "w-full items-center gap-2 px-2 py-0.5 hover:bg-[#1a1a1a] rounded cursor-pointer"
                    ).on("click", lambda p=path: _load_source(p)):
                        ui.label(prefix).classes("text-amber-500 font-mono text-[10px] shrink-0 w-3")
                        ui.label(path.stem).classes(
                            "text-[#666] font-mono text-[10px] truncate flex-grow"
                        )

            def _on_char_pick(name: str):
                pick_list.clear()
                if name == _NO_CHAR:
                    return
                char = get_character(name)
                if not char:
                    return
                char_dir = CHARACTERS_DIR / name
                if "ref_audio" in char:
                    ra = char_dir / char["ref_audio"]
                    if ra.exists():
                        _pick_row(ra, "★")
                for rc in char.get("ref_clips", []):
                    rp = char_dir / rc["file"]
                    if rp.exists():
                        _pick_row(rp, "●")
                for c in list_character_clips(name):
                    _pick_row(c, "◆")
                for f in list_character_favorites(name):
                    _pick_row(f, "♥")

            char_pick.on_value_change(lambda e: _on_char_pick(e.value))

            ui.separator().classes("my-1 opacity-20")

            # ── source info ─────────────────────────────────────────────────
            section_label("SOURCE")
            src_player = ui.audio("").classes("w-full rounded")
            src_now_playing = ui.label("").classes("text-[#333] font-mono text-[9px]")
            src_info = ui.label("no file loaded").classes(
                "text-[#444] font-mono text-[10px] tracking-wide"
            )

            ui.separator().classes("my-1 opacity-20")

            # ── trim ─────────────────────────────────────────────────────────
            section_label("TRIM")
            waveform_el = ui.html(_waveform_html()).classes("w-full")

            with ui.row().classes("w-full items-center gap-3 mt-1"):
                ts_lbl = ui.label("0.0s").classes(
                    "font-mono text-[9px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_ts(e):
                    _trim_start[0] = float(e.value or 0)
                    ts_lbl.set_text(f"{_trim_start[0]:.1f}s")
                    _update_trim_bar()
                trim_s_slider = ui.slider(
                    min=0.0, max=300.0, step=0.1, value=0.0, on_change=_on_ts,
                ).classes("flex-grow").props("color=amber label-always")

            with ui.row().classes("w-full items-center gap-3"):
                te_lbl = ui.label("0.0s").classes(
                    "font-mono text-[9px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_te(e):
                    _trim_end[0] = float(e.value or 0)
                    te_lbl.set_text(f"{_trim_end[0]:.1f}s")
                    _update_trim_bar()
                trim_e_slider = ui.slider(
                    min=0.0, max=300.0, step=0.1, value=0.0, on_change=_on_te,
                ).classes("flex-grow").props("color=amber label-always")

            with ui.row().classes("w-full items-center gap-4 mt-1"):
                ui.label("START").classes("text-[#333] font-mono text-[9px] w-10 text-right shrink-0")
                ui.label("END (from end)").classes("text-[#333] font-mono text-[9px]")

            silence_chk = ui.checkbox(
                "TRIM SILENCE  (auto-detect quiet edges)",
                on_change=lambda e: _trim_silence.__setitem__(0, bool(e.value)),
            ).classes("font-mono text-[10px] text-[#555] mt-1")

            ui.separator().classes("my-1 opacity-20")

            # ── adjustments ──────────────────────────────────────────────────
            section_label("VOLUME")
            with ui.row().classes("w-full items-center gap-3"):
                vol_lbl = ui.label("0 dB").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_vol(e):
                    _vol_db[0] = float(e.value)
                    vol_lbl.set_text(f"{e.value:+.1f} dB" if e.value != 0 else "0 dB")
                vol_slider = ui.slider(
                    min=-20, max=20, step=0.5, value=0, on_change=_on_vol,
                ).classes("flex-grow").props("color=amber")

            section_label("SPEED  (changes pitch — tape effect)")
            with ui.row().classes("w-full items-center gap-3"):
                spd_lbl = ui.label("1.00×").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_spd(e):
                    _speed[0] = float(e.value); spd_lbl.set_text(f"{e.value:.2f}×")
                spd_slider = ui.slider(
                    min=0.5, max=2.0, step=0.01, value=1.0, on_change=_on_spd,
                ).classes("flex-grow").props("color=amber")

            section_label("PITCH  (semitones, preserves duration)")
            with ui.row().classes("w-full items-center gap-3"):
                pit_lbl = ui.label("0.0 st").classes(
                    "font-mono text-[10px] text-amber-400 w-14 shrink-0 text-right"
                )
                def _on_pit(e):
                    _pitch[0] = float(e.value)
                    pit_lbl.set_text(f"{e.value:+.1f} st" if e.value != 0 else "0.0 st")
                pit_slider = ui.slider(
                    min=-12.0, max=12.0, step=0.1, value=0.0, on_change=_on_pit,
                ).classes("flex-grow").props("color=amber")

            ui.separator().classes("my-1 opacity-20")

            def _on_mono_stereo(e):
                val = bool(e.value)
                if _channels[0] > 1:
                    _mono[0] = val;  _stereo[0] = False
                else:
                    _stereo[0] = val;  _mono[0] = False

            mono_check = ui.checkbox(
                "SUM TO MONO  (average L+R into one channel)",
                on_change=_on_mono_stereo,
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
                char_pick.set_options(opts, value=char_name)
                _on_char_pick(char_name)
            save_char_sel.set_options(opts, value=char_name)
        _load_source(path)

    return send_to_edit
