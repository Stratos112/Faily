from pathlib import Path
from nicegui import ui, run as ni_run
from faily.core.model_manager import manager
from faily.modules.tts import get_models, generate as tts_generate
from faily.modules.vc import generate as vc_generate, VC_OUTPUT_DIR
from faily.ui.components import output_panel, section_label, show_error

_BTN = "font-mono tracking-widest"
_REFS_DIR = Path("outputs/vc/refs")
_NO_VOICE = "— none —"


def _scan_voices() -> list[str]:
    if not _REFS_DIR.exists():
        return [_NO_VOICE]
    files = sorted(
        f.name for f in _REFS_DIR.iterdir()
        if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}
    )
    return [_NO_VOICE] + files


def build_tts_tab():
    models = get_models()
    _prev_id: list[str | None] = [None]
    _progress: list[float] = [0.0]
    _out: dict = {}  # populated after output_panel() is called below

    # ── define _generate before the button so NiceGUI gets the right context ──
    async def _generate():
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter some text first", type="warning")
            return

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        chosen_voice = voice_select.value
        try:
            if chosen_voice and chosen_voice != _NO_VOICE:
                ref_path = _REFS_DIR / chosen_voice
                path = await ni_run.io_bound(vc_generate, text, ref_path, _progress)
                _out["main_player"].set_source(f"/outputs/vc/{path.name}")
            else:
                model_id = get_models().get(model_select.value, model_select.value)
                path = await ni_run.io_bound(
                    tts_generate, text, model_id, speed_slider.value, _progress
                )
                _prev_id[0] = model_id
                _out["main_player"].set_source(f"/outputs/tts/{path.name}")
            _out["status"].set_text(f"✓  {path.name}")
            _out["add_to_history"](path)
        except Exception as exc:
            show_error(exc)
            _out["status"].set_text("error")
        finally:
            _poll.active = False
            _out["model_loader"].set_visibility(False)
            _out["progress_bar"].set_value(1.0)
            _out["progress_bar"].set_visibility(True)
            await ui.run_javascript("await new Promise(r => setTimeout(r, 400))")
            _out["progress_bar"].set_visibility(False)
            gen_btn.enable()

    # ── build UI ───────────────────────────────────────────────────────────
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-5 p-8 border-r border-[#252525]"):
            section_label("MODEL")

            def _on_model_change(e):
                new_id = get_models().get(e.value, e.value)
                old_id = _prev_id[0]
                if old_id and old_id != new_id and old_id in manager.loaded:
                    manager.unload(old_id)
                _prev_id[0] = None

            with ui.row().classes("w-full gap-2 items-center"):
                model_select = (
                    ui.select(list(models.keys()), value=list(models.keys())[0])
                    .classes("flex-grow")
                    .props("outlined dark dense")
                    .on("update:model-value", _on_model_change)
                )
                ui.button(icon="refresh", on_click=lambda: _refresh()).props("flat dense color=grey")

            section_label("CLONE VOICE")
            with ui.row().classes("w-full gap-2 items-center"):
                voice_select = (
                    ui.select(_scan_voices(), value=_NO_VOICE)
                    .classes("flex-grow")
                    .props("outlined dark dense")
                )
                ui.button(
                    icon="refresh",
                    on_click=lambda: voice_select.set_options(_scan_voices(), value=voice_select.value),
                ).props("flat dense color=grey")

            section_label("TEXT")
            text_input = (
                ui.textarea(placeholder="Enter text to synthesize…")
                .classes("w-full")
                .props("outlined dark rows=7")
            )

            section_label("SPEED")
            with ui.row().classes("items-center gap-3 w-full"):
                speed_slider = ui.slider(min=0.5, max=2.0, step=0.05, value=1.0).classes("flex-grow")
                speed_val = ui.label("1.00×").classes("text-[#888] font-mono text-xs w-14 text-right")
            speed_val.bind_text_from(speed_slider, "value", lambda v: f"{v:.2f}×")

            ui.space()
            gen_btn = ui.button("GENERATE", on_click=_generate).classes(f"w-full {_BTN}").props(
                "color=amber unelevated"
            )

        pb, ml, mp, st, _, _, ath = output_panel("tts")
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    # ── timer created at page-build time so NiceGUI context is intact ─────
    def _tick():
        val = _progress[0]
        if val == 0.0:
            return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].props("indeterminate")
        _out["progress_bar"].set_value(val)

    _poll = ui.timer(0.15, _tick, active=False)

    def _refresh():
        new = get_models()
        model_select.set_options(list(new.keys()), value=model_select.value)
        models.clear()
        models.update(new)
