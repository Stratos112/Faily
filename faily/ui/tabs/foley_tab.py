from nicegui import ui, run as ni_run
from faily.core.model_manager import manager
from faily.modules.foley import get_models, generate as foley_generate
from faily.ui.components import output_panel, section_label

_BTN = "font-mono tracking-widest"


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def build_foley_tab():
    models = get_models()
    _prev_id: list[str | None] = [None]
    _progress: list[float] = [0.0]
    _out: dict = {}

    async def _generate():
        prompt = prompt_input.value.strip()
        if not prompt:
            ui.notify("Enter a sound description first", type="warning")
            return
        model_id = get_models().get(model_select.value, model_select.value)
        steps = int(steps_slider.value)
        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        try:
            path = await ni_run.io_bound(
                foley_generate,
                prompt, model_id,
                dur_slider.value, steps, guid_slider.value,
                _progress,
                candidates=int(cand_slider.value),
                normalize=normalize_toggle.value,
                fade=fade_slider.value,
            )
            _prev_id[0] = model_id
            _out["main_player"].set_source(f"/outputs/sfx/{path.name}")
            _out["status"].set_text(f"✓  {path.name}")
            _out["add_to_history"](path)
        except Exception as exc:
            ui.notify(str(exc), type="negative", timeout=8000)
            _out["status"].set_text("error — see notification")
        finally:
            _poll.active = False
            _out["model_loader"].set_visibility(False)
            _out["progress_bar"].set_value(1.0)
            _out["progress_bar"].set_visibility(True)
            await ui.run_javascript("await new Promise(r => setTimeout(r, 400))")
            _out["progress_bar"].set_visibility(False)
            gen_btn.enable()

    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):
            _section_row("MODEL", "Which AudioLDM2 checkpoint to use. Larger models capture richer textures but take longer to load.")

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

            _section_row("PROMPT", "Describe the sound in natural language. Be specific about texture, environment, and intensity — e.g. 'distant thunderstorm over a tin roof'.")
            prompt_input = (
                ui.textarea(placeholder="Describe the sound…  e.g. 'rain on a tin roof'")
                .classes("w-full")
                .props("outlined dark rows=4")
            )

            _section_row("DURATION", "Length of the generated clip, up to 3 minutes. Longer clips require significantly more VRAM and compute time.")
            with ui.row().classes("items-center gap-3 w-full"):
                dur_slider = ui.slider(min=1.0, max=180.0, step=1.0, value=5.0).classes("flex-grow")
                dur_val = ui.label("5 s").classes("text-[#888] font-mono text-xs w-16 text-right")
            dur_val.bind_text_from(
                dur_slider, "value",
                lambda v: f"{int(v)//60}:{int(v)%60:02d}" if v >= 60 else f"{int(v)} s",
            )

            _section_row("STEPS", "Diffusion denoising steps. More steps produce cleaner, more detailed audio but are slower. 50 is a solid default; push to 100–150 for final renders.")
            with ui.row().classes("items-center gap-3 w-full"):
                steps_slider = ui.slider(min=10, max=200, step=10, value=50).classes("flex-grow")
                steps_val = ui.label("50").classes("text-[#888] font-mono text-xs w-12 text-right")
            steps_val.bind_text_from(steps_slider, "value", lambda v: str(int(v)))

            _section_row("GUIDANCE", "Classifier-free guidance scale. Higher values follow the prompt more strictly but can sound over-processed. 3–5 is natural; 7+ is prompt-locked.")
            with ui.row().classes("items-center gap-3 w-full"):
                guid_slider = ui.slider(min=1.0, max=10.0, step=0.5, value=3.5).classes("flex-grow")
                guid_val = ui.label("3.5").classes("text-[#888] font-mono text-xs w-12 text-right")
            guid_val.bind_text_from(guid_slider, "value", lambda v: f"{v:.1f}")

            _section_row("CANDIDATES", "Generate N audio clips per run and keep the best. AudioLDM2 uses its internal CLAP model to score each clip against your prompt and selects the winner. Higher values improve quality at the cost of proportionally more time.")
            with ui.row().classes("items-center gap-3 w-full"):
                cand_slider = ui.slider(min=1, max=4, step=1, value=1).classes("flex-grow")
                cand_val = ui.label("1").classes("text-[#888] font-mono text-xs w-12 text-right")
            cand_val.bind_text_from(cand_slider, "value", lambda v: str(int(v)))

            _section_row("NORMALIZE", "Peak-normalize the output so the loudest sample reaches 0 dBFS. Useful when generated audio is inconsistently quiet or loud.")
            normalize_toggle = ui.switch("").props("color=amber")

            _section_row("FADE", "Apply a symmetric fade-in and fade-out of this duration to the clip. Smooths abrupt starts and endings — ideal for loops or background layers.")
            with ui.row().classes("items-center gap-3 w-full"):
                fade_slider = ui.slider(min=0.0, max=2.0, step=0.1, value=0.0).classes("flex-grow")
                fade_val = ui.label("off").classes("text-[#888] font-mono text-xs w-12 text-right")
            fade_val.bind_text_from(fade_slider, "value", lambda v: "off" if v == 0 else f"{v:.1f} s")

            ui.space()
            gen_btn = ui.button("GENERATE", on_click=_generate).classes(f"w-full {_BTN}").props(
                "color=amber unelevated"
            )

        pb, ml, mp, st, _, _, ath = output_panel("sfx")
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    def _tick():
        val = _progress[0]
        if val == 0.0:
            return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(val)

    _poll = ui.timer(0.15, _tick, active=False)

    def _refresh():
        new = get_models()
        model_select.set_options(list(new.keys()), value=model_select.value)
        models.clear()
        models.update(new)
