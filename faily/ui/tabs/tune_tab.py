from nicegui import ui, run as ni_run
from faily.modules.vc import tune_generate, EXPRESSION_ENGINES, STAGE2_BACKENDS
from faily.core.characters import list_characters, get_character, get_ref_chain, build_ref_audio
from faily.ui.components import output_panel, section_label, show_error, model_picker

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"
_DEFAULT_ENGINE = next(iter(EXPRESSION_ENGINES))


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _char_options() -> dict[str, str]:
    opts = {_NO_CHAR: _NO_CHAR}
    for c in list_characters():
        label = f"  ↳ {c['name']}" if "parent" in c else c["name"]
        opts[c["name"]] = label
    return opts


def build_tune_tab():
    _progress: list[float] = [0.0]
    _char_name: list[str] = [_NO_CHAR]
    _engine: list[str] = [_DEFAULT_ENGINE]
    _stage2: list[str] = ["freevc"]
    _normalize_db: list[float] = [-18.0]
    _max_tokens: list[int] = [500]
    _ov_tau:    list[float] = [0.3]
    _svc_steps: list[int]   = [10]
    _svc_cfg:   list[float] = [0.7]
    _out: dict = {}

    def _update_char_info(name: str):
        _char_name[0] = name
        if name == _NO_CHAR:
            char_info.set_text("")
            return
        char = get_character(name)
        chain = get_ref_chain(name)
        ancestry = f"↳ {char['parent']}" if char and "parent" in char else "base character"
        ref_label = f"{len(chain)} ref clip{'s' if len(chain) != 1 else ''}" if chain else "⚠  no reference audio — save from CLONE tab"
        char_info.set_text(f"{ancestry}  ·  {ref_label}")

    def _on_char(e):
        _update_char_info(e.value)

    def _on_engine(key: str):
        _engine[0] = key

    def _on_stage2(key: str):
        _stage2[0] = key
        ov_tau_row.set_visibility(key == "openvoice")
        svc_row.set_visibility(key == "seedvc")

    async def _generate():
        if _char_name[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning")
            return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter a line to speak", type="warning")
            return
        if not get_ref_chain(_char_name[0]):
            ui.notify("Character has no reference audio — save it from the CLONE tab first", type="warning")
            return

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        with build_ref_audio(_char_name[0]) as (ref, _):
            try:
                path = await ni_run.io_bound(
                    tune_generate,
                    text,
                    expression_input.value.strip(),
                    _engine[0],
                    ref,
                    _progress,
                    char_name=_char_name[0] if _char_name[0] != _NO_CHAR else None,
                    normalize_db=_normalize_db[0],
                    max_new_tokens=_max_tokens[0],
                    stage2_backend=_stage2[0],
                    ov_tau=_ov_tau[0],
                    svc_steps=_svc_steps[0],
                    svc_cfg=_svc_cfg[0],
                )
                _out["main_player"].set_source(f"/outputs/vc/{path.name}")
                _out["status"].set_text(f"✓  {path.name}")
                _out["add_to_history"](path, text)
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

    # ── UI ──────────────────────────────────────────────────────────────────
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            _section_row(
                "CHARACTER",
                "The voice to speak in. Characters are created in the CLONE tab. "
                "Their reference audio is used by FreeVC in stage 2.",
            )
            char_select = (
                ui.select(options=_char_options(), value=_NO_CHAR, on_change=_on_char)
                .props("outlined dark dense")
                .classes("w-full")
            )
            char_info = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

            _section_row(
                "EXPRESSION ENGINE",
                "Generates expressive intermediate audio from your text and style description. "
                "Hover each option for details.",
            )
            model_picker(EXPRESSION_ENGINES, _DEFAULT_ENGINE, _on_engine)

            _section_row(
                "VOICE CONVERSION",
                "Applies the character's voice to the expressive intermediate audio. "
                "Hover each option for details.",
            )
            model_picker(STAGE2_BACKENDS, "freevc", _on_stage2)

            # ── OpenVoice tau control ─────────────────────────────────────────
            with ui.column().classes("w-full gap-3") as ov_tau_row:
                _section_row(
                    "TAU",
                    "OpenVoice voice identity strength. Lower = cleaner speech, more original content. "
                    "Higher = stronger voice character but may distort intelligibility. Default 0.10.",
                )
                with ui.row().classes("w-full items-center gap-3"):
                    ov_tau_lbl = ui.label("0.30").classes(
                        "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                    )
                    def _on_ov_tau(e): _ov_tau[0] = float(e.value); ov_tau_lbl.set_text(f"{e.value:.2f}")
                    ui.slider(min=0.01, max=0.9, step=0.01, value=0.3, on_change=_on_ov_tau).classes(
                        "flex-grow"
                    ).props("color=amber")
            ov_tau_row.set_visibility(False)

            # ── Seed-VC controls ──────────────────────────────────────────────
            with ui.column().classes("w-full gap-3") as svc_row:
                _section_row(
                    "DIFFUSION STEPS",
                    "Seed-VC denoising steps. More steps = higher quality but slower. "
                    "10 is a good balance; raise to 20-30 for final renders.",
                )
                with ui.row().classes("w-full items-center gap-3"):
                    svc_steps_lbl = ui.label("10").classes(
                        "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                    )
                    def _on_svc_steps(e): _svc_steps[0] = int(e.value); svc_steps_lbl.set_text(str(int(e.value)))
                    ui.slider(min=4, max=50, step=1, value=10, on_change=_on_svc_steps).classes("flex-grow").props("color=amber")
                _section_row(
                    "CFG RATE",
                    "Classifier-free guidance scale for Seed-VC. "
                    "Higher values apply the target voice more strongly. 0.7 is a sensible default.",
                )
                with ui.row().classes("w-full items-center gap-3"):
                    svc_cfg_lbl = ui.label("0.70").classes(
                        "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                    )
                    def _on_svc_cfg(e): _svc_cfg[0] = float(e.value); svc_cfg_lbl.set_text(f"{e.value:.2f}")
                    ui.slider(min=0.1, max=1.0, step=0.05, value=0.7, on_change=_on_svc_cfg).classes("flex-grow").props("color=amber")
            svc_row.set_visibility(False)

            _section_row(
                "STYLE DESCRIPTION",
                "Describe how this line should be delivered — tone, emotion, pacing, manner. "
                "This is passed directly to the expression engine as a style prompt. "
                "Examples: 'sing-song and playful', 'cold fury, slow and deliberate', "
                "'breathless and panicked', 'warm but exhausted'. Leave blank for neutral.",
            )
            expression_input = (
                ui.textarea(placeholder="e.g. cold fury, slow and deliberate…")
                .classes("w-full")
                .props("outlined dark rows=3")
            )

            _section_row("LINE", "What the character says.")
            text_input = (
                ui.textarea(placeholder="Enter the line…")
                .classes("w-full")
                .props("outlined dark rows=4")
            )

            _section_row(
                "MAX TOKENS",
                "Parler generation length. Lower values produce shorter output that FreeVC handles more cleanly. "
                "Raise if lines are getting cut off.",
            )
            with ui.row().classes("w-full items-center gap-3"):
                tokens_lbl = ui.label("500").classes(
                    "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_tokens(e):
                    _max_tokens[0] = int(e.value)
                    tokens_lbl.set_text(str(int(e.value)))
                ui.slider(min=50, max=1200, step=50, value=500, on_change=_on_tokens).classes(
                    "flex-grow"
                ).props("color=amber")

            _section_row(
                "PRE-CONVERT LEVEL",
                "Normalise Parler output to this dBFS before FreeVC. "
                "Consistent input level reduces distortion in the voice conversion step.",
            )
            with ui.row().classes("w-full items-center gap-3"):
                norm_lbl = ui.label("-18 dBFS").classes(
                    "font-mono text-[10px] text-amber-400 w-16 shrink-0 text-right"
                )
                def _on_norm(e):
                    _normalize_db[0] = float(e.value)
                    norm_lbl.set_text(f"{int(e.value)} dBFS")
                ui.slider(min=-24, max=-3, step=1, value=-18, on_change=_on_norm).classes(
                    "flex-grow"
                ).props("color=amber")

            ui.space()
            gen_btn = (
                ui.button("GENERATE", on_click=_generate)
                .classes(f"w-full {_BTN}")
                .props("color=amber unelevated")
            )

        pb, ml, mp, st, _, _, ath = output_panel(
            "vc",
            get_char_name=lambda: _char_name[0] if _char_name[0] != _NO_CHAR else None,
        )
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    def _tick():
        val = _progress[0]
        if val == 0.0:
            return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(val)

    _poll = ui.timer(0.15, _tick, active=False)

    def refresh_characters():
        opts = _char_options()
        value = _char_name[0] if _char_name[0] in opts else _NO_CHAR
        char_select.set_options(opts, value=value)
        _char_name[0] = value

    def select_character(name: str):
        opts = _char_options()
        if name in opts:
            char_select.set_options(opts, value=name)
            _update_char_info(name)

    return refresh_characters, select_character
