from nicegui import ui, run as ni_run
from pathlib import Path
from faily.modules.vc import (
    tune_generate, EXPRESSION_ENGINES, STAGE2_BACKENDS,
    generate as vc_generate, BACKENDS,
)
from faily.core.characters import list_characters, get_character, get_ref_chain, build_ref_audio
from faily.ui.components import output_panel, section_label, show_error, model_picker

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"
_DEFAULT_ENGINE = next(iter(EXPRESSION_ENGINES))
_VC_DIR = Path("outputs/vc")


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _fmt(val: float, step: float) -> str:
    return str(int(val)) if step >= 1 else f"{val:.2f}"


def _all_chars() -> dict[str, str]:
    opts = {_NO_CHAR: _NO_CHAR}
    for c in list_characters():
        label = f"  ↳ {c['name']}" if "parent" in c else c["name"]
        opts[c["name"]] = label
    return opts


def _rvc_chars() -> dict[str, str]:
    opts = {_NO_CHAR: _NO_CHAR}
    for c in list_characters():
        if "rvc_model" in c:
            opts[c["name"]] = c["name"]
    return opts


def _build_expression(char_state: list[str], _out: dict, _current_char: list[str]):
    """Left controls for EXPRESSION sub-tab. Returns (refresh, select)."""
    _progress:     list[float] = [0.0]
    _engine:       list[str]   = [_DEFAULT_ENGINE]
    _stage2:       list[str]   = ["freevc"]
    _normalize_db: list[float] = [-18.0]
    _max_tokens:   list[int]   = [500]

    def _update_info(name: str):
        char_state[0] = name
        _current_char[0] = name
        if name == _NO_CHAR:
            char_info.set_text(""); return
        char = get_character(name)
        chain = get_ref_chain(name)
        ancestry = f"↳ {char['parent']}" if char and "parent" in char else "base"
        refs = f"{len(chain)} ref{'s' if len(chain) != 1 else ''}" if chain else "⚠  no ref audio"
        char_info.set_text(f"{ancestry}  ·  {refs}")

    async def _generate():
        if char_state[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning"); return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter a line to speak", type="warning"); return
        if not get_ref_chain(char_state[0]):
            ui.notify("Character has no reference audio — save from CLONE first", type="warning"); return
        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True
        with build_ref_audio(char_state[0]) as (ref, _):
            try:
                path = await ni_run.io_bound(
                    tune_generate,
                    text, expr_input.value.strip(), _engine[0], ref, _progress,
                    char_name=char_state[0] if char_state[0] != _NO_CHAR else None,
                    normalize_db=_normalize_db[0],
                    max_new_tokens=_max_tokens[0],
                    stage2_backend=_stage2[0],
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

    with ui.column().classes("gap-3 p-5 overflow-y-auto w-full h-full"):
        _section_row("CHARACTER", "The voice to speak in. Characters are created in the CLONE tab.")
        char_select = (
            ui.select(options=_all_chars(), value=_NO_CHAR, on_change=lambda e: _update_info(e.value))
            .props("outlined dark dense").classes("w-full")
        )
        char_info = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

        _section_row("EXPRESSION ENGINE", "Generates expressive intermediate audio. Hover each option for details.")
        model_picker(EXPRESSION_ENGINES, _DEFAULT_ENGINE, lambda k: _engine.__setitem__(0, k))

        _section_row("VOICE CONVERSION", "Applies the character's voice to the intermediate audio.")
        model_picker(STAGE2_BACKENDS, "freevc", lambda k: _stage2.__setitem__(0, k))

        _section_row(
            "STYLE",
            "How the line should be delivered — tone, emotion, pacing. "
            "Examples: 'cold fury, slow and deliberate', 'breathless and panicked'. Leave blank for neutral.",
        )
        expr_input = (
            ui.textarea(placeholder="e.g. cold fury, slow and deliberate…")
            .classes("w-full").props("outlined dark rows=3")
        )

        _section_row("LINE", "What the character says.")
        text_input = (
            ui.textarea(placeholder="Enter the line…")
            .classes("w-full").props("outlined dark rows=4")
        )

        _section_row("MAX TOKENS", "Parler generation length. Raise if lines are getting cut off.")
        with ui.row().classes("w-full items-center gap-3"):
            tok_lbl = ui.label("500").classes(
                "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
            )
            def _on_tok(e): _max_tokens[0] = int(e.value); tok_lbl.set_text(str(int(e.value)))
            ui.slider(min=50, max=1200, step=50, value=500, on_change=_on_tok).classes("flex-grow").props("color=amber")

        _section_row("PRE-CONVERT LEVEL", "Normalise Parler output before FreeVC. Consistent level reduces distortion.")
        with ui.row().classes("w-full items-center gap-3"):
            norm_lbl = ui.label("-18 dBFS").classes(
                "font-mono text-[10px] text-amber-400 w-16 shrink-0 text-right"
            )
            def _on_norm(e): _normalize_db[0] = float(e.value); norm_lbl.set_text(f"{int(e.value)} dBFS")
            ui.slider(min=-24, max=-3, step=1, value=-18, on_change=_on_norm).classes("flex-grow").props("color=amber")

        ui.space()
        gen_btn = (
            ui.button("GENERATE", on_click=_generate)
            .classes(f"w-full {_BTN}").props("color=amber unelevated")
        )

    def _tick():
        if _progress[0] == 0.0: return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(_progress[0])
    _poll = ui.timer(0.15, _tick, active=False)

    def refresh():
        opts = _all_chars()
        val = char_state[0] if char_state[0] in opts else _NO_CHAR
        char_select.set_options(opts, value=val)
        char_state[0] = val

    def select(name: str):
        opts = _all_chars()
        if name in opts:
            char_select.set_options(opts, value=name)
            _update_info(name)

    return refresh, select


def _build_character(char_state: list[str], _out: dict, _current_char: list[str]):
    """Left controls for CHARACTER sub-tab. Returns refresh."""
    _pitch:         list[float] = [0.0]
    _index_rate:    list[float] = [0.75]
    _filter_radius: list[int]   = [3]

    def _update_info(name: str):
        char_state[0] = name
        _current_char[0] = name
        if name == _NO_CHAR:
            char_info.set_text(""); return
        char = get_character(name)
        has_model = bool(char and char.get("rvc_model"))
        char_info.set_text("model ready" if has_model else "⚠  no model — train from CHARACTERS tab")

    async def _generate():
        if char_state[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning"); return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter a line to speak", type="warning"); return
        char = get_character(char_state[0])
        if not char or "rvc_model" not in char:
            ui.notify("No RVC model — train one from the CHARACTERS tab", type="warning"); return
        gen_btn.disable()
        _out["status"].set_text("—")
        _out["model_loader"].set_visibility(True)
        try:
            from faily.modules.rvc import speak_generate
            _VC_DIR.mkdir(parents=True, exist_ok=True)
            path = await ni_run.io_bound(
                speak_generate, text, char["rvc_model"], _VC_DIR,
                pitch_shift=int(_pitch[0]),
                index_rate=_index_rate[0],
                filter_radius=int(_filter_radius[0]),
                char_name=char_state[0],
            )
            _out["main_player"].set_source(f"/outputs/vc/{path.name}")
            _out["status"].set_text(f"✓  {path.name}")
            _out["add_to_history"](path, text)
        except Exception as exc:
            show_error(exc)
            _out["status"].set_text("error")
        finally:
            _out["model_loader"].set_visibility(False)
            gen_btn.enable()

    with ui.column().classes("gap-3 p-5 overflow-y-auto w-full h-full"):
        _section_row("CHARACTER", "A character with a trained RVC model. Train one from the CHARACTERS tab.")
        char_select = (
            ui.select(options=_rvc_chars(), value=_NO_CHAR, on_change=lambda e: _update_info(e.value))
            .props("outlined dark dense").classes("w-full")
        )
        char_info = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

        _section_row("PITCH SHIFT", "Shift pitch in semitones. 0 = no change.")
        with ui.row().classes("w-full items-center gap-3"):
            pitch_lbl = ui.label("0 st").classes(
                "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
            )
            def _on_pitch(e): _pitch[0] = float(e.value); pitch_lbl.set_text(f"{int(e.value):+d} st")
            ui.slider(min=-12, max=12, step=1, value=0, on_change=_on_pitch).classes("flex-grow").props("color=amber")

        _section_row("INDEX RATE", "How much the trained voice index influences output. Higher = more faithful.")
        with ui.row().classes("w-full items-center gap-3"):
            idx_lbl = ui.label("0.75").classes(
                "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
            )
            def _on_idx(e): _index_rate[0] = float(e.value); idx_lbl.set_text(f"{e.value:.2f}")
            ui.slider(min=0.0, max=1.0, step=0.05, value=0.75, on_change=_on_idx).classes("flex-grow").props("color=amber")

        _section_row("FILTER RADIUS", "Median filter strength on pitch. Higher = smoother. 0 = off.")
        with ui.row().classes("w-full items-center gap-3"):
            filt_lbl = ui.label("3").classes(
                "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
            )
            def _on_filt(e): _filter_radius[0] = int(e.value); filt_lbl.set_text(str(int(e.value)))
            ui.slider(min=0, max=7, step=1, value=3, on_change=_on_filt).classes("flex-grow").props("color=amber")

        _section_row("LINE", "What the character says.")
        text_input = (
            ui.textarea(placeholder="Enter the line…")
            .classes("w-full").props("outlined dark rows=5")
        )

        ui.space()
        gen_btn = (
            ui.button("GENERATE", on_click=_generate)
            .classes(f"w-full {_BTN}").props("color=amber unelevated")
        )

    def refresh():
        opts = _rvc_chars()
        val = char_state[0] if char_state[0] in opts else _NO_CHAR
        char_select.set_options(opts, value=val)
        char_state[0] = val

    return refresh


def _build_oneshot(char_state: list[str], _out: dict, _current_char: list[str]):
    """Left controls for ONE SHOT sub-tab. Returns refresh."""
    _progress: list[float] = [0.0]
    _backend:  list[str]   = ["xtts_v2"]
    _param1:   list[float] = [BACKENDS["xtts_v2"]["param1"]["default"]]
    _param2:   list[float] = [BACKENDS["xtts_v2"]["param2"]["default"]]

    def _update_info(name: str):
        char_state[0] = name
        _current_char[0] = name
        if name == _NO_CHAR:
            char_info.set_text(""); return
        chain = get_ref_chain(name)
        char_info.set_text(
            f"{len(chain)} ref clip{'s' if len(chain) != 1 else ''}" if chain else "⚠  no reference audio"
        )

    def _rebuild_params():
        params_col.clear()
        cfg = BACKENDS[_backend[0]]
        with params_col:
            for p, pval in ((cfg["param1"], _param1), (cfg["param2"], _param2)):
                pval[0] = p["default"]
                with ui.row().classes("items-center gap-1"):
                    section_label(p["label"])
                    ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(p["tooltip"])
                with ui.row().classes("w-full items-center gap-3"):
                    lbl = ui.label(_fmt(p["default"], p["step"])).classes(
                        "font-mono text-[10px] text-amber-400 w-7 shrink-0 text-right"
                    )
                    def _on_slider(e, pv=pval, lb=lbl, st=p["step"]):
                        pv[0] = e.value; lb.set_text(_fmt(e.value, st))
                    ui.slider(
                        min=p["min"], max=p["max"], step=p["step"],
                        value=p["default"], on_change=_on_slider,
                    ).classes("flex-grow").props("color=amber")

    def _on_backend(key: str):
        _backend[0] = key
        ref_text_row.set_visibility(key == "f5_tts")
        style_prompt_row.set_visibility(key == "kokoro")
        _rebuild_params()

    async def _generate():
        if char_state[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning"); return
        text = text_input.value.strip() or "sample text"
        chain = get_ref_chain(char_state[0])
        if not chain and _backend[0] != "kokoro":
            ui.notify("Character has no reference audio", type="warning"); return
        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True
        with build_ref_audio(char_state[0]) as (ref, chain_transcript):
            try:
                path = await ni_run.io_bound(
                    vc_generate,
                    text, ref, _progress, None,
                    _backend[0], _param1[0], _param2[0],
                    ref_text_input.value or chain_transcript,
                    style_prompt_input.value,
                    char_name=char_state[0],
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

    with ui.column().classes("gap-3 p-5 overflow-y-auto w-full h-full"):
        _section_row("CHARACTER", "Pick a character — uses their full reference chain for one-shot cloning.")
        char_select = (
            ui.select(options=_all_chars(), value=_NO_CHAR, on_change=lambda e: _update_info(e.value))
            .props("outlined dark dense").classes("w-full")
        )
        char_info = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

        _section_row("BACKEND", "Voice cloning engine — same options as the CLONE tab.")
        model_picker(BACKENDS, "xtts_v2", _on_backend)

        params_col = ui.column().classes("w-full gap-4")

        with ui.column().classes("w-full gap-2") as ref_text_row:
            _section_row(
                "REFERENCE TRANSCRIPT",
                "What the reference says. Leave blank to use stored transcripts from the character.",
            )
            ref_text_input = (
                ui.input(placeholder="leave blank to use stored transcripts…")
                .classes("w-full").props("outlined dark")
            )
        ref_text_row.set_visibility(False)

        with ui.column().classes("w-full gap-2") as style_prompt_row:
            _section_row("VOICE NAME", "Kokoro voice. Examples: af_heart, am_adam, af_bella, af_sky.")
            style_prompt_input = (
                ui.input(placeholder="e.g. af_heart, am_adam…")
                .classes("w-full").props("outlined dark")
            )
        style_prompt_row.set_visibility(False)

        _section_row("TEXT", "What the character says.")
        text_input = (
            ui.textarea(placeholder="Enter the line…")
            .classes("w-full").props("outlined dark rows=5")
        )

        ui.space()
        gen_btn = (
            ui.button("GENERATE", on_click=_generate)
            .classes(f"w-full {_BTN}").props("color=amber unelevated")
        )

    def _tick():
        if _progress[0] == 0.0: return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(_progress[0])
    _poll = ui.timer(0.15, _tick, active=False)

    _rebuild_params()

    def refresh():
        opts = _all_chars()
        val = char_state[0] if char_state[0] in opts else _NO_CHAR
        char_select.set_options(opts, value=val)
        char_state[0] = val

    return refresh


def build_speak_tab():
    """Returns (refresh_all, select_character)."""
    _exp_state:  list[str] = [_NO_CHAR]
    _char_state: list[str] = [_NO_CHAR]
    _shot_state: list[str] = [_NO_CHAR]
    _current_char: list[str] = [_NO_CHAR]

    _out: dict = {}

    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        # ── LEFT: sub-tabs + swappable controls ──────────────────────────────
        with ui.column().classes("w-full h-full gap-0 border-r border-[#252525]"):
            with ui.tabs().classes("w-full").props("dense").style(
                "background:#0a0a0a; border-bottom:1px solid #1a1a1a; min-height:36px;"
            ) as inner_tabs:
                exp_tab  = ui.tab("EXPRESSION",  icon="psychology")
                char_tab = ui.tab("CHARACTER",   icon="model_training")
                shot_tab = ui.tab("ONE SHOT",    icon="flash_on")

            with ui.tab_panels(inner_tabs, value=exp_tab).classes("w-full flex-grow"):
                with ui.tab_panel(exp_tab):
                    r_exp, s_exp = _build_expression(_exp_state, _out, _current_char)
                with ui.tab_panel(char_tab):
                    r_char = _build_character(_char_state, _out, _current_char)
                with ui.tab_panel(shot_tab):
                    r_shot = _build_oneshot(_shot_state, _out, _current_char)

        # ── RIGHT: shared output panel ────────────────────────────────────────
        pb, ml, mp, st, _, _, ath = output_panel(
            "vc",
            get_char_name=lambda: _current_char[0] if _current_char[0] != _NO_CHAR else None,
        )
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    def refresh_all():
        r_exp()
        r_char()
        r_shot()

    def select_character(name: str):
        s_exp(name)
        inner_tabs.set_value("EXPRESSION")

    return refresh_all, select_character
