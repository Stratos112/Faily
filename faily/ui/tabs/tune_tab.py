from nicegui import ui, run as ni_run
from faily.modules.vc import tune_generate, EXPRESSION_ENGINES
from faily.core.characters import list_characters, get_character, get_ref_path
from faily.ui.components import output_panel, section_label, show_error

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
    _out: dict = {}

    def _update_char_info(name: str):
        _char_name[0] = name
        if name == _NO_CHAR:
            char_info.set_text("")
            return
        char = get_character(name)
        ref = get_ref_path(name)
        ancestry = f"↳ {char['parent']}" if char and "parent" in char else "base character"
        ref_label = ref.name if (ref and ref.exists()) else "⚠  no reference audio — save from CLONE tab"
        char_info.set_text(f"{ancestry}  ·  {ref_label}")

    def _on_char(e):
        _update_char_info(e.value)

    def _on_engine(e):
        _engine[0] = e.value
        engine_desc.set_text(EXPRESSION_ENGINES[e.value]["desc"])

    async def _generate():
        if _char_name[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning")
            return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter a line to speak", type="warning")
            return
        ref = get_ref_path(_char_name[0])
        if ref is None or not ref.exists():
            ui.notify("Character has no reference audio — save it from the CLONE tab first", type="warning")
            return

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        try:
            path = await ni_run.io_bound(
                tune_generate,
                text,
                expression_input.value.strip(),
                _engine[0],
                ref,
                _progress,
            )
            _out["main_player"].set_source(f"/outputs/vc/{path.name}")
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
                "Model that interprets your style description and generates the expressive intermediate audio. "
                "FreeVC then converts that audio to the character's voice.",
            )
            engine_desc = ui.label(EXPRESSION_ENGINES[_DEFAULT_ENGINE]["desc"]).classes(
                "text-[#444] font-mono text-[10px] tracking-wide"
            )
            ui.select(
                options={k: v["label"] for k, v in EXPRESSION_ENGINES.items()},
                value=_DEFAULT_ENGINE,
                on_change=_on_engine,
            ).props("outlined dark dense").classes("w-full")

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

            ui.space()
            gen_btn = (
                ui.button("GENERATE", on_click=_generate)
                .classes(f"w-full {_BTN}")
                .props("color=amber unelevated")
            )

        pb, ml, mp, st, _, _, ath = output_panel("vc")
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
