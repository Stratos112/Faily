from nicegui import ui, run as ni_run
from pathlib import Path
from faily.core.characters import list_characters, get_character
from faily.ui.components import output_panel, section_label, show_error

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"
SPEAK_OUTPUT_DIR = Path("outputs/speak")


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _fmt(val: float, decimals: int = 2) -> str:
    return str(int(val)) if decimals == 0 else f"{val:.{decimals}f}"


def _rvc_chars() -> dict[str, str]:
    opts = {_NO_CHAR: _NO_CHAR}
    for c in list_characters():
        if "rvc_model" in c:
            opts[c["name"]] = c["name"]
    return opts


def build_speak_tab():
    _char_name: list[str] = [_NO_CHAR]
    _pitch: list[float] = [0.0]
    _index_rate: list[float] = [0.75]
    _filter_radius: list[int] = [3]
    _out: dict = {}

    def _update_char_info(name: str):
        _char_name[0] = name
        if name == _NO_CHAR:
            char_info.set_text("")
            return
        char = get_character(name)
        model_path = char.get("rvc_model", "") if char else ""
        char_info.set_text("model ready" if model_path else "⚠  no model — build one from TUNE tab")

    def _on_char(e):
        _update_char_info(e.value)

    async def _generate():
        if _char_name[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning")
            return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter a line to speak", type="warning")
            return
        char = get_character(_char_name[0])
        if not char or "rvc_model" not in char:
            ui.notify("No RVC model for this character — build one from the TUNE tab first", type="warning")
            return

        gen_btn.disable()
        _out["status"].set_text("—")
        _out["model_loader"].set_visibility(True)

        try:
            from faily.modules.rvc import speak_generate
            SPEAK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = await ni_run.io_bound(
                speak_generate,
                text,
                char["rvc_model"],
                SPEAK_OUTPUT_DIR,
                pitch_shift=int(_pitch[0]),
                index_rate=_index_rate[0],
                filter_radius=int(_filter_radius[0]),
                char_name=_char_name[0],
            )
            _out["main_player"].set_source(f"/outputs/speak/{path.name}")
            _out["status"].set_text(f"✓  {path.name}")
            _out["add_to_history"](path, text)
        except Exception as exc:
            show_error(exc)
            _out["status"].set_text("error")
        finally:
            _out["model_loader"].set_visibility(False)
            gen_btn.enable()

    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            _section_row("CHARACTER", "A character with a trained RVC model. Build one from the TUNE tab.")
            char_select = (
                ui.select(options=_rvc_chars(), value=_NO_CHAR, on_change=_on_char)
                .props("outlined dark dense")
                .classes("w-full")
            )
            char_info = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

            _section_row(
                "PITCH SHIFT",
                "Shift pitch in semitones. 0 = no change. Useful for gender-adjusting the output.",
            )
            with ui.row().classes("w-full items-center gap-3"):
                pitch_lbl = ui.label("0 st").classes(
                    "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_pitch(e):
                    _pitch[0] = float(e.value)
                    pitch_lbl.set_text(f"{int(e.value):+d} st")
                ui.slider(min=-12, max=12, step=1, value=0, on_change=_on_pitch).classes(
                    "flex-grow"
                ).props("color=amber")

            _section_row(
                "INDEX RATE",
                "How much the trained voice index influences the output. "
                "Higher = more faithful to character, lower = slightly more natural.",
            )
            with ui.row().classes("w-full items-center gap-3"):
                index_lbl = ui.label("0.75").classes(
                    "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_index(e):
                    _index_rate[0] = float(e.value)
                    index_lbl.set_text(f"{e.value:.2f}")
                ui.slider(min=0.0, max=1.0, step=0.05, value=0.75, on_change=_on_index).classes(
                    "flex-grow"
                ).props("color=amber")

            _section_row(
                "FILTER RADIUS",
                "Median filter strength applied to pitch. "
                "Higher = smoother/less breathy output. 0 = off.",
            )
            with ui.row().classes("w-full items-center gap-3"):
                filter_lbl = ui.label("3").classes(
                    "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                )
                def _on_filter(e):
                    _filter_radius[0] = int(e.value)
                    filter_lbl.set_text(str(int(e.value)))
                ui.slider(min=0, max=7, step=1, value=3, on_change=_on_filter).classes(
                    "flex-grow"
                ).props("color=amber")

            _section_row("LINE", "What the character says.")
            text_input = (
                ui.textarea(placeholder="Enter the line…")
                .classes("w-full")
                .props("outlined dark rows=5")
            )

            ui.space()
            gen_btn = (
                ui.button("GENERATE", on_click=_generate)
                .classes(f"w-full {_BTN}")
                .props("color=amber unelevated")
            )

        pb, ml, mp, st, _, _, ath = output_panel(
            "speak",
            get_char_name=lambda: _char_name[0] if _char_name[0] != _NO_CHAR else None,
        )
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    def refresh_characters():
        opts = _rvc_chars()
        value = _char_name[0] if _char_name[0] in opts else _NO_CHAR
        char_select.set_options(opts, value=value)
        _char_name[0] = value

    return refresh_characters
