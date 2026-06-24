from nicegui import ui, run as ni_run
from faily.modules.vc import generate as vc_generate, BACKENDS
from faily.core.characters import (
    list_characters, get_character, get_ref_path,
    save_character, save_sub_character,
)
from faily.ui.components import output_panel, section_label, show_error
from pathlib import Path

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select character —"


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _fmt(val: float, step: float) -> str:
    return str(int(val)) if step >= 1 else f"{val:.2f}"


def _char_options() -> dict[str, str]:
    chars = list_characters()
    opts = {_NO_CHAR: _NO_CHAR}
    for c in chars:
        label = f"  ↳ {c['name']}" if "parent" in c else c["name"]
        opts[c["name"]] = label
    return opts


def build_tune_tab():
    _progress: list[float] = [0.0]
    _char_name: list[str] = [_NO_CHAR]
    _backend: list[str] = ["chatterbox"]
    _param1: list[float] = [BACKENDS["chatterbox"]["param1"]["default"]]
    _param2: list[float] = [BACKENDS["chatterbox"]["param2"]["default"]]
    _speed: list[float] = [1.0]
    _out: dict = {}

    def _rebuild_params():
        params_col.clear()
        cfg = BACKENDS[_backend[0]]
        with params_col:
            for p, pval in ((cfg["param1"], _param1), (cfg["param2"], _param2)):
                pval[0] = p["default"]
                _section_row(p["label"], p["tooltip"])
                with ui.row().classes("w-full items-center gap-3"):
                    lbl = ui.label(_fmt(p["default"], p["step"])).classes(
                        "font-mono text-[10px] text-amber-400 w-7 shrink-0 text-right"
                    )
                    def _on_change(e, pval=pval, lbl=lbl, step=p["step"]):
                        pval[0] = e.value
                        lbl.set_text(_fmt(e.value, step))
                    ui.slider(
                        min=p["min"], max=p["max"], step=p["step"],
                        value=p["default"], on_change=_on_change,
                    ).classes("flex-grow").props("color=amber")

    def _on_char(e):
        _char_name[0] = e.value
        if e.value == _NO_CHAR:
            char_desc.set_text("")
            return
        char = get_character(e.value)
        if not char:
            return
        if "parent" in char:
            parent = char["parent"]
            backend = char.get("backend", "chatterbox")
            char_desc.set_text(f"sub-character of  {parent}  ·  {BACKENDS.get(backend, {}).get('label', backend)}")
            _backend[0] = backend
            backend_select.set_value(backend)
            _param1[0] = char.get("param1", BACKENDS[backend]["param1"]["default"])
            _param2[0] = char.get("param2", BACKENDS[backend]["param2"]["default"])
            _speed[0] = char.get("speed", 1.0)
            style_input.set_value(char.get("style_prompt", ""))
            _rebuild_params()
        else:
            char_desc.set_text(f"base character  ·  ref: {char.get('ref_audio', '?')}")

    def _on_backend(e):
        _backend[0] = e.value
        backend_desc.set_text(BACKENDS[e.value]["desc"])
        _rebuild_params()

    async def _generate():
        if _char_name[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning")
            return
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter text to speak", type="warning")
            return
        ref = get_ref_path(_char_name[0])
        if ref is None or not ref.exists():
            ui.notify("Character has no reference audio — save it from the CLONE tab first", type="warning")
            return

        char = get_character(_char_name[0])
        ref_text = char.get("transcript", "") if char else ""

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        try:
            path = await ni_run.io_bound(
                vc_generate, text, ref, _progress, None,
                _backend[0], _param1[0], _param2[0], ref_text,
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

    def _save():
        if _char_name[0] == _NO_CHAR:
            ui.notify("Select a character first", type="warning")
            return
        raw = save_name_input.value.strip()
        if not raw:
            ui.notify("Enter a name to save", type="warning")
            return
        parent = _char_name[0]
        char = get_character(parent)
        # If saving over the base character's own name: update config keeping ref audio
        if raw == parent and char and "ref_audio" in char:
            ref = get_ref_path(parent)
            transcript = char.get("transcript", "")
            save_character(parent, ref, transcript)
            ui.notify(f"updated  {parent}", type="positive", timeout=2000)
        else:
            # Resolve true parent (in case loaded char is itself a sub-character)
            true_parent = char.get("parent", parent) if char else parent
            save_sub_character(
                name=raw,
                parent=true_parent,
                backend=_backend[0],
                param1=_param1[0],
                param2=_param2[0],
                speed=_speed[0],
                style_prompt=style_input.value.strip(),
            )
            ui.notify(f"saved  {raw}", type="positive", timeout=2000)
        # Refresh character picker
        char_select.set_options(_char_options(), value=raw)
        _char_name[0] = raw

    # ── UI ────────────────────────────────────────────────────────────────
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            _section_row(
                "CHARACTER",
                "Select a character created in the CLONE tab. Sub-characters load their saved expression settings.",
            )
            char_select = (
                ui.select(options=_char_options(), value=_NO_CHAR, on_change=_on_char)
                .props("outlined dark dense")
                .classes("w-full")
            )
            char_desc = ui.label("").classes("text-[#444] font-mono text-[10px] tracking-wide")

            _section_row(
                "BACKEND",
                "Which voice cloning engine to use for this expression. Chatterbox has the best emotion controls.",
            )
            backend_desc = ui.label(BACKENDS["chatterbox"]["desc"]).classes(
                "text-[#444] font-mono text-[10px] tracking-wide"
            )
            backend_select = (
                ui.select(
                    options={k: v["label"] for k, v in BACKENDS.items()},
                    value="chatterbox",
                    on_change=_on_backend,
                )
                .props("outlined dark dense")
                .classes("w-full")
            )

            _section_row(
                "STYLE PROMPT",
                "Describe how the voice should feel — pace, tone, emotion. "
                "Saved with sub-characters for documentation; will drive Parler/StyleTTS2 in future.",
            )
            style_input = (
                ui.input(placeholder="e.g. speaks with cold fury, slow and deliberate…")
                .classes("w-full")
                .props("outlined dark")
            )

            params_col = ui.column().classes("w-full gap-4")

            _section_row("SPEED", "Speech rate. 1.0 is natural pace.")
            with ui.row().classes("w-full items-center gap-3"):
                speed_lbl = ui.label("1.00").classes(
                    "font-mono text-[10px] text-amber-400 w-7 shrink-0 text-right"
                )
                def _on_speed(e):
                    _speed[0] = e.value
                    speed_lbl.set_text(f"{e.value:.2f}")
                ui.slider(min=0.05, max=2.0, step=0.05, value=1.0, on_change=_on_speed).classes(
                    "flex-grow"
                ).props("color=amber")

            _section_row("TEXT", "What the character should say. Supports longer passages.")
            text_input = (
                ui.textarea(placeholder="Enter text…")
                .classes("w-full")
                .props("outlined dark rows=4")
            )

            ui.separator().classes("opacity-10")

            _section_row(
                "SAVE AS",
                "Save the current expression settings as a character. "
                "Use the same name to overwrite, or a new name like 'DarthVader-angry' for a sub-character.",
            )
            with ui.row().classes("w-full gap-2 items-center"):
                save_name_input = (
                    ui.input(placeholder="character name…")
                    .props("outlined dark dense")
                    .classes("flex-grow")
                )
                ui.button(icon="save", on_click=_save).props("flat dense color=amber").tooltip("Save character")

            ui.space()
            gen_btn = (
                ui.button("GENERATE", on_click=_generate)
                .classes(f"w-full {_BTN}")
                .props("color=amber unelevated")
            )

        pb, ml, mp, st, _, _, ath = output_panel("vc")
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    _rebuild_params()

    def _tick():
        val = _progress[0]
        if val == 0.0:
            return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(val)

    _poll = ui.timer(0.15, _tick, active=False)

    def refresh_characters():
        current = _char_name[0]
        opts = _char_options()
        value = current if current in opts else _NO_CHAR
        char_select.set_options(opts, value=value)
        _char_name[0] = value

    return refresh_characters
