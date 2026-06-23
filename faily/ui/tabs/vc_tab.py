from nicegui import ui, run as ni_run
from faily.modules.vc import generate as vc_generate, VC_OUTPUT_DIR, BACKENDS, transcribe_ref
from faily.ui.components import output_panel, section_label, show_error
from pathlib import Path

_REFS_DIR = VC_OUTPUT_DIR / "refs"
_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_BTN = "font-mono tracking-widest"


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _fmt(val: float, step: float) -> str:
    return str(int(val)) if step >= 1 else f"{val:.2f}"


def build_vc_tab():
    _progress: list[float] = [0.0]
    _ref_path: list[Path | None] = [None]
    _backend: list[str] = ["xtts_v2"]
    _param1: list[float] = [BACKENDS["xtts_v2"]["param1"]["default"]]
    _param2: list[float] = [BACKENDS["xtts_v2"]["param2"]["default"]]
    _out: dict = {}

    def _scan_refs() -> list[Path]:
        if not _REFS_DIR.exists():
            return []
        return sorted(f for f in _REFS_DIR.iterdir() if f.suffix.lower() in _EXTS)

    async def _autofill_transcript():
        if _backend[0] != "f5_tts" or _ref_path[0] is None:
            return
        ref_text_input.set_value("transcribing…")
        try:
            text = await ni_run.io_bound(transcribe_ref, _ref_path[0])
            ref_text_input.set_value(text)
        except Exception:
            ref_text_input.set_value("")

    async def _on_select(path: Path):
        _ref_path[0] = path
        name_input.set_value(path.stem)
        _rebuild_list()
        await _autofill_transcript()

    def _rebuild_list():
        ref_list.clear()
        with ref_list:
            files = _scan_refs()
            if not files:
                ui.label("no samples yet — upload one above").classes(
                    "text-[#333] font-mono text-[10px] px-1 py-2"
                )
                return
            for p in files:
                active = _ref_path[0] == p
                cls = (
                    "border-amber-500/70 text-amber-400"
                    if active else
                    "border-[#1e1e1e] text-[#777] hover:border-[#333]"
                )
                with ui.row().classes(
                    f"w-full items-center gap-2 px-3 py-1 rounded cursor-pointer border {cls}"
                ) as row:
                    ui.icon("mic", size="13px").classes("shrink-0")
                    ui.label(p.name).classes("font-mono text-[10px] truncate flex-grow")
                    row.on("click", lambda p=p: _on_select(p))

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

    async def _on_upload(e):
        _REFS_DIR.mkdir(parents=True, exist_ok=True)
        dest = _REFS_DIR / e.file.name
        dest.write_bytes(await e.file.read())
        _ref_path[0] = dest
        name_input.set_value(dest.stem)
        _rebuild_list()
        await _autofill_transcript()

    def _rename():
        if _ref_path[0] is None or not _ref_path[0].exists():
            ui.notify("Select a sample first", type="warning")
            return
        new_stem = name_input.value.strip()
        if not new_stem:
            return
        new_path = _REFS_DIR / (new_stem + _ref_path[0].suffix)
        _ref_path[0].rename(new_path)
        _ref_path[0] = new_path
        _rebuild_list()
        ui.notify(f"saved as  {new_path.name}", type="positive", timeout=2000)

    async def _generate():
        if _ref_path[0] is None or not _ref_path[0].exists():
            ui.notify("Select a reference sample first", type="warning")
            return
        text = text_input.value.strip() or "sample text"

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        try:
            path = await ni_run.io_bound(
                vc_generate, text, _ref_path[0], _progress, None,
                _backend[0], _param1[0], _param2[0], ref_text_input.value,
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

    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):

        with ui.column().classes("gap-4 p-8 border-r border-[#252525] overflow-y-auto"):

            _section_row(
                "BACKEND",
                "Voice cloning engine. Each uses a different approach — see description below the selector.",
            )
            desc_label = ui.label(BACKENDS["xtts_v2"]["desc"]).classes(
                "text-[#444] font-mono text-[10px] tracking-wide"
            )

            async def _on_backend(e):
                _backend[0] = e.value
                desc_label.set_text(BACKENDS[e.value]["desc"])
                ref_text_row.set_visibility(e.value == "f5_tts")
                _rebuild_params()
                await _autofill_transcript()

            ui.select(
                options={k: v["label"] for k, v in BACKENDS.items()},
                value="xtts_v2",
                on_change=_on_backend,
            ).props("outlined dark dense").classes("w-full")

            _section_row(
                "SAMPLES",
                "Upload 5–30 s of clean speech from the target voice. "
                "Select a clip below to use it. Multiple samples can be stored and switched between.",
            )
            (
                ui.upload(on_upload=_on_upload, multiple=True, auto_upload=True)
                .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Upload'")
                .classes("w-full")
            )
            ref_list = ui.column().classes("w-full gap-1 mt-1")

            _section_row(
                "NAME",
                "Rename the selected sample. The name shows up in the TTS CLONE VOICE picker — "
                "name it something memorable before switching voices.",
            )
            with ui.row().classes("w-full gap-2 items-center"):
                name_input = (
                    ui.input(placeholder="clone name…")
                    .props("outlined dark dense")
                    .classes("flex-grow")
                )
                ui.button(icon="check", on_click=_rename).props("flat dense color=amber").tooltip("Save name")

            _section_row("TEXT", "Short phrase to audition the cloned voice.")
            text_input = (
                ui.input(value="sample text")
                .classes("w-full")
                .props("outlined dark")
            )

            params_col = ui.column().classes("w-full gap-4")

            with ui.column().classes("w-full gap-2") as ref_text_row:
                _section_row(
                    "REFERENCE TRANSCRIPT",
                    "What is being said in the reference clip. Optional — leave blank to auto-transcribe. "
                    "Providing it manually gives better quality.",
                )
                ref_text_input = (
                    ui.input(placeholder="type what the reference clip says…")
                    .classes("w-full")
                    .props("outlined dark")
                )
            ref_text_row.set_visibility(False)

            ui.space()
            gen_btn = (
                ui.button("CLONE SAMPLE", on_click=_generate)
                .classes(f"w-full {_BTN}")
                .props("color=amber unelevated")
            )

        pb, ml, mp, st, _, _, ath = output_panel("vc")
        _out.update(progress_bar=pb, model_loader=ml, main_player=mp, status=st, add_to_history=ath)

    _rebuild_list()
    _rebuild_params()

    def _tick():
        val = _progress[0]
        if val == 0.0:
            return
        _out["model_loader"].set_visibility(False)
        _out["progress_bar"].set_visibility(True)
        _out["progress_bar"].set_value(val)

    _poll = ui.timer(0.15, _tick, active=False)
