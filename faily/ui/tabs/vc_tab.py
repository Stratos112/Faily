from nicegui import ui, run as ni_run
from faily.modules.vc import generate as vc_generate, VC_OUTPUT_DIR
from faily.ui.components import output_panel, section_label
from pathlib import Path

_BTN = "font-mono tracking-widest"


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def build_vc_tab():
    _progress: list[float] = [0.0]
    _ref_path: list[Path | None] = [None]
    _out: dict = {}

    async def _on_upload(e):
        ref_dir = VC_OUTPUT_DIR / "refs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        dest = ref_dir / e.name
        dest.write_bytes(e.content.read())
        _ref_path[0] = dest
        ref_player.set_source(f"/outputs/vc/refs/{e.name}")
        ref_player.set_visibility(True)
        ref_status.set_text(f"✓  {e.name}")

    async def _generate():
        text = text_input.value.strip()
        if not text:
            ui.notify("Enter text to synthesize", type="warning")
            return
        if _ref_path[0] is None:
            ui.notify("Upload a reference voice clip first", type="warning")
            return

        gen_btn.disable()
        _out["status"].set_text("—")
        _progress[0] = 0.0
        _out["model_loader"].set_visibility(True)
        _poll.active = True

        try:
            path = await ni_run.io_bound(
                vc_generate,
                text,
                _ref_path[0],
                _progress,
            )
            _out["main_player"].set_source(f"/outputs/vc/{path.name}")
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

            _section_row(
                "MODEL",
                "SpeechT5 (Microsoft) conditioned on speaker x-vector embeddings extracted from your reference clip. "
                "The vocoder is SpeechT5 HiFi-GAN.",
            )
            ui.label("SpeechT5 · X-Vector Encoder · HiFi-GAN").classes(
                "text-[#444] font-mono text-[10px] tracking-wide"
            )

            _section_row(
                "REFERENCE VOICE",
                "Upload 5–30 seconds of clean speech from the target voice. "
                "Avoid background music or noise — the cleaner the clip, the more accurate the speaker embedding.",
            )
            ref_status = ui.label("no file loaded").classes("text-[#444] font-mono text-[10px]")
            (
                ui.upload(on_upload=_on_upload, max_files=1, auto_upload=True)
                .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Upload audio'")
                .classes("w-full")
            )
            ref_player = ui.audio("").classes("w-full rounded mt-1")
            ref_player.set_visibility(False)

            _section_row(
                "TEXT",
                "Text to synthesize in the cloned voice. "
                "SpeechT5 handles short-to-medium sentences best — very long inputs may be truncated.",
            )
            text_input = (
                ui.textarea(placeholder="Enter text to synthesize in the cloned voice…")
                .classes("w-full")
                .props("outlined dark rows=7")
            )

            ui.space()
            gen_btn = (
                ui.button("CLONE + SYNTHESIZE", on_click=_generate)
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
