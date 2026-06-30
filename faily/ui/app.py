import os
from pathlib import Path
from nicegui import app, ui
from faily.ui.tabs.tts_tab import build_tts_tab
from faily.ui.tabs.foley_tab import build_foley_tab
from faily.ui.tabs.vc_tab import build_vc_tab
from faily.ui.tabs.tune_tab import build_tune_tab
from faily.ui.tabs.characters_tab import build_characters_tab

OUTPUTS = Path("outputs")

_GLOBAL_CSS = """
    body, .q-page { background: #0f0f0f !important; }
    .q-tab__label { font-family: monospace; letter-spacing: 0.15em; font-size: 13px; }
    .q-tab-panels { background: #0f0f0f !important; }
    .q-tabs { background: #0f0f0f !important; border-bottom: 1px solid #252525; }
    .q-tab--active { color: #f59e0b !important; }
    .q-tab__indicator { background: #f59e0b !important; }
    audio { filter: invert(0.9) hue-rotate(180deg); }
    @keyframes faily-wave {
        0%, 100% { transform: scaleY(0.12); }
        50%       { transform: scaleY(1.0);  }
    }
    .faily-loader {
        display: inline-flex;
        align-items: flex-end;
        gap: 3px;
        height: 28px;
    }
    .faily-loader span {
        display: block;
        width: 4px;
        height: 28px;
        background: #f59e0b;
        border-radius: 2px;
        animation: faily-wave 1.1s ease-in-out infinite;
        transform-origin: bottom;
    }
    .faily-loader span:nth-child(1) { animation-delay: 0.00s; }
    .faily-loader span:nth-child(2) { animation-delay: 0.16s; }
    .faily-loader span:nth-child(3) { animation-delay: 0.32s; }
    .faily-loader span:nth-child(4) { animation-delay: 0.48s; }
    .faily-loader span:nth-child(5) { animation-delay: 0.64s; }
    .faily-loader span:nth-child(6) { animation-delay: 0.48s; }
    .faily-loader span:nth-child(7) { animation-delay: 0.32s; }
"""


def run():
    for d in [OUTPUTS, OUTPUTS / "tts", OUTPUTS / "sfx", OUTPUTS / "vc", OUTPUTS / "vc" / "refs", OUTPUTS / "characters"]:
        d.mkdir(parents=True, exist_ok=True)
    app.add_static_files("/outputs", str(OUTPUTS))

    @ui.page("/")
    def index():
        ui.dark_mode().enable()
        ui.add_css(_GLOBAL_CSS)

        with ui.header().classes(
            "bg-[#0f0f0f] border-b border-[#252525] flex items-center px-8 py-4"
        ).style("min-height:56px"):
            ui.label("F A I L Y").classes(
                "text-white font-mono text-lg tracking-[0.4em] font-bold"
            )
            ui.space()
            from faily.core.model_manager import manager
            ui.label(manager.device.upper()).classes(
                "text-[#555] font-mono text-[10px] tracking-widest border border-[#333] px-2 py-1 rounded"
            )

        with ui.tabs().classes("w-full") as tabs:
            vc_tab       = ui.tab("CLONE",      icon="mic")
            chars_tab    = ui.tab("CHARACTERS", icon="manage_accounts")
            speak_tab    = ui.tab("SPEAK",      icon="record_voice_over")
            tts_tab      = ui.tab("TTS",        icon="text_fields")
            foley_tab    = ui.tab("FOLEY",      icon="graphic_eq")

        # deferred callbacks — populated after panels are built
        _speak_refresh: list = [lambda: None]
        _speak_select:  list = [lambda name: None]
        _chars_refresh: list = [lambda: None]
        _vc_refresh:    list = [lambda: None]

        def _on_tab_change(e):
            if e.value == "SPEAK":
                _speak_refresh[0]()
            elif e.value == "CHARACTERS":
                _chars_refresh[0]()

        def _on_speak(name: str):
            """Called from CHARACTERS tab — navigate to SPEAK tab with char pre-selected."""
            _speak_refresh[0]()
            tabs.set_value("SPEAK")
            _speak_select[0](name)

        def _on_char_change():
            """Called when a character is deleted from CHARACTERS tab."""
            _speak_refresh[0]()
            _vc_refresh[0]()

        with ui.tab_panels(tabs, value=vc_tab, on_change=_on_tab_change).classes("w-full flex-grow"):
            with ui.tab_panel(vc_tab):
                _vc_refresh[0] = build_vc_tab()

            with ui.tab_panel(chars_tab):
                _chars_refresh[0] = build_characters_tab(
                    on_speak=_on_speak,
                    on_change=_on_char_change,
                )

            with ui.tab_panel(speak_tab):
                _speak_refresh[0], _speak_select[0] = build_tune_tab()

            with ui.tab_panel(tts_tab):
                build_tts_tab()

            with ui.tab_panel(foley_tab):
                build_foley_tab()

    _native = os.environ.get("FAILY_NATIVE", "0") == "1"
    ui.run(
        title="Faily",
        native=_native,
        window_size=(1280, 780) if _native else None,
        reload=False,
        dark=True,
        host="127.0.0.1" if _native else "0.0.0.0",
        port=7842,
    )
