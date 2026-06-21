import os
from pathlib import Path
from nicegui import app, ui
from faily.ui.tabs.tts_tab import build_tts_tab
from faily.ui.tabs.foley_tab import build_foley_tab
from faily.ui.tabs.vc_tab import build_vc_tab

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
    for d in [OUTPUTS, OUTPUTS / "tts", OUTPUTS / "sfx", OUTPUTS / "vc", OUTPUTS / "vc" / "refs"]:
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
            vc_tab    = ui.tab("CLONE", icon="mic")
            tts_tab   = ui.tab("TTS",   icon="record_voice_over")
            foley_tab = ui.tab("FOLEY", icon="graphic_eq")

        with ui.tab_panels(tabs, value=tts_tab).classes("w-full flex-grow"):
            with ui.tab_panel(tts_tab):
                build_tts_tab()
            with ui.tab_panel(vc_tab):
                build_vc_tab()
            with ui.tab_panel(foley_tab):
                build_foley_tab()

    # native=True opens a pywebview desktop window.
    # Requires GTK (python3-gi) or Qt (PySide6 + system libs) on the host.
    # Set FAILY_NATIVE=1 to enable; falls back to web mode automatically.
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
