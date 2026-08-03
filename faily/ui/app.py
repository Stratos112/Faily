import os
from pathlib import Path
from nicegui import app, ui
from faily.ui.tabs.foley_tab import build_foley_tab
from faily.ui.tabs.vc_tab import build_vc_tab
from faily.ui.tabs.speak_tab import build_speak_tab
from faily.ui.tabs.edit_tab import build_edit_tab
from faily.ui.tabs.daw_tab import build_daw_tab
from faily.ui.tabs.characters_tab import build_characters_tab
from faily.ui.components import set_edit_callback, set_daw_callback

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

    .pipeline-tab { border-top: 2px solid rgba(245, 158, 11, 0.28) !important; }
    .pipeline-first { border-left: 2px solid rgba(245, 158, 11, 0.28) !important; border-top-left-radius: 4px !important; }
    .pipeline-last  { border-right: 2px solid rgba(245, 158, 11, 0.28) !important; border-top-right-radius: 4px !important; }
"""


def run():
    for d in [OUTPUTS, OUTPUTS / "tts", OUTPUTS / "sfx", OUTPUTS / "vc", OUTPUTS / "vc" / "refs", OUTPUTS / "characters", OUTPUTS / "speak", OUTPUTS / "edit"]:
        d.mkdir(parents=True, exist_ok=True)
    app.add_static_files("/outputs", str(OUTPUTS))

    @ui.page("/")
    def index():
        ui.dark_mode().enable()
        ui.add_css(_GLOBAL_CSS)

        def _open_settings():
            from faily.core.settings import load_settings, save_settings
            cfg = load_settings()
            with ui.dialog() as dlg, ui.card().classes(
                "bg-[#1a1a1a] border border-[#333] min-w-[460px] gap-3"
            ):
                ui.label("SETTINGS").classes("text-white font-mono text-xs tracking-widest")
                ui.separator().classes("opacity-20")
                ui.label("DOWNLOAD LOCATION").classes(
                    "text-[#444] font-mono text-[10px] tracking-widest"
                )
                path_inp = (
                    ui.input(value=cfg.get("download_dir", str(Path.home() / "Downloads")))
                    .props("outlined dark dense")
                    .classes("w-full font-mono text-[11px]")
                )
                ui.label(
                    "All download buttons save to this folder."
                ).classes("text-[#333] font-mono text-[10px]")

                def _save_cfg():
                    new_dir = path_inp.value.strip() or str(Path.home() / "Downloads")
                    save_settings({**cfg, "download_dir": new_dir})
                    dlg.close()
                    ui.notify("Settings saved", type="positive", timeout=2000)

                with ui.row().classes("w-full justify-end gap-2 mt-1"):
                    ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                    ui.button("Save", on_click=_save_cfg).props("color=amber unelevated dense")
            dlg.open()

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
            ui.button(icon="settings", on_click=_open_settings).props(
                "flat dense color=grey"
            ).tooltip("Settings")

        with ui.tabs().classes("w-full") as tabs:
            chars_tab = ui.tab("CHARACTERS", icon="manage_accounts").classes("pipeline-tab pipeline-first")
            vc_tab    = ui.tab("CLONE",      icon="mic").classes("pipeline-tab pipeline-last")
            speak_tab = ui.tab("SPEAK",      icon="spatial_audio")
            edit_tab  = ui.tab("EDIT",       icon="tune")
            daw_tab   = ui.tab("DAW",        icon="piano")
            foley_tab = ui.tab("FOLEY",      icon="graphic_eq")

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
            """Called from CHARACTERS tab — navigate to SPEAK/EXPRESSION with char pre-selected."""
            _speak_refresh[0]()
            tabs.set_value("SPEAK")
            _speak_select[0](name)

        def _on_char_change():
            """Called when characters are added/deleted."""
            _speak_refresh[0]()
            _vc_refresh[0]()

        with ui.tab_panels(tabs, value=speak_tab, on_change=_on_tab_change).classes("w-full flex-grow"):
            with ui.tab_panel(vc_tab):
                _vc_refresh[0] = build_vc_tab()

            with ui.tab_panel(chars_tab):
                _chars_refresh[0] = build_characters_tab(
                    on_speak=_on_speak,
                    on_change=_on_char_change,
                )

            with ui.tab_panel(speak_tab):
                _speak_refresh[0], _speak_select[0] = build_speak_tab()

            with ui.tab_panel(edit_tab):
                set_edit_callback(build_edit_tab())

            with ui.tab_panel(daw_tab):
                set_daw_callback(build_daw_tab())

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
