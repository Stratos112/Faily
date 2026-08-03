from pathlib import Path
from nicegui import ui, run as ni_run
from faily.modules.edit import audio_info, mix_tracks
from faily.ui.components import section_label, show_error

_BTN = "font-mono tracking-widest"
EDIT_DIR = Path("outputs/edit")

_DAW_JS = """
window.FailyDAW = (function () {
  let ctx = null;
  const T = [{}, {}, {}];

  function _ac() {
    if (!ctx) ctx = new AudioContext();
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  async function loadTrack(idx, url) {
    const ac = _ac();
    T[idx].buffer = null;
    try {
      const res = await fetch(url);
      const buf = await ac.decodeAudioData(await res.arrayBuffer());
      T[idx].buffer = buf;
      if (!T[idx].gain) {
        T[idx].gain = ac.createGain();
        T[idx].gain.gain.value = T[idx].vol != null ? T[idx].vol : 1.0;
        T[idx].gain.connect(ac.destination);
      }
    } catch (e) { console.error('DAW loadTrack', idx, e); }
  }

  function clearTrack(idx) {
    if (T[idx].src) { try { T[idx].src.stop(); } catch (_) {} T[idx].src = null; }
    T[idx].buffer = null;
  }

  function setGain(idx, value) {
    T[idx].vol = value;
    if (T[idx].gain) T[idx].gain.gain.setTargetAtTime(value, _ac().currentTime, 0.015);
  }

  function playAll() {
    const ac = _ac();
    stopAll();
    const t0 = ac.currentTime + 0.08;
    for (let i = 0; i < 3; i++) {
      if (!T[i].buffer) continue;
      if (!T[i].gain) {
        T[i].gain = ac.createGain();
        T[i].gain.gain.value = T[i].vol != null ? T[i].vol : 1.0;
        T[i].gain.connect(ac.destination);
      }
      const src = ac.createBufferSource();
      src.buffer = T[i].buffer;
      src.connect(T[i].gain);
      src.start(t0);
      T[i].src = src;
    }
  }

  function stopAll() {
    for (let i = 0; i < 3; i++) {
      if (T[i].src) { try { T[i].src.stop(); } catch (_) {} T[i].src = null; }
    }
  }

  return { loadTrack, clearTrack, setGain, playAll, stopAll };
})();
"""


def build_daw_tab():
    """Returns send_to_daw(path, char_name=None) callable."""

    ui.add_head_html(f"<script>{_DAW_JS}</script>")

    # ── per-track Python state ────────────────────────────────────────────────
    _paths: list[Path | None] = [None, None, None]
    _vols:  list[float]       = [1.0,  1.0,  1.0]
    _muted: list[bool]        = [False, False, False]

    # UI element refs filled in _build_track
    _players:    list = [None, None, None]
    _info_lbls:  list = [None, None, None]
    _mute_show:  list = [None, None, None]  # (mute_btn, unmute_btn) per track
    _vol_lbls:   list = [None, None, None]

    def _load(idx: int, path: Path):
        _paths[idx] = path
        try:
            rel = path.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"
        except ValueError:
            url = f"/outputs/edit/{path.name}"
        _players[idx].set_source(url)
        try:
            info = audio_info(path)
            dur = f"{info['duration']:.1f}s"
            ch  = "S" if info["channels"] > 1 else "M"
            _info_lbls[idx].set_text(f"{path.stem}  ·  {dur}  ·  {ch}")
        except Exception:
            _info_lbls[idx].set_text(path.stem)
        ui.run_javascript(f"FailyDAW.loadTrack({idx}, '{url}')")

    def _clear(idx: int):
        _paths[idx] = None
        _players[idx].set_source("")
        _info_lbls[idx].set_text("empty")
        ui.run_javascript(f"FailyDAW.clearTrack({idx})")

    def _set_vol(idx: int, val: float):
        _vols[idx] = val
        effective = 0.0 if _muted[idx] else val
        _vol_lbls[idx].set_text(f"{int(val * 100)}%")
        ui.run_javascript(f"FailyDAW.setGain({idx}, {effective})")

    def _toggle_mute(idx: int):
        _muted[idx] = not _muted[idx]
        muted = _muted[idx]
        effective = 0.0 if muted else _vols[idx]
        ui.run_javascript(f"FailyDAW.setGain({idx}, {effective})")
        mute_btn, unmute_btn = _mute_show[idx]
        mute_btn.set_visibility(not muted)
        unmute_btn.set_visibility(muted)

    def _build_track(idx: int):
        def _make_upload(i):
            async def _handler(e):
                EDIT_DIR.mkdir(parents=True, exist_ok=True)
                dest = EDIT_DIR / e.file.name
                dest.write_bytes(await e.file.read())
                _load(i, dest)
            return _handler

        with ui.column().classes(
            "gap-3 p-5 rounded border border-[#1e1e1e] bg-[#0c0c0c] w-full h-full"
        ):
            # header
            with ui.row().classes("w-full items-center gap-2"):
                ui.label(f"TRACK  {idx + 1}").classes(
                    "text-amber-500 font-mono text-[10px] tracking-[0.3em] flex-grow"
                )
                ui.button(
                    icon="close",
                    on_click=lambda i=idx: _clear(i),
                ).props("flat dense color=grey").classes("shrink-0 opacity-30 hover:opacity-100").tooltip("Clear")

            info_lbl = ui.label("empty").classes("text-[#333] font-mono text-[10px] truncate")
            _info_lbls[idx] = info_lbl

            player = ui.audio("").classes("w-full rounded")
            _players[idx] = player

            # volume
            with ui.row().classes("w-full items-center gap-2 mt-1"):
                ui.icon("volume_up", size="14px").classes("text-[#333] shrink-0")
                vol_lbl = ui.label("100%").classes(
                    "font-mono text-[10px] text-amber-400 w-9 shrink-0 text-right"
                )
                _vol_lbls[idx] = vol_lbl

                def _on_vol(e, i=idx):
                    _set_vol(i, float(e.value))
                ui.slider(
                    min=0.0, max=1.0, step=0.01, value=1.0, on_change=_on_vol,
                ).classes("flex-grow").props("color=amber")

            # mute / unmute (swapping pair)
            with ui.row().classes("w-full gap-1"):
                mute_btn = (
                    ui.button("MUTE", icon="volume_off", on_click=lambda i=idx: _toggle_mute(i))
                    .props("flat dense color=grey").classes(f"flex-1 {_BTN}")
                )
                unmute_btn = (
                    ui.button("MUTED", icon="volume_off", on_click=lambda i=idx: _toggle_mute(i))
                    .props("unelevated dense color=negative").classes(f"flex-1 {_BTN}")
                )
                unmute_btn.set_visibility(False)
                _mute_show[idx] = (mute_btn, unmute_btn)

            # upload
            (
                ui.upload(
                    on_upload=_make_upload(idx),
                    multiple=False, auto_upload=True,
                )
                .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Load audio'")
                .classes("w-full")
            )

    # ── layout ────────────────────────────────────────────────────────────────
    with ui.column().classes("w-full h-full gap-0 p-6"):

        with ui.grid(columns=3).classes("w-full gap-5 flex-grow"):
            for i in range(3):
                _build_track(i)

        ui.separator().classes("my-5 opacity-20")

        with ui.column().classes("w-full items-center gap-2"):
            with ui.row().classes("items-center justify-center gap-6"):
                (
                    ui.button("PLAY ALL", icon="play_arrow", on_click=lambda: ui.run_javascript("FailyDAW.playAll()"))
                    .props("color=amber unelevated")
                    .classes(f"px-10 {_BTN}")
                )
                (
                    ui.button("STOP", icon="stop", on_click=lambda: ui.run_javascript("FailyDAW.stopAll()"))
                    .props("flat color=grey")
                    .classes(f"px-8 {_BTN}")
                )

            mix_lbl = ui.label("").classes("text-[#444] font-mono text-[9px]")

            async def _mix_download():
                tracks_data = [
                    {"path": _paths[i], "vol": _vols[i], "muted": _muted[i]}
                    for i in range(3)
                ]
                if all(not t["path"] for t in tracks_data):
                    ui.notify("No tracks loaded", type="warning")
                    return
                mix_btn.disable()
                mix_lbl.set_text("mixing…")
                try:
                    import datetime
                    from faily.core.settings import get_download_dir
                    downloads = get_download_dir()
                    downloads.mkdir(parents=True, exist_ok=True)
                    out = downloads / f"faily_mix_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
                    await ni_run.io_bound(mix_tracks, tracks_data, out)
                    mix_lbl.set_text(f"✓  {out.name}")
                    ui.notify(f"Saved to {downloads.name}/{out.name}", type="positive", timeout=3000)
                except Exception as exc:
                    show_error(exc)
                    mix_lbl.set_text("error")
                finally:
                    mix_btn.enable()

            mix_btn = (
                ui.button("MIX & DOWNLOAD", icon="merge", on_click=_mix_download)
                .props("flat color=amber")
                .classes(_BTN)
            )

    # ── public callback ───────────────────────────────────────────────────────
    def send_to_daw(path: Path, char_name: str | None = None):
        for i in range(3):
            if _paths[i] is None:
                _load(i, path)
                ui.notify(f"Loaded on Track {i + 1} — switch to DAW", timeout=2000)
                return
        # all tracks loaded — ask which to replace
        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#333] min-w-[300px] gap-3"
        ):
            ui.label("SEND TO DAW").classes("text-white font-mono text-xs tracking-widest")
            ui.label("All tracks are in use. Replace which one?").classes(
                "text-[#555] font-mono text-[10px]"
            )
            with ui.row().classes("w-full justify-center gap-3 mt-1"):
                for i in range(3):
                    def _replace(i=i):
                        _load(i, path)
                        dlg.close()
                        ui.notify(f"Track {i + 1} replaced", timeout=1500)
                    ui.button(
                        f"TRACK {i + 1}",
                        on_click=_replace,
                    ).props("color=amber unelevated").classes("font-mono text-[10px]")
            ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey").classes("w-full mt-1")
        dlg.open()

    return send_to_daw
