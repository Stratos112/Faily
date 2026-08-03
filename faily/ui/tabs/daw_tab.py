import json
from pathlib import Path
from nicegui import ui, run as ni_run
from faily.modules.edit import audio_info, mix_tracks
from faily.core.characters import (
    list_characters, get_character, CHARACTERS_DIR,
    list_character_clips, list_character_favorites,
)
from faily.ui.components import section_label, show_error

_BTN = "font-mono tracking-widest"
_NO_CHAR = "— select —"

# px constants that JS and Python must agree on
_RH = 20   # ruler height
_TH = 72   # track height

_DAW_JS = """
window.FailyDAW = (function () {
  const PPS = 80;
  const RH = 20, TH = 72;
  const C = [
    ['rgba(245,158,11,', '#f59e0b'],
    ['rgba(129,140,248,', '#818cf8'],
    ['rgba(52,211,153,',  '#34d399'],
  ];

  const T = [0,1,2].map(() => ({
    name:'', url:'', buffer:null, gain:null, src:null,
    vol:1.0, muted:false, offset:0.0, duration:0.0, stereo:false,
  }));
  let ctx = null, drag = null;

  function _ac() {
    if (!ctx) ctx = new AudioContext();
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  function render() {
    const tl = document.getElementById('faily-daw-tl');
    if (!tl) return;
    const pw = tl.parentElement ? tl.parentElement.clientWidth : 800;
    const totalDur = Math.max(...T.map(t => t.offset + t.duration), 10) + 2;
    const W = Math.max(Math.ceil(totalDur * PPS), pw);
    const H = RH + T.length * TH;
    tl.style.cssText = `position:relative;width:${W}px;height:${H}px;`;

    // keep existing children tagged for reuse avoidance — just rebuild
    tl.innerHTML = '';

    // ruler
    for (let s = 0; s <= Math.ceil(totalDur); s += 2) {
      const m = document.createElement('div');
      m.style.cssText = `position:absolute;left:${s*PPS}px;top:0;height:${RH}px;`+
        `border-left:1px solid #1e1e1e;padding-left:3px;`+
        `font:9px monospace;color:#333;line-height:${RH}px;white-space:nowrap;`;
      m.textContent = s + 's';
      tl.appendChild(m);
    }

    // track stripes + clips
    for (let i = 0; i < T.length; i++) {
      const top = RH + i * TH;
      const stripe = document.createElement('div');
      stripe.style.cssText = `position:absolute;left:0;top:${top}px;width:100%;height:${TH}px;`+
        `border-top:1px solid #1a1a1a;background:${i%2===0?'#0c0c0c':'#0a0a0a'};`;
      tl.appendChild(stripe);

      const t = T[i];
      if (!t.duration) continue;
      const left = Math.round(t.offset * PPS);
      const width = Math.max(Math.round(t.duration * PPS), 8);
      const [fill, border] = C[i];
      const clip = document.createElement('div');
      clip.style.cssText = [
        `position:absolute`,
        `left:${left}px`,
        `top:${top+4}px`,
        `width:${width}px`,
        `height:${TH-8}px`,
        `background:${fill}${t.muted?'0.18)':'0.6)'}`  ,
        `border:1px solid ${fill}${t.muted?'0.3)':'0.85)'}`  ,
        `border-radius:3px`,
        `cursor:grab`,
        `user-select:none`,
        `overflow:hidden`,
        `box-sizing:border-box`,
      ].join(';');

      const lbl = document.createElement('div');
      lbl.style.cssText = `padding:4px 6px 1px;font:bold 10px monospace;color:${t.muted?'#555':'rgba(0,0,0,0.85)'};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`;
      lbl.textContent = t.name;
      const sub = document.createElement('div');
      sub.style.cssText = `padding:0 6px;font:9px monospace;color:${t.muted?'#444':'rgba(0,0,0,0.55)'};`;
      sub.textContent = t.duration.toFixed(1) + 's' + (t.stereo ? '  ST' : '  MO') +
        (t.offset > 0 ? '  @' + t.offset.toFixed(1) + 's' : '');
      clip.appendChild(lbl);
      clip.appendChild(sub);

      clip.addEventListener('mousedown', (e) => {
        drag = {i, x0: e.clientX, o0: T[i].offset};
        document.addEventListener('mousemove', _drag);
        document.addEventListener('mouseup', _drop);
        e.preventDefault();
      });
      tl.appendChild(clip);
    }
  }

  function _drag(e) {
    if (!drag) return;
    T[drag.i].offset = Math.max(0, drag.o0 + (e.clientX - drag.x0) / PPS);
    render();
  }
  function _drop() {
    drag = null;
    document.removeEventListener('mousemove', _drag);
    document.removeEventListener('mouseup', _drop);
  }

  async function loadTrack(idx, url, name, duration, stereo) {
    T[idx].url = url; T[idx].name = name;
    T[idx].duration = duration; T[idx].stereo = !!stereo;
    T[idx].buffer = null;
    render();
    const ac = _ac();
    try {
      const res = await fetch(url);
      T[idx].buffer = await ac.decodeAudioData(await res.arrayBuffer());
      if (!T[idx].gain) {
        T[idx].gain = ac.createGain();
        T[idx].gain.gain.value = T[idx].vol;
        T[idx].gain.connect(ac.destination);
      }
    } catch(e) { console.error('DAW load', idx, e); }
    render();
  }

  function clearTrack(idx) {
    if (T[idx].src) { try { T[idx].src.stop(); } catch(_){} T[idx].src = null; }
    Object.assign(T[idx], {buffer:null, duration:0, name:'', url:'', offset:0, stereo:false});
    render();
  }

  function setGain(idx, val) {
    T[idx].vol = val;
    if (T[idx].gain) T[idx].gain.gain.setTargetAtTime(val, _ac().currentTime, 0.015);
  }

  function setMuted(idx, muted) {
    T[idx].muted = !!muted;
    if (T[idx].gain)
      T[idx].gain.gain.setTargetAtTime(muted ? 0 : T[idx].vol, _ac().currentTime, 0.015);
    render();
  }

  function playAll() {
    const ac = _ac(); stopAll();
    const t0 = ac.currentTime + 0.08;
    for (let i = 0; i < T.length; i++) {
      if (!T[i].buffer || T[i].muted) continue;
      if (!T[i].gain) {
        T[i].gain = ac.createGain();
        T[i].gain.gain.value = T[i].vol;
        T[i].gain.connect(ac.destination);
      }
      const src = ac.createBufferSource();
      src.buffer = T[i].buffer;
      src.connect(T[i].gain);
      src.start(t0 + T[i].offset);
      T[i].src = src;
    }
  }

  function stopAll() {
    for (let i = 0; i < T.length; i++)
      if (T[i].src) { try { T[i].src.stop(); } catch(_){} T[i].src = null; }
  }

  function getOffsets() { return T.map(t => t.offset); }

  return { loadTrack, clearTrack, setGain, setMuted, playAll, stopAll, getOffsets, render };
})();
"""

_TRACK_COLORS = ["text-amber-400", "text-indigo-400", "text-emerald-400"]
_TRACK_NAMES  = ["TRACK 1", "TRACK 2", "TRACK 3"]


def build_daw_tab():
    """Returns send_to_daw(path, char_name=None) callable."""

    ui.add_head_html(f"<script>{_DAW_JS}</script>")

    _paths: list[Path | None] = [None, None, None]
    _vols:  list[float]       = [1.0,  1.0,  1.0]
    _muted: list[bool]        = [False, False, False]

    _vol_lbls:   list = [None, None, None]
    _mute_btns:  list = [None, None, None]
    _umute_btns: list = [None, None, None]

    def _load(idx: int, path: Path):
        _paths[idx] = path
        try:
            rel = path.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"
        except ValueError:
            url = f"/outputs/edit/{path.name}"
        try:
            info = audio_info(path)
            dur    = info["duration"]
            stereo = info["channels"] > 1
        except Exception:
            dur, stereo = 0.0, False
        _vol_lbls[idx].set_text(f"{int(_vols[idx] * 100)}%")
        ui.run_javascript(
            f"FailyDAW.loadTrack({idx}, '{url}', {json.dumps(path.stem)}, "
            f"{dur:.3f}, {'true' if stereo else 'false'})"
        )

    def _clear(idx: int):
        _paths[idx] = None
        ui.run_javascript(f"FailyDAW.clearTrack({idx})")

    def _set_vol(idx: int, val: float):
        _vols[idx] = val
        effective = 0.0 if _muted[idx] else val
        _vol_lbls[idx].set_text(f"{int(val * 100)}%")
        ui.run_javascript(f"FailyDAW.setGain({idx}, {effective})")

    def _toggle_mute(idx: int):
        _muted[idx] = not _muted[idx]
        muted = _muted[idx]
        ui.run_javascript(f"FailyDAW.setMuted({idx}, {'true' if muted else 'false'})")
        _mute_btns[idx].set_visibility(not muted)
        _umute_btns[idx].set_visibility(muted)

    # ── layout ────────────────────────────────────────────────────────────────
    with ui.column().classes("w-full h-full gap-0"):

        # ── TIMELINE ROW ──────────────────────────────────────────────────────
        with ui.row().classes("w-full flex-grow gap-0 overflow-hidden"):

            # LEFT: fixed-width track headers
            with ui.column().classes("gap-0 shrink-0 border-r border-[#1a1a1a]").style(
                "width:190px; background:#0a0a0a;"
            ):
                # ruler spacer
                with ui.row().classes("items-center px-3").style(
                    f"height:{_RH}px; min-height:{_RH}px; border-bottom:1px solid #1a1a1a;"
                ):
                    ui.label("TIMELINE").classes("text-[#252525] font-mono text-[9px] tracking-widest")

                # per-track headers
                for i in range(3):
                    bg = "#0c0c0c" if i % 2 == 0 else "#0a0a0a"
                    with ui.column().classes("gap-1.5 justify-center px-3").style(
                        f"height:{_TH}px; min-height:{_TH}px; background:{bg}; "
                        "border-top:1px solid #1a1a1a; box-sizing:border-box;"
                    ):
                        # name row
                        with ui.row().classes("items-center gap-1 w-full"):
                            ui.label(_TRACK_NAMES[i]).classes(
                                f"{_TRACK_COLORS[i]} font-mono text-[10px] tracking-widest flex-grow"
                            )
                            ui.button(icon="close", on_click=lambda i=i: _clear(i)).props(
                                "flat dense color=grey"
                            ).classes("shrink-0 opacity-30 hover:opacity-100").tooltip("Clear")

                        # vol + mute row
                        with ui.row().classes("items-center gap-1 w-full"):
                            mute_btn = (
                                ui.button("M", on_click=lambda i=i: _toggle_mute(i))
                                .props("flat dense color=grey")
                                .classes("font-mono text-[9px] shrink-0 px-1")
                                .tooltip("Mute")
                            )
                            umute_btn = (
                                ui.button("M", on_click=lambda i=i: _toggle_mute(i))
                                .props("unelevated dense color=negative")
                                .classes("font-mono text-[9px] shrink-0 px-1")
                                .tooltip("Unmute")
                            )
                            umute_btn.set_visibility(False)
                            _mute_btns[i]  = mute_btn
                            _umute_btns[i] = umute_btn

                            vol_lbl = ui.label("100%").classes(
                                "font-mono text-[9px] text-amber-400 w-8 shrink-0 text-right"
                            )
                            _vol_lbls[i] = vol_lbl

                            def _on_vol(e, i=i):
                                _set_vol(i, float(e.value))
                            ui.slider(
                                min=0.0, max=1.0, step=0.01, value=1.0, on_change=_on_vol,
                            ).classes("flex-grow").props("color=amber")

            # RIGHT: JS timeline canvas
            with ui.element("div").classes("flex-grow overflow-x-auto overflow-y-hidden").style(
                "background:#0a0a0a;"
            ):
                ui.html('<div id="faily-daw-tl"></div>')

        ui.separator().classes("opacity-10 shrink-0")

        # ── CHARACTER BROWSER (collapsible) ───────────────────────────────────
        _show: list[bool] = [False]
        browser_col = ui.column().classes("w-full gap-2 px-6 pb-3")
        browser_col.set_visibility(False)

        def _toggle_browser():
            _show[0] = not _show[0]
            browser_col.set_visibility(_show[0])
            toggle_btn.set_text("CHARACTERS ▲" if _show[0] else "CHARACTERS ▼")
            if _show[0]:
                char_pick.set_options({_NO_CHAR: _NO_CHAR, **_char_opts()}, value=_NO_CHAR)

        toggle_btn = (
            ui.button("CHARACTERS ▼", on_click=_toggle_browser)
            .props("flat dense color=grey")
            .classes("font-mono text-[9px] tracking-widest w-full shrink-0")
        )

        with browser_col:
            def _char_opts() -> dict:
                return {c["name"]: c["name"] for c in list_characters()}

            with ui.row().classes("w-full items-center gap-2"):
                char_pick = (
                    ui.select(options={_NO_CHAR: _NO_CHAR}, value=_NO_CHAR)
                    .props("outlined dark dense")
                    .classes("flex-grow")
                )
                ui.button(
                    icon="refresh",
                    on_click=lambda: char_pick.set_options(
                        {_NO_CHAR: _NO_CHAR, **_char_opts()}, value=_NO_CHAR
                    ),
                ).props("flat dense color=grey")

            pick_list = ui.row().classes("w-full flex-wrap gap-2")

            def _clip_chip(path: Path, prefix: str):
                with pick_list:
                    with ui.row().classes(
                        "items-center gap-1 px-2 py-1 rounded border border-[#1e1e1e] bg-[#0c0c0c]"
                    ):
                        ui.label(prefix).classes("text-amber-500 font-mono text-[10px] shrink-0")
                        ui.label(path.stem).classes(
                            "text-[#666] font-mono text-[10px] truncate max-w-[120px]"
                        )
                        for i in range(3):
                            ui.button(
                                f"T{i+1}",
                                on_click=lambda p=path, i=i: (
                                    _load(i, p),
                                    ui.notify(f"Loaded on Track {i+1}", timeout=1500),
                                ),
                            ).props("flat dense color=grey").classes("font-mono text-[9px] shrink-0 px-1")

            def _on_char_pick(name: str):
                pick_list.clear()
                if name == _NO_CHAR:
                    return
                char = get_character(name)
                if not char:
                    return
                char_dir = CHARACTERS_DIR / name
                if "ref_audio" in char:
                    ra = char_dir / char["ref_audio"]
                    if ra.exists():
                        _clip_chip(ra, "★")
                for rc in char.get("ref_clips", []):
                    rp = char_dir / rc["file"]
                    if rp.exists():
                        _clip_chip(rp, "●")
                for c in list_character_clips(name):
                    _clip_chip(c, "◆")
                for f in list_character_favorites(name):
                    _clip_chip(f, "♥")

            char_pick.on_value_change(lambda e: _on_char_pick(e.value))

        ui.separator().classes("opacity-10 shrink-0")

        # ── MASTER CONTROLS ───────────────────────────────────────────────────
        with ui.column().classes("w-full items-center gap-2 py-4 shrink-0"):
            with ui.row().classes("items-center gap-6"):
                (
                    ui.button("PLAY ALL", icon="play_arrow",
                              on_click=lambda: ui.run_javascript("FailyDAW.playAll()"))
                    .props("color=amber unelevated")
                    .classes(f"px-10 {_BTN}")
                )
                (
                    ui.button("STOP", icon="stop",
                              on_click=lambda: ui.run_javascript("FailyDAW.stopAll()"))
                    .props("flat color=grey")
                    .classes(f"px-8 {_BTN}")
                )

            mix_lbl = ui.label("").classes("text-[#444] font-mono text-[9px]")

            async def _mix_download():
                offsets_raw = await ui.run_javascript(
                    "JSON.stringify(FailyDAW.getOffsets())", respond=True
                )
                offsets = json.loads(offsets_raw) if offsets_raw else [0.0, 0.0, 0.0]
                tracks_data = [
                    {"path": _paths[i], "vol": _vols[i], "muted": _muted[i], "offset": offsets[i]}
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
                    dl = get_download_dir()
                    dl.mkdir(parents=True, exist_ok=True)
                    out = dl / f"faily_mix_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
                    await ni_run.io_bound(mix_tracks, tracks_data, out)
                    mix_lbl.set_text(f"✓  {out.name}")
                    ui.notify(f"Saved to {dl.name}/{out.name}", type="positive", timeout=3000)
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

    # initial render after DOM is ready
    ui.timer(0.4, lambda: ui.run_javascript("FailyDAW.render()"), once=True)

    # ── public callback ───────────────────────────────────────────────────────
    def send_to_daw(path: Path, char_name: str | None = None):
        for i in range(3):
            if _paths[i] is None:
                _load(i, path)
                ui.notify(f"Loaded on Track {i + 1} — switch to DAW", timeout=2000)
                return
        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#333] min-w-[280px] gap-3"
        ):
            ui.label("REPLACE WHICH TRACK?").classes("text-white font-mono text-xs tracking-widest")
            with ui.row().classes("w-full justify-center gap-3 mt-1"):
                for i in range(3):
                    def _replace(i=i):
                        _load(i, path)
                        dlg.close()
                        ui.notify(f"Track {i + 1} replaced", timeout=1500)
                    ui.button(
                        f"TRACK {i + 1}", on_click=_replace,
                    ).props("color=amber unelevated").classes("font-mono text-[10px]")
            ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey").classes("w-full mt-1")
        dlg.open()

    return send_to_daw
