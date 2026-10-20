"""Clean-up wizard for reference clips — a spell-check-style review, one clip per screen.

CHECK EVERYTHING analyzes the clips (volume/clipping, leading/trailing silence and
noise, transcripts that disagree with what Whisper hears), then shows each clip that
needs attention on ONE screen: a headline of everything wrong with it ("6.6 dB too
loud · 1.4 s of silence at the end · 2 transcript differences") and a control for each
fix. ACCEPT applies the fixes that are ticked, IGNORE leaves the clip alone, DELETE
removes it. Nothing is applied without you looking at it. Logic lives in
faily/modules/cleanup.py.
"""
import html
import shutil
import time
from pathlib import Path

from nicegui import run as ni_run, ui

from faily.modules import cleanup as K
from faily.modules.importer import commit_stage, discard_stage
from faily.ui.components import bind_play, play_url, show_error

_PASSES = ("volume", "trim", "transcript")
_TITLES = {"volume": "VOLUME", "trim": "TRIM", "transcript": "TRANSCRIPT"}
_PREVIEW_DIR = Path("outputs/_cleanup_preview")
_LBL = "text-[#444] font-mono text-[10px] tracking-widest"
_SECTION = "w-full gap-1 rounded border border-[#222] bg-[#101010] p-3"


def _url(path: Path) -> str:
    return "/outputs/" + path.relative_to(Path("outputs")).as_posix()


def _headline(suggestions: dict[str, dict], clip: K.Clip, staging: bool) -> str:
    """'6.6 dB too loud · 1.4 s of silence at the end · 2 transcript differences'"""
    parts = []
    v = suggestions.get("volume")
    if v:
        if any(i.startswith("clipping") for i in v["issues"]):
            parts.append(f"clipping ({abs(v['gain_db']):.1f} dB of gain reduction suggested)")
        else:
            off = v.get("off_db", -v["gain_db"])
            parts.append(f"{abs(off):.1f} dB too {'loud' if off > 0 else 'quiet'}")
    t = suggestions.get("trim")
    if t:
        lead, tail = t["speech_start"], t["duration"] - t["speech_end"]
        bits = []
        if t["trim_start"] > 0:
            bits.append(f"{lead:.1f} s of space at the start")
        if t["trim_end"] < t["duration"]:
            bits.append(f"{tail:.1f} s at the end" if bits else f"{tail:.1f} s of space at the end")
        if any("noise" in i for i in t["issues"]):
            bits.append("stray noise at the edge")
        parts.append(" and ".join(bits))
    tr = suggestions.get("transcript")
    if tr:
        if staging or not clip.transcript.strip():
            parts.append("transcript to confirm")
        else:
            n = K.count_differences(clip.transcript, tr.get("suggested", ""))
            parts.append(f"{n} transcript difference{'s' if n != 1 else ''}")
    return "  ·  ".join(p for p in parts if p)


def open_cleanup_dialog(char_name: str, on_done=lambda: None, staging: dict | None = None) -> None:
    """Spell-check review of a character's own reference clips.

    staging: set when reviewing freshly imported slices held in a hidden staging
    character (modules/importer.py) — {"target": character they will join,
    "is_new": bool, "on_commit": fn(result)}. The checks start immediately, the
    transcript is pre-filled from Whisper, and the end of the review offers ADD /
    DISCARD instead of just closing."""
    clips: dict[str, K.Clip] = {c.key: c for c in K.own_clips(char_name)}    # key -> current Clip
    if not clips:
        ui.notify("No reference clips of its own to clean up", type="warning")
        return
    by_clip: dict[str, dict[str, dict]] = {}       # clip key -> {pass: suggestion}
    queue: list[str] = []                          # clip keys in review order
    pos = [0]
    stats = {"accepted": 0, "ignored": 0, "deleted": 0}
    state = {"analyzing": False}
    prog = {"text": "", "frac": 0.0}
    cancel = [False]
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # ── layout ───────────────────────────────────────────────────────────
    with ui.dialog().props("persistent").classes("w-full") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full max-w-3xl gap-3 p-5"
    ):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(
                f"IMPORT  ·  {len(clips)} slices for {staging['target']}" if staging else f"CLEAN UP DATA  ·  {char_name}"
            ).classes("text-white font-mono text-xs tracking-widest")
            ui.label(f"{len(clips)} clips").classes("text-[#555] font-mono text-[10px]")

        with ui.column().classes("w-full gap-2") as intro_box:
            ui.label(
                "Checks your reference clips and shows each one that needs attention on a single "
                "screen: everything wrong with it, and a control for each fix. Accept the fixes you "
                "want (adjust them first if you like), ignore the clip, or delete it. Nothing changes "
                "until you accept. Original audio is kept in refs_original/ and deleted clips in "
                "refs_trash/."
            ).classes("text-[#888] font-mono text-[11px] leading-snug")
            for t, d in (
                ("VOLUME", "Clipping, too quiet, or far louder/quieter than the rest → suggested gain."),
                ("TRIM", "Silence at the start/end and stray noise (breaths, clicks) → suggested cut points. "
                         "Whisper double-checks what's in each cut."),
                ("TRANSCRIPT", "Clips whose text disagrees with what Whisper hears → suggested correction. "
                               "Whisper leaves out “um”/“uh”, so those aren't counted as errors."),
            ):
                with ui.row().classes("gap-2 items-baseline flex-nowrap"):
                    ui.label(t).classes("text-amber-500 font-mono text-[10px] tracking-widest w-24 shrink-0")
                    ui.label(d).classes("text-[#666] font-mono text-[10px] leading-snug")
            with ui.row().classes("gap-2 mt-2 items-center"):
                ui.button("CHECK EVERYTHING", icon="spellcheck", on_click=lambda: _start(_PASSES)).props(
                    "color=amber unelevated"
                ).classes("font-mono tracking-widest").tooltip("Run all three checks, then review each clip")
                for p in _PASSES:
                    ui.button(f"{_TITLES[p]} ONLY", on_click=lambda p=p: _start((p,))).props(
                        "flat dense color=grey"
                    ).classes("font-mono text-[10px]")

        with ui.column().classes("w-full gap-1") as progress_box:
            progress_lbl = ui.label("").classes("text-[#888] font-mono text-[11px]")
            progress_bar = ui.linear_progress(value=0, show_value=False).props("color=amber track-color=grey-10")
            ui.button("CANCEL", on_click=lambda: cancel.__setitem__(0, True)).props("flat dense color=negative")
        progress_box.set_visibility(False)

        with ui.column().classes("w-full gap-2") as review_box:
            with ui.row().classes("w-full items-center gap-2"):
                step_lbl = ui.label("").classes("text-amber-500 font-mono text-[10px] tracking-widest flex-grow")
                tally_lbl = ui.label("").classes("text-[#555] font-mono text-[10px]")
            review_bar = ui.linear_progress(value=0, show_value=False, size="3px").props(
                "color=amber track-color=grey-10"
            )
            item_box = ui.column().classes("w-full gap-2 min-h-[220px]")
        review_box.set_visibility(False)

        with ui.row().classes("w-full justify-end"):
            finish_btn = ui.button(
                "FINISH", icon="check", on_click=lambda: _finish_staging() if staging else _close()
            ).props("flat dense color=amber").classes("font-mono tracking-widest")

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _release():
        cancel[0] = True
        shutil.rmtree(_PREVIEW_DIR, ignore_errors=True)
        try:
            from faily.modules.asr import unload
            unload()
        except Exception:
            pass

    def _close():
        _release()
        dlg.close()
        if any(stats.values()):
            on_done()

    def _commit():
        try:
            result = commit_stage(char_name, staging["target"], staging["is_new"])
        except Exception as exc:
            show_error(exc)
            return
        _release()
        dlg.close()
        staging["on_commit"](result)

    def _discard():
        discard_stage(char_name)
        _release()
        dlg.close()

    def _finish_staging():
        """FINISH during an import: ask what to do with the slices."""
        missing = sum(1 for c in clips.values() if not c.transcript.strip())
        with ui.dialog() as d, ui.card().classes("bg-[#1a1a1a] border border-[#333] gap-3"):
            ui.label("LEAVE THE IMPORT?").classes("text-white font-mono text-xs tracking-widest")
            ui.label(
                f"{len(clips)} slices are staged"
                + (f", {missing} without a transcript (they can't be used for training until given one)" if missing else "")
                + ". Slices you haven't reviewed are added as they are."
            ).classes("text-[#888] font-mono text-[11px] max-w-md")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("KEEP REVIEWING", on_click=d.close).props("flat dense color=grey")
                ui.button("DISCARD ALL", on_click=lambda: (d.close(), _discard())).props("flat dense color=negative")
                ui.button(f"ADD {len(clips)}", icon="playlist_add", on_click=lambda: (d.close(), _commit())).props(
                    "color=amber unelevated dense")
        d.open()

    def _advance(kind: str):
        stats[kind] += 1
        pos[0] += 1
        _show()

    # ── actions on the current clip ──────────────────────────────────────
    def _delete(key: str):
        clip = clips[key]
        if not K.delete_clip(char_name, clip):
            return
        clips.pop(key, None)
        ui.notify(f"Deleted {clip.audio.stem} (kept in refs_trash/)", timeout=2500)
        _advance("deleted")

    def _preview(key: str, gain: float, t0: float, t1: float | None):
        out = _PREVIEW_DIR / f"p_{int(time.time() * 1000) % 10**9}.wav"
        try:
            K.render_preview(clips[key].audio, out, gain, t0, t1)
        except Exception as exc:
            show_error(exc)
            return
        play_url(_url(out))

    # ── one clip, all of its fixes ───────────────────────────────────────
    def _show_clip(key: str):
        sug = by_clip[key]
        clip = clips[key]
        use = {}                      # pass -> checkbox
        ctl = {}                      # control widgets by name

        with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
            btn = ui.button(icon="play_arrow").props("round flat color=amber")
            bind_play(btn, url=_url(clip.audio))
            with ui.column().classes("gap-0 min-w-0 flex-grow"):
                ui.label(clip.audio.stem + ("   · padding clip" if clip.source == "buffer" else "")).classes(
                    "text-white font-mono text-sm [overflow-wrap:anywhere]"
                )
                ui.label(_headline(sug, clip, bool(staging))).classes("text-[#e0b13a] font-mono text-[12px]")

        if "volume" in sug:
            r = sug["volume"]
            with ui.column().classes(_SECTION):
                with ui.row().classes("w-full items-center gap-2"):
                    use["volume"] = ui.checkbox("VOLUME", value=True).props("dense color=amber").classes(
                        "font-mono text-[11px] text-[#bbb]"
                    )
                    ui.label("; ".join(r["issues"])).classes("text-[#c9a227] font-mono text-[10px] flex-grow")
                ui.label(f"Speech level {r['level_db']:.0f} dBFS · peak {r['peak']:.2f}"
                         + (f"  —  {r['note']}" if r.get("note") else "")).classes("text-[#666] font-mono text-[10px]")
                with ui.row().classes("items-center gap-2"):
                    ctl["gain"] = ui.number(value=r["gain_db"], format="%.1f", step=0.5, min=-24, max=24, label="change").props(
                        "dense outlined dark suffix=dB"
                    ).classes("w-32 font-mono")

        if "trim" in sug:
            r = sug["trim"]
            with ui.column().classes(_SECTION):
                with ui.row().classes("w-full items-center gap-2"):
                    use["trim"] = ui.checkbox("TRIM", value=True).props("dense color=amber").classes(
                        "font-mono text-[11px] text-[#bbb]"
                    )
                    ui.label("; ".join(r["issues"])).classes("text-[#c9a227] font-mono text-[10px] flex-grow")
                ui.label(f"{r['duration']:.1f} s long · speech runs {r['speech_start']:.1f} – {r['speech_end']:.1f} s").classes(
                    "text-[#666] font-mono text-[10px]"
                )
                if r.get("caution"):
                    ui.label("⚠ " + r["caution"]).classes("text-amber-500 font-mono text-[10px]")
                with ui.row().classes("items-center gap-2"):
                    ctl["t0"] = ui.number(value=r["trim_start"], format="%.2f", step=0.05, min=0, label="keep from").props(
                        "dense outlined dark suffix=s"
                    ).classes("w-32 font-mono")
                    ctl["t1"] = ui.number(value=r["trim_end"], format="%.2f", step=0.05, min=0, label="keep to").props(
                        "dense outlined dark suffix=s"
                    ).classes("w-32 font-mono")

        if "transcript" in sug:
            r = sug["transcript"]
            suggestion = r.get("suggested", "")
            with ui.column().classes(_SECTION):
                with ui.row().classes("w-full items-center gap-2"):
                    use["transcript"] = ui.checkbox("TRANSCRIPT", value=True).props("dense color=amber").classes(
                        "font-mono text-[11px] text-[#bbb]"
                    )
                    ui.label("; ".join(r["issues"])).classes("text-[#c9a227] font-mono text-[10px] flex-grow")
                if staging or not clip.transcript.strip():
                    ui.label("Suggested by Whisper — adjust anything it got wrong").classes(_LBL)
                else:
                    ui.label("YOURS").classes(_LBL)
                    ui.label(clip.transcript).classes("text-[#999] font-mono text-[11px] [overflow-wrap:anywhere]")
                    ui.label("WHISPER HEARS — edit before accepting").classes(_LBL)
                    ui.html("&nbsp;".join(
                        f'<span style="color:{"#aaa" if kind == "same" else "#f59e0b"}">{html.escape(w)}</span>'
                        for w, kind in K.diff_words(clip.transcript, suggestion)
                    ), sanitize=False).classes("font-mono text-[11px] [overflow-wrap:anywhere]")
                ctl["text"] = ui.textarea(value=suggestion).props("outlined dense dark autogrow").classes(
                    "w-full font-mono text-[12px]"
                )
                if K.has_filler(clip.transcript):
                    ui.label("Your text has “um”/“uh” that Whisper leaves out — keep them if they're real.").classes(
                        "text-[#666] font-mono text-[10px]"
                    )

        def audio_args() -> tuple[float, float, float | None]:
            gain = (ctl["gain"].value or 0.0) if "volume" in use and use["volume"].value else 0.0
            if "trim" in use and use["trim"].value:
                return gain, ctl["t0"].value or 0.0, ctl["t1"].value
            return gain, 0.0, None

        def accept():
            gain, t0, t1 = audio_args()
            text = (ctl["text"].value or "").strip() if "transcript" in use and use["transcript"].value else None
            if "transcript" in use and use["transcript"].value and not text:
                ui.notify("The transcript is empty — type one, or untick TRANSCRIPT", type="warning")
                return
            did = False
            cur = clips[key]
            try:
                if gain or t0 > 0 or t1 is not None:
                    new = K.apply_audio_fix(char_name, cur, gain_db=gain, trim_start=t0, trim_end=t1)
                    if new.key != key:                       # mp3 -> wav changes the key
                        clips[new.key] = clips.pop(key)
                    clips[new.key] = cur = new
                    did = True
                if text is not None:
                    K.set_transcript(char_name, cur, text)
                    clips[cur.key] = K.Clip(cur.key, cur.audio, text, cur.source)
                    did = True
            except Exception as exc:
                show_error(exc)
                return
            _advance("accepted" if did else "ignored")

        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ui.button("ACCEPT", icon="check", on_click=accept).props("color=amber unelevated").classes(
                "font-mono tracking-widest"
            ).tooltip("Apply the ticked fixes to this clip")
            ui.button("IGNORE", icon="skip_next", on_click=lambda: _advance("ignored")).props(
                "outline color=grey"
            ).classes("font-mono tracking-widest")
            if "volume" in sug or "trim" in sug:
                ui.button(
                    "PREVIEW", icon="volume_up",
                    on_click=lambda: _preview(key, *audio_args()),
                ).props("flat dense color=grey").classes("font-mono text-[10px]").tooltip(
                    "Hear the clip with the ticked volume and trim fixes applied"
                )
            ui.space()
            if key != K.PRIMARY:
                ui.button("DELETE CLIP", icon="delete_outline", on_click=lambda: _delete(key)).props(
                    "flat color=negative"
                ).classes("font-mono tracking-widest").tooltip(
                    "Remove this clip from the character (kept in refs_trash/)"
                )

    def _show():
        item_box.clear()
        tally_lbl.set_text(f"{stats['accepted']} accepted · {stats['ignored']} ignored · {stats['deleted']} deleted")
        if pos[0] >= len(queue):
            step_lbl.set_text("ALL DONE")
            review_bar.set_value(1.0)
            with item_box:
                ui.icon("check_circle", size="40px").classes("text-green-400 self-center mt-6")
                ui.label(
                    "That's everything." if any(stats.values()) else "Nothing needed changing."
                ).classes("text-white font-mono text-sm self-center")
                if staging:
                    missing = sum(1 for c in clips.values() if not c.transcript.strip())
                    ui.label(f"{len(clips)} clip{'s' if len(clips) != 1 else ''} ready"
                             + (f" — {missing} without a transcript" if missing else "")).classes(
                        "text-[#888] font-mono text-[11px] self-center"
                    )
                    with ui.row().classes("self-center gap-2 mt-2"):
                        ui.button(f"ADD {len(clips)} TO {staging['target'].upper()}", icon="playlist_add",
                                  on_click=_commit).props("color=amber unelevated").classes("font-mono tracking-widest")
                        ui.button("DISCARD ALL", icon="delete_outline", on_click=_discard).props(
                            "flat color=negative").classes("font-mono tracking-widest")
                else:
                    ui.button("DONE", icon="check", on_click=_close).props("color=amber unelevated").classes(
                        "font-mono tracking-widest self-center mt-2"
                    )
            finish_btn.set_visibility(False)
            return
        step_lbl.set_text(f"CLIP {pos[0] + 1} OF {len(queue)}")
        review_bar.set_value(pos[0] / max(len(queue), 1))
        with item_box:
            _show_clip(queue[pos[0]])

    # ── analysis ─────────────────────────────────────────────────────────
    async def _start(passes: tuple[str, ...]):
        if state["analyzing"]:
            return
        state["analyzing"], cancel[0] = True, False
        intro_box.set_visibility(False)
        progress_box.set_visibility(True)
        prog.update(text="Starting…", frac=0.0)

        def on_progress(p, i, n, name):
            idx = passes.index(p)
            prog["frac"] = (idx + i / max(n, 1)) / len(passes)
            prog["text"] = f"{_TITLES[p].lower()} check · {i + 1}/{n} · {name}" + (
                "  (first transcript check loads Whisper — can take ~30 s)" if p == "transcript" and i == 0 else ""
            )

        timer = ui.timer(0.3, lambda: (progress_lbl.set_text(prog["text"]), progress_bar.set_value(prog["frac"])))
        try:
            found = await ni_run.io_bound(K.analyze_all, char_name, passes, None, on_progress, cancel)
        except Exception as exc:
            show_error(exc)
            found = None
        finally:
            timer.cancel()
            state["analyzing"] = False
            try:
                from faily.modules.asr import unload
                unload()
            except Exception:
                pass
        if found is None or cancel[0]:             # failed, cancelled, or app shutting down
            if staging:
                _discard()
                return
            progress_box.set_visibility(False)
            intro_box.set_visibility(True)
            return
        by_clip.clear()
        for p in passes:
            for k, sug in found.get(p, {}).items():
                by_clip.setdefault(k, {})[p] = sug
        queue[:] = [k for k in clips if k in by_clip]          # file order
        pos[0] = 0
        progress_box.set_visibility(False)
        review_box.set_visibility(True)
        _show()

    dlg.open()
    if staging:
        intro_box.set_visibility(False)
        ui.timer(0.2, lambda: _start(_PASSES), once=True)
