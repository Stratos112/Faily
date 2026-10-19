"""Clean-up wizard for a character's reference clips — a spell-check-style review.

CHECK EVERYTHING analyzes the clips (volume/clipping, leading/trailing silence
and noise, transcripts that disagree with what Whisper hears), then walks
through every suggestion one at a time, like a spelling-and-grammar check:
ACCEPT (as suggested or after adjusting it yourself), IGNORE, or DELETE the
clip. Nothing is applied without you looking at it. Logic lives in
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


def _url(path: Path) -> str:
    return "/outputs/" + path.relative_to(Path("outputs")).as_posix()


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
    res: dict[str, dict[str, dict]] = {p: {} for p in _PASSES}     # pass -> key -> suggestion
    queue: list[tuple[str, str]] = []                              # (pass, key) in review order
    pos = [0]
    total = [0]                                                    # items in this review (shrinks on delete)
    handled = [0]
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
                "Checks your reference clips and walks you through each problem it finds, one at a "
                "time, like a spelling check: accept the suggested fix (or adjust it first), ignore "
                "it, or delete the clip. Nothing changes until you accept. Original audio is kept in "
                "refs_original/ and deleted clips in refs_trash/."
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
                ).classes("font-mono tracking-widest").tooltip("Run all three checks, then review each suggestion")
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
            ).props(
                "flat dense color=amber"
            ).classes("font-mono tracking-widest")

    # ── bookkeeping ──────────────────────────────────────────────────────
    def _close():
        cancel[0] = True
        shutil.rmtree(_PREVIEW_DIR, ignore_errors=True)
        try:
            from faily.modules.asr import unload
            unload()
        except Exception:
            pass
        dlg.close()
        if any(stats.values()):
            on_done()

    def _release():
        cancel[0] = True
        shutil.rmtree(_PREVIEW_DIR, ignore_errors=True)
        try:
            from faily.modules.asr import unload
            unload()
        except Exception:
            pass

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

    def _rekey(old: str, new: str):
        """An accepted fix can change a clip's key (mp3 -> wav)."""
        if old == new:
            return
        for p in _PASSES:
            if old in res[p]:
                res[p][new] = res[p].pop(old)
        clips[new] = clips.pop(old)
        for i, (p, k) in enumerate(queue):
            if k == old:
                queue[i] = (p, new)

    def _advance(kind: str | None):
        if kind:
            stats[kind] += 1
        handled[0] += 1
        pos[0] += 1
        _show()

    def _pending_for(key: str) -> int:
        return sum(1 for p, k in queue[pos[0] + 1:] if k == key and k in res[p])

    # ── actions ──────────────────────────────────────────────────────────
    def _accept_audio(key: str, gain: float, t0: float, t1: float | None, p: str):
        try:
            new = K.apply_audio_fix(char_name, clips[key], gain_db=gain, trim_start=t0, trim_end=t1)
        except Exception as exc:
            show_error(exc)
            return
        res[p].pop(key, None)
        _rekey(key, new.key)
        clips[new.key] = new
        _advance("accepted")

    def _accept_transcript(key: str, text: str):
        text = (text or "").strip()
        if not text:
            ui.notify("Transcript is empty", type="warning")
            return
        K.set_transcript(char_name, clips[key], text)
        clips[key] = K.Clip(clips[key].key, clips[key].audio, text, clips[key].source)
        res["transcript"].pop(key, None)
        _advance("accepted")

    def _ignore(p: str, key: str):
        res[p].pop(key, None)
        _advance("ignored")

    def _delete(p: str, key: str):
        clip = clips[key]
        if not K.delete_clip(char_name, clip):
            return
        total[0] -= _pending_for(key)              # its other suggestions vanish with it
        for q in _PASSES:
            res[q].pop(key, None)
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

    # ── rendering one suggestion ─────────────────────────────────────────
    def _actions(p: str, key: str, accept):
        with ui.row().classes("w-full items-center gap-2 mt-2"):
            ui.button("ACCEPT", icon="check", on_click=accept).props("color=amber unelevated").classes(
                "font-mono tracking-widest"
            )
            ui.button("IGNORE", icon="skip_next", on_click=lambda: _ignore(p, key)).props(
                "outline color=grey"
            ).classes("font-mono tracking-widest")
            ui.space()
            if key != K.PRIMARY:
                ui.button("DELETE CLIP", icon="delete_outline", on_click=lambda: _delete(p, key)).props(
                    "flat color=negative"
                ).classes("font-mono tracking-widest").tooltip(
                    "Remove this clip from the character (kept in refs_trash/)"
                )

    def _head(key: str, r: dict):
        clip = clips[key]
        with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
            btn = ui.button(icon="play_arrow").props("round flat color=amber")
            bind_play(btn, url=_url(clip.audio))
            with ui.column().classes("gap-0 min-w-0 flex-grow"):
                ui.label(clip.audio.stem + ("   · padding clip" if clip.source == "buffer" else "")).classes(
                    "text-white font-mono text-sm [overflow-wrap:anywhere]"
                )
                ui.label("; ".join(r["issues"])).classes("text-[#c9a227] font-mono text-[11px]")

    def _show_volume(key: str):
        r = res["volume"][key]
        _head(key, r)
        ui.label(f"Speech level {r['level_db']:.0f} dBFS · peak {r['peak']:.2f}"
                 + (f"  —  {r['note']}" if r.get("note") else "")).classes("text-[#666] font-mono text-[10px]")
        ui.label("SUGGESTED CHANGE").classes(_LBL)
        with ui.row().classes("items-center gap-2"):
            gain = ui.number(value=r["gain_db"], format="%.1f", step=0.5, min=-24, max=24).props(
                "dense outlined dark suffix=dB"
            ).classes("w-32 font-mono")
            ui.button("PREVIEW", icon="volume_up", on_click=lambda: _preview(key, gain.value or 0.0, 0.0, None)).props(
                "flat dense color=grey"
            ).classes("font-mono text-[10px]")
        _actions("volume", key, lambda: _accept_audio(key, gain.value or 0.0, 0.0, None, "volume"))

    def _show_trim(key: str):
        r = res["trim"][key]
        _head(key, r)
        ui.label(f"{r['duration']:.1f}s long · speech runs {r['speech_start']:.1f}s – {r['speech_end']:.1f}s").classes(
            "text-[#666] font-mono text-[10px]"
        )
        if r.get("caution"):
            ui.label("⚠ " + r["caution"]).classes("text-amber-500 font-mono text-[10px]")
        ui.label("KEEP FROM / TO").classes(_LBL)
        with ui.row().classes("items-center gap-2"):
            t0 = ui.number(value=r["trim_start"], format="%.2f", step=0.05, min=0).props(
                "dense outlined dark suffix=s"
            ).classes("w-28 font-mono")
            t1 = ui.number(value=r["trim_end"], format="%.2f", step=0.05, min=0).props(
                "dense outlined dark suffix=s"
            ).classes("w-28 font-mono")
            ui.button("PREVIEW", icon="content_cut",
                      on_click=lambda: _preview(key, 0.0, t0.value or 0.0, t1.value)).props(
                "flat dense color=grey"
            ).classes("font-mono text-[10px]")
        _actions("trim", key, lambda: _accept_audio(key, 0.0, t0.value or 0.0, t1.value, "trim"))

    def _show_transcript(key: str):
        r = res["transcript"][key]
        clip = clips[key]
        _head(key, r)
        sug = r.get("suggested", "")
        if staging or not clip.transcript.strip():
            ui.label("TRANSCRIPT — suggested by Whisper; adjust anything it got wrong").classes(_LBL)
        else:
            ui.label("YOUR TRANSCRIPT").classes(_LBL)
            ui.label(clip.transcript).classes("text-[#999] font-mono text-[11px] [overflow-wrap:anywhere]")
            ui.label("WHISPER HEARS — edit before accepting").classes(_LBL)
            ui.html("&nbsp;".join(
                f'<span style="color:{"#aaa" if kind == "same" else "#f59e0b"}">{html.escape(w)}</span>'
                for w, kind in K.diff_words(clip.transcript, sug)
            ), sanitize=False).classes("font-mono text-[11px] [overflow-wrap:anywhere]")
        box = ui.textarea(value=sug).props("outlined dense dark autogrow").classes("w-full font-mono text-[12px]")
        if K.has_filler(clip.transcript):
            ui.label("Your text has “um”/“uh” that Whisper leaves out — keep them if they're real.").classes(
                "text-[#666] font-mono text-[10px]"
            )
        _actions("transcript", key, lambda: _accept_transcript(key, box.value))

    def _show():
        item_box.clear()
        while pos[0] < len(queue) and queue[pos[0]][1] not in res[queue[pos[0]][0]]:
            pos[0] += 1                           # already handled (e.g. clip deleted)
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
        p, key = queue[pos[0]]
        step_lbl.set_text(f"{_TITLES[p]}  ·  {min(handled[0] + 1, total[0])} of {total[0]}")
        review_bar.set_value(handled[0] / max(total[0], 1))
        with item_box:
            {"volume": _show_volume, "trim": _show_trim, "transcript": _show_transcript}[p](key)

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
        for p in passes:
            res[p] = found.get(p, {})
        queue[:] = [(p, k) for p in _PASSES for k in res[p]]
        total[0], handled[0], pos[0] = len(queue), 0, 0
        progress_box.set_visibility(False)
        review_box.set_visibility(True)
        _show()

    dlg.open()
    if staging:
        intro_box.set_visibility(False)
        ui.timer(0.2, lambda: _start(_PASSES), once=True)
