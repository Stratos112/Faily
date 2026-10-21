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
from faily.modules import padding_audit as PA
from faily.modules.importer import commit_stage, discard_stage
from faily.ui.components import bind_play, play_url, show_error
from faily.ui.waveform import region_picker

_PASSES = ("volume", "trim", "transcript")
_TITLES = {"volume": "VOLUME", "trim": "TRIM", "transcript": "TRANSCRIPT", "artifacts": "NOISES"}
_PREVIEW_DIR = Path("outputs/_cleanup_preview")
_LBL = "text-[#444] font-mono text-[10px] tracking-widest"
_SECTION = "w-full gap-1 rounded border border-[#3a3217] bg-[#14110a] p-3"      # has an issue
_SECTION_OK = "w-full gap-1 rounded border border-[#1d3a26] bg-[#0d1a12] p-3"   # nothing wrong


def _url(path: Path) -> str:
    return "/outputs/" + path.relative_to(Path("outputs")).as_posix()


def _headline(suggestions: dict[str, dict], clip: K.Clip, staging: bool) -> str:
    """'6.6 dB too loud · 1.4 s of silence at the end · 2 transcript differences' — only the
    characteristics that have a problem; 'no issues found' if none do."""
    parts = []
    v = suggestions.get("volume")
    if v and v["issues"]:
        if any(i.startswith("clipping") for i in v["issues"]):
            parts.append(f"clipping ({abs(v['gain_db']):.1f} dB of gain reduction suggested)")
        else:
            off = v.get("off_db", -v["gain_db"])
            parts.append(f"{abs(off):.1f} dB too {'loud' if off > 0 else 'quiet'}")
    t = suggestions.get("trim")
    if t and t["issues"]:
        lead, tail = t["speech_start"], t["duration"] - t["speech_end"]
        bits = []
        if t["trim_start"] > 0:
            bits.append(f"{lead:.1f} s of space at the start")
        if t["trim_end"] < t["duration"]:
            bits.append(f"{tail:.1f} s at the end" if bits else f"{tail:.1f} s of space at the end")
        if any("noise" in i for i in t["issues"]):
            bits.append("stray noise at the edge")
        bits.extend(i for i in t["issues"] if "extra word" in i)
        parts.append(" and ".join(bits))
    pa = suggestions.get("artifacts")
    if pa:
        parts.extend(pa["issues"])
    tr = suggestions.get("transcript")
    if tr and tr["issues"]:
        if staging or not clip.transcript.strip():
            parts.append("transcript to confirm")
        else:
            n = K.count_differences(clip.transcript, tr.get("suggested", ""))
            parts.append(f"{n} transcript difference{'s' if n != 1 else ''}")
    return "  ·  ".join(p for p in parts if p) or "no issues found"


def _needs_attention(sug: dict[str, dict]) -> bool:
    for p, r in sug.items():
        if p == "artifacts":
            if (r.get("issues") or r.get("mute") or r.get("recommend_delete")
                    or (r.get("words") and r["words"]["kind"] == "mismatch")):
                return True
        elif r.get("issues"):
            return True
    return False


def _fold_audit_into_trim(sug: dict[str, dict]) -> None:
    """A generated clip is checked twice for things at its edges (the normal trim check
    and the audit's noise/extra-word check). Show one TRIM section: the tighter of the
    two cut windows, with both findings. What's left of the audit (noises between words,
    wrong words, duplicate, background) stays under "artifacts"."""
    audit = sug.get("artifacts")
    if not audit:
        return
    at, gt = audit.get("trim"), sug.get("trim")
    edge = list(audit.get("edge_issues", []))
    if at:
        base = gt or {"issues": [], "duration": at["duration"], "speech_start": at["speech_start"],
                      "speech_end": at["speech_end"], "trim_start": 0.0, "trim_end": at["duration"]}
        merged = dict(base)
        merged["trim_start"] = max(base["trim_start"], at["trim_start"])
        merged["trim_end"] = min(base["trim_end"], at["trim_end"])
        merged["issues"] = list(dict.fromkeys(base["issues"] + edge))
        if audit.get("words") and audit["words"].get("verified"):
            merged["note"] = "Checked by re-transcribing the trimmed audio: it now says exactly its script."
        merged["ok"] = False
        sug["trim"] = merged
        audit["trim"] = None
    elif gt and edge:
        gt["issues"] = list(dict.fromkeys(gt["issues"] + edge))
        gt["ok"] = False


def open_cleanup_dialog(char_name: str, on_done=lambda: None, staging: dict | None = None) -> None:
    """Spell-check review of a character's own reference clips.

    staging: set when reviewing freshly imported slices held in a hidden staging
    character (modules/importer.py) — {"target": character they will join,
    "is_new": bool, "on_commit": fn(result), optional "noun": "slices"/"clips"}. The checks
    (all of them, noises included) start immediately, the
    transcript is pre-filled from Whisper, and the end of the review offers ADD /
    DISCARD instead of just closing.

    Every clip is also checked for noises and breaths (modules/padding_audit.py), each fix
    verified not to change the words. Generated padding clips (source "buffer") get
    more on the same screen — words that aren't in their script, duplicates, background
    noise — and are checked against their script instead of having it "corrected" by
    Whisper."""
    clips: dict[str, K.Clip] = {c.key: c for c in K.own_clips(char_name)}    # key -> current Clip
    if not clips:
        ui.notify("No reference clips of its own to clean up", type="warning")
        return
    has_padding = any(c.source == "buffer" for c in clips.values())
    noun = (staging or {}).get("noun", "slices")
    all_checks = _PASSES + ("artifacts",)
    by_clip: dict[str, dict[str, dict]] = {}       # clip key -> {pass: suggestion}
    queue: list[str] = []                          # clip keys in review order
    pos = [0]
    stats = {"accepted": 0, "ignored": 0, "deleted": 0}
    state = {"analyzing": False}
    prog = {"text": "", "frac": 0.0}
    cancel = [False]
    _PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    include_clean = None
    scope = None

    # ── layout ───────────────────────────────────────────────────────────
    with ui.dialog().props("persistent").classes("w-full") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full max-w-3xl gap-3 p-5"
    ):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(
                f"{'NEW PADDING' if noun == 'clips' else 'IMPORT'}  ·  {len(clips)} {noun} for {staging['target']}"
                if staging else f"CLEAN UP DATA  ·  {char_name}"
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
            ) + ((
                ("NOISES", "Breaths, hiss and stray noises (found from the signal, then confirmed by Whisper "
                           "hearing no words in them). At the edges → trimmed off; between words → muted, so the "
                           "timing doesn't change. Every fix is re-checked: it's dropped if the clip wouldn't say "
                           "exactly the same thing afterwards. Natural breaths in your own recordings start "
                           "unticked."),
            ) + ((
                ("PADDING", "Generated padding clips are also checked against their script: extra or wrong "
                            "words, duplicates, and background noise that's too high."),
            ) if has_padding else ())):
                with ui.row().classes("gap-2 items-baseline flex-nowrap"):
                    ui.label(t).classes("text-amber-500 font-mono text-[10px] tracking-widest w-24 shrink-0")
                    ui.label(d).classes("text-[#666] font-mono text-[10px] leading-snug")
            n_padding = sum(1 for c in clips.values() if c.source == "buffer")
            if has_padding and n_padding < len(clips) and not staging:
                ui.label("CHECK").classes("text-[#444] font-mono text-[10px] tracking-widest mt-2")
                scope = ui.toggle(
                    {"both": f"Both  ({len(clips)})",
                     "original": f"Original clips  ({len(clips) - n_padding})",
                     "padding": f"Padding clips  ({n_padding})"},
                    value="both",
                ).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
                ui.label("Original = your recordings, imports and the primary reference. Padding = the "
                         "generated clips.").classes("text-[#555] font-mono text-[10px]")
            else:
                scope = None
            include_clean = None if staging else ui.checkbox(
                "Also step through clips with no issues", value=False,
            ).props("dense color=amber").classes("font-mono text-[11px] text-[#888] mt-1").tooltip(
                "Normally only clips with a problem come up. Tick this to see every checked clip — "
                "sections with nothing wrong are shown in green."
            )
            with ui.row().classes("gap-2 mt-2 items-center"):
                ui.button(
                    "CHECK EVERYTHING", icon="spellcheck", on_click=lambda: _start(all_checks),
                ).props("color=amber unelevated").classes("font-mono tracking-widest").tooltip(
                    "Run every check, then review each clip that needs attention"
                )
                for p in all_checks:
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
                f"{len(clips)} {noun} are staged"
                + (f", {missing} without a transcript (they can't be used for training until given one)" if missing else "")
                + ". Ones you haven't reviewed are added as they are."
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

    def _preview(key: str, gain: float, t0: float, t1: float | None, mute: tuple = ()):
        out = _PREVIEW_DIR / f"p_{int(time.time() * 1000) % 10**9}.wav"
        try:
            K.render_preview(clips[key].audio, out, gain, t0, t1, mute)
        except Exception as exc:
            show_error(exc)
            return
        play_url(_url(out))

    # ── one clip, every characteristic ───────────────────────────────────
    def _show_clip(key: str):
        sug = by_clip[key]
        clip = clips[key]
        use: dict = {}                # section -> its tick box
        ctl: dict = {}                # control widgets by name
        audit = sug.get("artifacts")
        generated = clip.source == "buffer"
        mute_boxes: list[tuple] = []  # (checkbox, (start, end)) per noise between words
        GOOD = "text-[#6fbf8b]"

        def header(title: str, ok: bool, status: str, tickable: bool = True, ticked: bool | None = None):
            """Section header: tick box (the fix is applied if ticked) + 'no issue' or the problem."""
            with ui.row().classes("w-full items-center gap-2"):
                if tickable:
                    cb = ui.checkbox(title, value=(not ok) if ticked is None else ticked).props(
                        "dense color=green" if ok else "dense color=amber"
                    ).classes("font-mono text-[11px] text-[#bbb]")
                else:
                    cb = None
                    ui.label(title).classes("font-mono text-[11px] text-[#bbb]")
                ui.label("✓ no issue" if ok else status).classes(
                    f"{GOOD if ok else 'text-[#c9a227]'} font-mono text-[10px] flex-grow"
                )
            return cb

        def tick_on_edit(widget, name: str):
            widget.on_value_change(lambda e: use[name].set_value(True))

        with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
            btn = ui.button(icon="play_arrow").props("round flat color=amber")
            bind_play(btn, url=_url(clip.audio))
            with ui.column().classes("gap-0 min-w-0 flex-grow"):
                ui.label(clip.audio.stem + ("   · padding clip" if generated else "")).classes(
                    "text-white font-mono text-sm [overflow-wrap:anywhere]"
                )
                head_text = _headline(sug, clip, bool(staging))
                ui.label(head_text).classes(
                    ("text-[#6fbf8b]" if head_text == "no issues found" else "text-[#e0b13a]") + " font-mono text-[12px]"
                )
        if audit and audit["recommend_delete"]:
            ui.label(f"Recommended: delete this clip — {audit['reason']}.").classes("text-red-400 font-mono text-[11px]")

        # VOLUME ------------------------------------------------------------
        if "volume" in sug:
            r = sug["volume"]
            ok = not r["issues"]
            with ui.column().classes(_SECTION_OK if ok else _SECTION):
                use["volume"] = header("VOLUME", ok, "; ".join(r["issues"]))
                ui.label(f"Speech level {r['level_db']:.0f} dBFS (typical for this set {r['target_db']:.0f}) · "
                         f"peak {r['peak']:.2f}" + (f"  —  {r['note']}" if r.get("note") else "")).classes(
                    "text-[#666] font-mono text-[10px]")
                ctl["gain"] = ui.number(value=r["gain_db"], format="%.1f", step=0.5, min=-24, max=24, label="change").props(
                    "dense outlined dark suffix=dB").classes("w-32 font-mono")
                tick_on_edit(ctl["gain"], "volume")

        # TRIM: waveform + region ---------------------------------------------
        if "trim" in sug:
            r = sug["trim"]
            ok = not r["issues"]
            with ui.column().classes(_SECTION_OK if ok else _SECTION):
                use["trim"] = header("TRIM", ok, "; ".join(r["issues"]))
                lead, tail = r["speech_start"], r["duration"] - r["speech_end"]
                ui.label(f"{r['duration']:.1f} s long · {lead:.1f} s before the speech, {tail:.1f} s after").classes(
                    "text-[#666] font-mono text-[10px]")
                if r.get("caution"):
                    ui.label("⚠ " + r["caution"]).classes("text-amber-500 font-mono text-[10px]")
                if r.get("note"):
                    ui.label(r["note"]).classes("text-[#7a9a7a] font-mono text-[10px]")
                marks = [(e.start, e.end) for e in (audit["events"] if audit else [])]
                ctl["region"] = region_picker(
                    clip.audio, r["trim_start"], r["trim_end"], marks=marks,
                    on_change=lambda a, b: use["trim"].set_value(True),
                )
                if marks:
                    ui.label("red = a noise that was found").classes("text-[#a55] font-mono text-[9px]")

        # NOISES ------------------------------------------------------------
        if audit is not None:
            if audit["mute"]:
                with ui.column().classes(_SECTION):
                    header("NOISES BETWEEN WORDS", False, "ticked ones are muted — the timing doesn't change", tickable=False)
                    if clip.source != "buffer":
                        ui.label("Natural breaths in a recording start unticked.").classes("text-[#666] font-mono text-[10px]")
                    for e in (e for e in audit["events"] if e.where == "inside"):
                        natural = not generated and e.kind == "breath"        # a person breathing is normal
                        cb = ui.checkbox(f"{e.kind}  {e.start:.2f} – {e.end:.2f} s  ({e.level_db:+.0f} dB vs speech)",
                                         value=not natural)
                        cb.props("dense color=amber").classes("font-mono text-[11px] text-[#bbb]")
                        mute_boxes.append((cb, (e.start, e.end)))
            else:
                with ui.column().classes(_SECTION_OK):
                    header("NOISES", True, "", tickable=False)
                    ui.label("No breaths, hiss or stray noises between the words."
                             + (f"  ({audit['skipped']} possible one{'s' if audit['skipped'] != 1 else ''} left alone: "
                                "removing it would change the words.)" if audit.get("skipped") else "")).classes(
                        "text-[#666] font-mono text-[10px]")

            # generated clips: the script, and duplicates / background
            if generated and audit.get("words"):
                w = audit["words"]
                if w["kind"] == "mismatch":
                    with ui.column().classes(_SECTION):
                        use["transcript"] = header("USE WHAT WAS SAID", False, "the audio doesn't match its script", ticked=False)
                        ui.label("SCRIPT").classes(_LBL)
                        ui.label(clip.transcript).classes("text-[#999] font-mono text-[11px] [overflow-wrap:anywhere]")
                        ui.label("HEARD").classes(_LBL)
                        ui.html("&nbsp;".join(
                            f'<span style="color:{"#aaa" if kind == "same" else "#f59e0b"}">{html.escape(wd)}</span>'
                            for wd, kind in K.diff_words(clip.transcript, w["heard"])
                        ), sanitize=False).classes("font-mono text-[11px] [overflow-wrap:anywhere]")
                        ctl["text"] = ui.textarea(value=w["heard"]).props("outlined dense dark autogrow").classes(
                            "w-full font-mono text-[12px]")
                        ui.label("Tick to keep the clip with a transcript that matches what is really said; "
                                 "otherwise delete it.").classes("text-[#666] font-mono text-[10px]")
                elif w["kind"] == "ok":
                    with ui.column().classes(_SECTION_OK):
                        header("SCRIPT", True, "", tickable=False)
                        ui.label("Says exactly its script.").classes("text-[#666] font-mono text-[10px]")
            if generated:
                other = [i for i in audit["issues"] if i.startswith(("duplicate", "noisy"))]
                with ui.column().classes(_SECTION if other else _SECTION_OK):
                    header("DUPLICATES & BACKGROUND", not other, "; ".join(other), tickable=False)
                    if not other:
                        ui.label(f"Not a repeat of another script · speech is {audit['snr_db']:.0f} dB above the background.").classes(
                            "text-[#666] font-mono text-[10px]")

        # TRANSCRIPT (recordings and imports; a generated clip's text is its script) ---------
        if "transcript" in sug:
            r = sug["transcript"]
            ok = not r["issues"]
            suggestion = r.get("suggested", "")
            with ui.column().classes(_SECTION_OK if ok else _SECTION):
                use["transcript"] = header("TRANSCRIPT", ok, "; ".join(r["issues"]))
                if ok:
                    ui.label(f"Matches what Whisper hears ({r['wer']:.0%} of words differ).").classes(
                        "text-[#666] font-mono text-[10px]")
                    ctl["text"] = ui.textarea(value=clip.transcript).props("outlined dense dark autogrow").classes(
                        "w-full font-mono text-[12px]")
                    tick_on_edit(ctl["text"], "transcript")
                else:
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
                        "w-full font-mono text-[12px]")
                if K.has_filler(clip.transcript):
                    ui.label("Your text has “um”/“uh” that Whisper leaves out — keep them if they're real.").classes(
                        "text-[#666] font-mono text-[10px]")

        def audio_args() -> tuple[float, float, float | None, tuple]:
            gain = (ctl["gain"].value or 0.0) if "volume" in use and use["volume"].value else 0.0
            mute = tuple(rng for cb, rng in mute_boxes if cb.value)
            if "trim" in use and use["trim"].value:
                reg = ctl["region"]
                if reg.start > 0.005 or reg.end < reg.duration - 0.005:      # the whole clip is not a trim
                    return gain, reg.start, reg.end, mute
            return gain, 0.0, None, mute

        def accept():
            gain, t0, t1, mute = audio_args()
            text = (ctl["text"].value or "").strip() if "transcript" in use and use["transcript"].value else None
            if "transcript" in use and use["transcript"].value and not text:
                ui.notify("The transcript is empty — type one, or untick TRANSCRIPT", type="warning")
                return
            did = False
            cur = clips[key]
            try:
                if gain or t0 > 0 or t1 is not None or mute:
                    new = K.apply_audio_fix(char_name, cur, gain_db=gain, trim_start=t0, trim_end=t1, mute=mute)
                    if new.key != key:                       # mp3 -> wav changes the key
                        clips[new.key] = clips.pop(key)
                    clips[new.key] = cur = new
                    did = True
                if text is not None and text != cur.transcript:
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
            if "volume" in sug or "trim" in sug or audit is not None:
                ui.button(
                    "PREVIEW", icon="volume_up",
                    on_click=lambda: _preview(key, *audio_args()),
                ).props("flat dense color=grey").classes("font-mono text-[10px]").tooltip(
                    "Hear the clip with the ticked volume, trim and noise fixes applied"
                )
            ui.space()
            if key != K.PRIMARY:
                ui.button("DELETE CLIP", icon="delete_outline", on_click=lambda: _delete(key)).props(
                    "color=negative unelevated" if audit and audit["recommend_delete"] else "flat color=negative"
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
            idx = generic.index(p)
            prog["frac"] = share * (idx + i / max(n, 1)) / len(generic)
            prog["text"] = f"{_TITLES[p].lower()} check · {i + 1}/{n} · {name}" + (
                "  (first transcript check loads Whisper — can take ~30 s)" if p == "transcript" and i == 0 else ""
            )

        timer = ui.timer(0.3, lambda: (progress_lbl.set_text(prog["text"]), progress_bar.set_value(prog["frac"])))
        want = scope.value if scope is not None else "both"
        keys = {k for k, c in clips.items()
                if want == "both" or (c.source == "buffer") == (want == "padding")}
        generic = tuple(p for p in passes if p != "artifacts")
        do_audit = "artifacts" in passes
        share = 0.5 if (generic and do_audit) else 1.0           # progress share of the generic checks

        def on_audit(i, n, name):
            prog["frac"] = (1 - share) + share * i / max(n, 1) if generic else i / max(n, 1)
            prog["text"] = f"padding clips: noises and script · {i + 1}/{n} · {name}" + (
                "  (loads Whisper — can take ~30 s)" if i == 0 and "transcript" not in generic else "")

        def run_checks():
            out = K.analyze_all(char_name, generic, None, on_progress, cancel, keys=keys) if generic else {}
            if do_audit and not cancel[0]:
                out["artifacts"] = PA.audit_character(char_name, on_audit, cancel, keys=keys)
            return out


        try:
            found = await ni_run.io_bound(run_checks)
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
                if k in clips:
                    by_clip.setdefault(k, {})[p] = sug
        for sug in by_clip.values():
            _fold_audit_into_trim(sug)
        queue[:] = [k for k in clips if k in by_clip
                    and (include_clean is not None and include_clean.value or _needs_attention(by_clip[k]))]   # file order
        pos[0] = 0
        progress_box.set_visibility(False)
        review_box.set_visibility(True)
        _show()

    dlg.open()
    if staging:
        intro_box.set_visibility(False)
        ui.timer(0.2, lambda: _start(all_checks), once=True)
