"""Piper training dialog + status chip.

Training runs as a background task owned by the server, not by the browser
tab that started it. The dialog is a view onto that run: it polls the status
file written by piper.train() (plus the in-memory log of this server process),
so it can be hidden, reopened, or reopened from a fresh page load and still
show where things stand. See faily/modules/piper.py (read_training_status).
"""
import time
from pathlib import Path

from nicegui import background_tasks, run as ni_run, ui

from faily.core.characters import CHARACTERS_DIR
from faily.core.model_manager import manager
from faily.modules.piper import (
    BASE_VOICES, _write_status, append_log, base_voice_ready, download_base_voice,
    infer, list_candidates, promote_candidate, read_training_status, request_stop,
    start_log, train, training_history, training_in_progress, training_log_path,
)
from faily.modules.pronunciation import cached_score
from faily.ui.components import output_url, show_error
from faily.ui.training_charts import history_chart

_DEFAULT_PREVIEW = "Hello, this is a test of the trained voice."
_STALL_SECS = 300

# name -> {"log": [...], "proc": [Popen|None], "task": Task|None}. In-memory,
# so it lasts as long as the server process; the status file and the log file
# (piper.training_log_path, what the dialog shows) outlive it.
_SESSIONS: dict[str, dict] = {}


def _fmt_eta(secs: float | None) -> str:
    if secs is None:
        return "estimating…"
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m = rem // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


async def _run_session(
    name: str, chain: list[dict], base_voice: str, learning_rate: float | None,
    match: dict | None, resume_model: dict | None, replace_id: str | None, sess: dict,
    max_epochs: int = 1000,
) -> None:
    char_dir = CHARACTERS_DIR / name

    def log(line: str):
        sess["log"].append(line)
        append_log(char_dir, line)

    try:
        # piper_train is a separate OS process; any TTS/VC model still resident
        # here just competes with it for the same physical VRAM.
        if manager.loaded:
            log(f"Freeing GPU memory ({', '.join(manager.loaded)})…")
            manager.unload_all()
        if not base_voice_ready(base_voice):
            log(f"Downloading base voice: {BASE_VOICES.get(base_voice, {}).get('label', base_voice)}…")
            await download_base_voice(base_voice, log)
        await train(
            chain, char_dir, log, sess["proc"], base_voice=base_voice, learning_rate=learning_rate,
            match=match, resume_model=resume_model, replace_id=replace_id,
            max_epochs=max_epochs,
        )
    except Exception as exc:
        log(f"ERROR: {exc}")
        st = read_training_status(char_dir) or {}
        if st.get("state") != "error":   # failed before train() could record it
            _write_status(char_dir, state="error", message=str(exc)[-2000:])
    finally:
        sess["task"] = None


def start_training(
    name: str, chain: list[dict], base_voice: str, learning_rate: float | None = None,
    match: dict | None = None, resume_model: dict | None = None, replace_id: str | None = None,
    max_epochs: int = 1000,
) -> bool:
    """Kick off a background training run. False if one is already active."""
    busy = training_in_progress(CHARACTERS_DIR)
    if busy:
        ui.notify(
            f"Already training '{busy}' — wait for it to finish (or stop it) first",
            type="warning",
        )
        return False
    sess = {"log": [], "proc": [None], "task": None}
    _SESSIONS[name] = sess
    start_log(CHARACTERS_DIR / name)
    _write_status(
        CHARACTERS_DIR / name, state="preparing", started=time.time(), message="", stopping=False,
        epoch=None, target=None, start_epoch=None, fraction=0.0, eta=None,
        detail="", candidates=0,
    )
    sess["task"] = background_tasks.create(
        _run_session(name, chain, base_voice, learning_rate, match, resume_model, replace_id, sess, max_epochs)
    )
    return True


def open_training_dialog(name: str, on_change, on_done=lambda: None) -> None:
    """Open (or re-open) the training dialog for a character. Works whether
    the run was started by this browser tab, another one, or before a reload."""
    char_dir = CHARACTERS_DIR / name
    sess = _SESSIONS.get(name)
    state = {"picker": False, "previewing": False, "chart_at": 0.0, "charting": False,
             "shown": set(), "score_rows": [], "scored": set(), "save_btns": [], "log_pos": 0, "log_note": False}
    first = (read_training_status(char_dir) or {}).get("state")
    start_tab = ("CHECKPOINTS" if first in ("ready", "done") else
                 "LOG" if first in ("preparing", "error", "interrupted", None) else "PROGRESS")

    with ui.dialog().props("persistent").classes("w-full max-w-4xl") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full gap-3 p-5"
    ):
        ui.label(f"TRAINING  {name}").classes("text-white font-mono text-xs tracking-widest")
        status_lbl = ui.label("…").classes("text-[#888] font-mono text-[11px]")
        bar = ui.linear_progress(value=0, show_value=False).props("color=amber track-color=grey-10")
        detail_lbl = ui.label("").classes("text-[#555] font-mono text-[10px]")
        warn_lbl = ui.label("").classes("text-amber-500 font-mono text-[10px]")
        with ui.tabs().props("dense no-caps align=left active-color=amber indicator-color=amber").classes(
            "w-full text-[#888] font-mono border-b border-[#1a1a1a]"
        ) as tabs:
            ui.tab("PROGRESS")
            ui.tab("CHECKPOINTS")
            ui.tab("LOG")
        with ui.tab_panels(tabs, value=start_tab).props("animated=false").classes("w-full bg-transparent"):
            with ui.tab_panel("PROGRESS").classes("p-0 gap-1"):
                chart_wait = ui.label("Graphs appear once the first epoch is logged.").classes(
                    "text-[#555] font-mono text-[10px]")
                with ui.column().classes("w-full gap-1") as chart_box:
                    update_chart = history_chart()
                    trend_lbl = ui.label("").classes("text-[#888] font-mono text-[10px]")
                chart_box.set_visibility(False)
            with ui.tab_panel("CHECKPOINTS").classes("p-0 gap-2"):
                ui.label(
                    "Each saved checkpoint is a full, independent snapshot — the last one "
                    "isn't always the best-sounding. Listen, then SAVE & USE your favourite, or "
                    "SAVE several to compare later in the MODELS tab. Unsaved ones are discarded "
                    "when you train again. They show up here while training runs (every 5 epochs) "
                    "and their pronunciation is scored as they come in; saving unlocks when the run "
                    "ends (or is stopped)."
                ).classes("text-[#444] font-mono text-[10px] leading-snug")
                ui.label(
                    "PREVIEW LINE — spoken by each checkpoint as it comes in. Edit it any time; "
                    "checkpoints already previewed keep their take."
                ).classes("text-[#444] font-mono text-[10px] leading-snug")
                preview_input = (
                    ui.input(value=_DEFAULT_PREVIEW).props("outlined dark dense")
                    .classes("w-full font-mono text-[11px]")
                )
                picker_note = ui.label("No checkpoints yet.").classes("text-[#666] font-mono text-[10px]")
                candidates_col = ui.column().classes("w-full gap-1")
            with ui.tab_panel("LOG").classes("p-0 gap-1"):
                log = ui.log(max_lines=5000).classes(
                    "w-full font-mono text-[10px] rounded border border-[#1a1a1a]"
                ).style("height:420px; background:#050505")
                ui.button(
                    "Copy log", icon="content_copy",
                    on_click=lambda: ui.clipboard.write(_read_log()),
                ).props("flat dense color=amber").classes("self-end")
        with ui.row().classes("w-full justify-end gap-2"):
            stop_btn = ui.button("STOP EARLY", on_click=lambda: _stop()).props(
                "flat dense color=negative"
            ).tooltip(
                "Finish the current epoch, save a checkpoint of it, and move on to "
                "auditioning. Press again to kill immediately."
            )
            ui.button("HIDE" , icon="visibility_off", on_click=dlg.close).props(
                "flat dense color=grey"
            ).tooltip("Training keeps running — reopen from the character page")

    def _read_log() -> str:
        try:
            return training_log_path(char_dir).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "\n".join(sess["log"]) if sess else ""

    def _tail_log():
        """Push the log file's new complete lines (it is rewritten when a new run starts)."""
        path = training_log_path(char_dir)
        try:
            size = path.stat().st_size
        except OSError:
            if not state["log_note"]:
                state["log_note"] = True
                log.push("(No log saved for this run — older runs kept it only in memory.)")
            return
        if size < state["log_pos"]:
            log.clear()
            state["log_pos"] = 0
        if size == state["log_pos"]:
            return
        with path.open("rb") as f:
            f.seek(state["log_pos"])
            chunk = f.read(size - state["log_pos"])
        complete = chunk[:chunk.rfind(b"\n") + 1]         # a line still being written waits for the next tick
        state["log_pos"] += len(complete)
        for line in complete.decode("utf-8", errors="replace").splitlines():
            log.push(line)

    def _show_score(lbl: ui.label, r: dict, best: bool = False):
        acc = r["accuracy"]
        lbl.set_text(f"pronunciation {acc:.0%} · {r['words_per_sec']:.1f} words/s" + ("  ★ best" if best else ""))
        lbl.classes(replace="font-mono text-[10px] shrink-0 w-52 " + (
            "text-green-400" if acc >= 0.95 else "text-amber-400" if acc >= 0.85 else "text-red-400"))
        tip = "Share of words a speech recogniser gets right on 16 sentences the model never trained on (base voice ≈ 99%)."
        if r["worst"]:
            tip += "  Examples that went wrong: " + " | ".join(f"“{w['heard'][:60]}”" for w in r["worst"][:3])
        lbl.tooltip(tip)

    def _live_scores():
        """Fill in scores the training run has cached since the rows were drawn."""
        for epoch, onnx, lbl in state["score_rows"]:
            if epoch not in state["scored"] and (r := cached_score(onnx)):
                state["scored"].add(epoch)
                _show_score(lbl, r)

    async def _save(onnx: Path, activate: bool, row_btns: list):
        """Save a candidate as a model (kept with its checkpoint, see core/piper_models.py)."""
        try:
            saved = await ni_run.io_bound(promote_candidate, char_dir, onnx, activate)
        except Exception as exc:
            show_error(exc)
            return
        if saved is None:
            return
        ui.notify(
            f"Saved as “{saved['name']}”" + (" and set as the voice" if activate else ""),
            type="positive", timeout=4000,
        )
        for b in row_btns:
            b.disable()
        on_change()
        on_done()

    async def _score_all(rows: list[tuple[int, Path, ui.label]]):
        """Pronunciation score per candidate (see modules/pronunciation.py). The run scores
        them as they come in, so this is mostly cache hits; it fills any gap and marks the best."""
        from faily.modules.pronunciation import score_model
        scores = []
        for epoch, onnx, lbl in rows:
            r = cached_score(onnx)
            if r is None:
                _set_status(f"Scoring pronunciation — epoch {epoch}…", "text-[#888]")
                try:
                    r = await ni_run.io_bound(score_model, onnx)
                except Exception as exc:
                    lbl.set_text("score unavailable")
                    lbl.tooltip(str(exc)[-300:])
                    continue
                if r is None:        # run.io_bound returns None when cancelled / shutting down
                    return
            state["scored"].add(epoch)
            _show_score(lbl, r)
            scores.append((r["accuracy"], lbl, r))
        if len(scores) > 1:
            acc, lbl, r = max(scores, key=lambda s: s[0])
            _show_score(lbl, r, best=True)
        try:
            from faily.modules.asr import unload
            unload()
        except Exception:
            pass

    async def _previews(final: bool):
        """Add a row for every candidate not shown yet. While training runs (final=False) the
        rows can only be listened to, and their pronunciation shows up as the run scores it
        (_live_scores); once it has ended, SAVE unlocks and any unscored row is scored."""
        state["previewing"] = True
        try:
            text = preview_input.value.strip() or _DEFAULT_PREVIEW
            for epoch, onnx in list_candidates(char_dir):
                if epoch in state["shown"]:
                    continue
                wav = onnx.with_name(f"preview_epoch_{epoch}.wav")
                if not wav.exists():
                    picker_note.set_text(f"Generating preview — epoch {epoch}…")
                    try:
                        await ni_run.io_bound(infer, text, onnx, wav)
                    except Exception as exc:
                        show_error(exc)
                        continue
                    finally:
                        picker_note.set_text("")
                if not state["shown"]:
                    picker_note.set_text("")          # "No checkpoints yet."
                state["shown"].add(epoch)
                with candidates_col:
                    with ui.row().classes(
                        "w-full items-center gap-2 px-3 py-1 rounded "
                        "border border-[#1e1e1e] hover:border-[#333]"
                    ):
                        ui.label(f"epoch {epoch}").classes("text-[#888] font-mono text-[10px] w-20 shrink-0")
                        score_lbl = ui.label("scoring…").classes(
                            "text-[#555] font-mono text-[10px] shrink-0 w-52")
                        state["score_rows"].append((epoch, onnx, score_lbl))
                        ui.audio(output_url(wav)).classes("flex-grow").props("controls")
                        btns = []
                        btns.append(ui.button(
                            "SAVE & USE", icon="check", on_click=lambda p=onnx, b=btns: _save(p, True, b),
                        ).props("flat dense color=amber").classes("font-mono text-[10px] shrink-0"))
                        btns.append(ui.button(
                            "SAVE", icon="save", on_click=lambda p=onnx, b=btns: _save(p, False, b),
                        ).props("flat dense color=grey").classes("font-mono text-[10px] shrink-0").tooltip(
                            "Keep this as a model without making it the character's voice"
                        ))
                        state["save_btns"] += btns
                        if not final:
                            for b in btns:
                                b.disable()
            if final:
                for b in state["save_btns"]:
                    b.enable()
                state["picker"] = True
                await _score_all(state["score_rows"])
        finally:
            state["previewing"] = False

    def _note(line: str):
        if sess:
            sess["log"].append(line)
        append_log(char_dir, line)

    def _stop():
        st = read_training_status(char_dir) or {}
        proc = sess["proc"][0] if sess else None
        if st.get("stopping") and proc:
            proc.terminate()   # second press: don't wait for the epoch
            _note("Killed — checkpoints since the last periodic save are lost.")
        elif st.get("state") == "training" and request_stop(char_dir):
            ui.notify("Stopping after this epoch — a checkpoint of it will be saved", timeout=5000)
        elif proc:
            proc.terminate()
            _note(
                "Stopped immediately (this trainer can't stop gracefully) — work since the "
                "last periodic checkpoint is lost."
            )
        else:
            ui.notify("Can't stop this run from here — it predates stop support", type="warning")

    async def _refresh_chart():
        if time.time() - state["chart_at"] < 15 or state["charting"]:
            return
        state["charting"] = True
        try:
            hist = await ni_run.io_bound(training_history, char_dir)
            state["chart_at"] = time.time()
            if hist["loss"]["epochs"] or hist["checkpoints"]:
                update_chart(hist)
                trend_lbl.set_text(hist["loss"]["trend"])
                chart_box.set_visibility(True)
                chart_wait.set_visibility(False)
        except Exception:
            pass   # the graph is a convenience; never break the dialog over it
        finally:
            state["charting"] = False

    def _set_status(text: str, color: str):
        status_lbl.set_text(text)
        status_lbl.classes(replace=f"font-mono text-[11px] {color}")

    async def _tick():
        _tail_log()
        _live_scores()
        st = read_training_status(char_dir) or {}
        s = st.get("state", "none")
        active = s in ("preparing", "training", "exporting")
        stop_btn.set_visibility(s == "training" or (active and bool(sess)))
        stop_btn.set_text("KILL NOW" if st.get("stopping") else "STOP EARLY")
        if s in ("training", "exporting", "ready", "error"):
            await _refresh_chart()
        if s in ("training", "exporting") and st.get("candidates") and not state["previewing"]:
            state["previewing"] = True            # claimed now: the task starts on the next loop turn
            background_tasks.create(_previews(final=False))
        warn_lbl.set_text("")
        if s == "training":
            ep, tgt = st.get("epoch"), st.get("target")
            frac = st.get("fraction") or 0.0
            bar.set_value(frac)
            _set_status(
                ("stopping after this epoch — " if st.get("stopping") else "training — ")
                + f"epoch {ep} / {tgt}  ({frac * 100:.1f}%)",
                "text-amber-400",
            )
            detail_lbl.set_text(
                f"batch {st.get('batch', 0)}/{st.get('batches', '?')}  ·  "
                f"{st.get('detail', '')}  ·  ETA {_fmt_eta(st.get('eta'))}"
            )
            idle = time.time() - st.get("updated", time.time())
            if idle > _STALL_SECS:
                warn_lbl.set_text(
                    f"⚠ no progress for {idle / 60:.0f} min — possible GPU stall. "
                    "STOP keeps the checkpoints saved so far."
                )
        elif s == "preparing":
            bar.props("indeterminate")
            _set_status("preparing dataset…", "text-[#888]")
        elif s == "exporting":
            _set_status(st.get("message") or "exporting checkpoints…", "text-[#888]")
        elif s == "ready":
            bar.set_value(1.0)
            bar.props(remove="indeterminate")
            detail_lbl.set_text("")
            if not state["picker"] and not state["previewing"]:
                state["previewing"] = True        # claimed now: the task starts on the next loop turn
                background_tasks.create(_previews(final=True))
            if state["picker"]:
                _set_status(f"✓ {st.get('candidates', 0)} checkpoint(s) ready — pick one on CHECKPOINTS", "text-green-400")
        elif s == "done":
            _set_status("✓ finished — model installed", "text-green-400")
        elif s == "error":
            _set_status(f"error — {st.get('message', '')[-300:]}", "text-red-400")
        elif s == "interrupted":
            _set_status(
                "interrupted — the server restarted mid-run. Checkpoints saved so far are in "
                f"outputs/characters/{name}/piper_train; resume with scripts/piper_resume_supervised.py",
                "text-red-400",
            )
        else:
            _set_status("no training recorded", "text-[#555]")
        if s != "preparing":
            bar.props(remove="indeterminate")

    _tail_log()             # the log so far, now rather than on the first tick
    ui.timer(1.0, _tick)
    dlg.open()


def _chip_look(st: dict | None) -> tuple[str, str] | None:
    """(label, quasar color) for a status worth surfacing, else None."""
    s = (st or {}).get("state")
    if s == "training":
        pct = (st.get("fraction") or 0) * 100
        return f"TRAINING  epoch {st.get('epoch')}/{st.get('target')}  ({pct:.0f}%)", "amber"
    if s in ("preparing", "exporting"):
        return f"TRAINING  {s}…", "amber"
    if s == "ready":
        return "TRAINING FINISHED — PICK A CHECKPOINT", "positive"
    if s == "error":
        return "TRAINING FAILED — VIEW", "negative"
    if s == "interrupted":
        return "TRAINING INTERRUPTED — VIEW", "negative"
    return None


def training_status_chip(name: str, on_change, on_done=lambda: None) -> None:
    """A clickable one-line status for one character's header; hidden when idle."""
    char_dir = CHARACTERS_DIR / name
    btn = ui.button(
        on_click=lambda: open_training_dialog(name, on_change, on_done)
    ).props("flat dense no-caps").classes("font-mono text-[10px]")

    def _refresh():
        look = _chip_look(read_training_status(char_dir))
        if not look:
            btn.set_visibility(False)
            return
        btn.set_text(f"{look[0]}  — OPEN")
        btn.props(f"color={look[1]}")
        btn.set_visibility(True)

    _refresh()
    ui.timer(3.0, _refresh)


def global_training_chip(on_change, on_done=lambda: None) -> None:
    """App-header indicator, visible from every tab. Shows the most relevant
    run across all characters: running > finished-awaiting-pick > failed or
    interrupted within the last day (older failures are left to the
    character's own chip)."""
    target = [None]
    btn = ui.button(
        on_click=lambda: target[0] and open_training_dialog(target[0], on_change, on_done)
    ).props("flat dense no-caps icon=model_training").classes("font-mono text-[10px]")
    rank = {"training": 0, "preparing": 0, "exporting": 0, "ready": 1, "interrupted": 2, "error": 2}

    def _refresh():
        best = None
        if CHARACTERS_DIR.is_dir():
            for d in CHARACTERS_DIR.iterdir():
                st = read_training_status(d)
                s = (st or {}).get("state")
                if s not in rank:
                    continue
                if rank[s] == 2 and time.time() - st.get("updated", 0) > 86400:
                    continue
                if best is None or rank[s] < rank[best[1]["state"]]:
                    best = (d.name, st)
        look = _chip_look(best[1]) if best else None
        if not look:
            target[0] = None
            btn.set_visibility(False)
            return
        target[0] = best[0]
        btn.set_text(f"{best[0]}: {look[0]}")
        btn.props(f"color={look[1]}")
        btn.set_visibility(True)

    _refresh()
    ui.timer(3.0, _refresh)
