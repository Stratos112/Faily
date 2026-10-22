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
    BASE_VOICES, _write_status, base_voice_ready, download_base_voice,
    infer, list_candidates, promote_candidate, read_loss_history,
    read_training_status, request_stop, train, training_in_progress,
)
from faily.ui.components import show_error

_DEFAULT_PREVIEW = "Hello, this is a test of the trained voice."
_STALL_SECS = 300

# name -> {"log": [...], "proc": [Popen|None], "task": Task|None}. In-memory,
# so it lasts as long as the server process; the status file outlives it.
_SESSIONS: dict[str, dict] = {}


def _smooth(xs: list, window: int = 10) -> list:
    """Trailing mean, skipping gaps (None)."""
    out = []
    for i in range(len(xs)):
        vals = [v for v in xs[max(0, i - window + 1):i + 1] if v is not None]
        out.append(sum(vals) / len(vals) if vals else None)
    return out


def _loss_chart_options(hist: dict) -> dict:
    ep = hist["epochs"]

    def series(name, data, color, smooth):
        pts = [[e, None if v is None else round(v, 3)] for e, v in zip(ep, data)]
        return {
            "name": name, "type": "line", "data": pts, "showSymbol": False,
            "connectNulls": True, "smooth": False,
            "lineStyle": {"color": color, "width": 2 if smooth else 1, "opacity": 1 if smooth else 0.25},
            "itemStyle": {"color": color},
            "silent": not smooth,
        }

    return {
        "backgroundColor": "transparent",
        "textStyle": {"color": "#888", "fontFamily": "monospace"},
        "animation": False,
        "grid": {"left": 44, "right": 16, "top": 28, "bottom": 28},
        "legend": {"data": ["generator", "validation"], "textStyle": {"color": "#888"}, "top": 0},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "value", "scale": True, "name": "epoch", "nameLocation": "middle",
                  "nameGap": 22, "splitLine": {"show": False}},
        "yAxis": {"type": "value", "scale": True, "splitLine": {"lineStyle": {"color": "#1a1a1a"}}},
        "series": [
            series("generator", hist["gen"], "#f59e0b", False),
            series("validation", hist["val"], "#38bdf8", False),
            # same names so the legend toggles raw + smoothed together
            {**series("generator", _smooth(hist["gen"]), "#f59e0b", True), "name": "generator"},
            {**series("validation", _smooth(hist["val"]), "#38bdf8", True), "name": "validation"},
        ],
    }


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
    log = sess["log"].append
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
    cursor = [0]
    state = {"picker": False, "previewing": False, "chart_at": 0.0, "charting": False}

    with ui.dialog().props("persistent").classes("w-full max-w-3xl") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full gap-3 p-5"
    ):
        ui.label(f"TRAINING  {name}").classes("text-white font-mono text-xs tracking-widest")
        status_lbl = ui.label("…").classes("text-[#888] font-mono text-[11px]")
        bar = ui.linear_progress(value=0, show_value=False).props("color=amber track-color=grey-10")
        detail_lbl = ui.label("").classes("text-[#555] font-mono text-[10px]")
        warn_lbl = ui.label("").classes("text-amber-500 font-mono text-[10px]")
        with ui.column().classes("w-full gap-1") as chart_box:
            ui.label("LOSS — lower is better; faint = per epoch, bold = 10-epoch average").classes(
                "text-[#444] font-mono text-[10px] tracking-widest"
            )
            loss_chart = ui.echart({"series": []}).classes("w-full").style("height:220px")
            trend_lbl = ui.label("").classes("text-[#888] font-mono text-[10px]")
        chart_box.set_visibility(False)
        log = ui.log(max_lines=2000).classes(
            "w-full font-mono text-[10px] rounded border border-[#1a1a1a]"
        ).style("height:300px; background:#050505")
        if sess is None:
            log.push("(No live log: this run started before the server last restarted "
                     "or was never started from here. Progress above is still tracked.)")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button(
                "Copy log", icon="content_copy",
                on_click=lambda: ui.clipboard.write("\n".join(sess["log"]) if sess else ""),
            ).props("flat dense color=amber")
            stop_btn = ui.button("STOP EARLY", on_click=lambda: _stop()).props(
                "flat dense color=negative"
            ).tooltip(
                "Finish the current epoch, save a checkpoint of it, and move on to "
                "auditioning. Press again to kill immediately."
            )
            ui.button("HIDE" , icon="visibility_off", on_click=dlg.close).props(
                "flat dense color=grey"
            ).tooltip("Training keeps running — reopen from the character page")

        ui.separator().classes("opacity-20")
        ui.label(
            "PREVIEW LINE — used to audition each saved checkpoint once training "
            "finishes. Edit it any time before then."
        ).classes("text-[#444] font-mono text-[10px] leading-snug")
        preview_input = (
            ui.input(value=_DEFAULT_PREVIEW).props("outlined dark dense")
            .classes("w-full font-mono text-[11px]")
        )
        with ui.column().classes("w-full gap-2") as picker_section:
            ui.separator().classes("opacity-20")
            ui.label("PICK A CHECKPOINT").classes("text-amber-400 font-mono text-[10px] tracking-widest")
            ui.label(
                "Each saved checkpoint is a full, independent snapshot — the last one "
                "isn't always the best-sounding. Listen, then SAVE & USE your favourite, or "
                "SAVE several to compare later in the MODELS tab. Unsaved ones are discarded "
                "when you train again."
            ).classes("text-[#444] font-mono text-[10px] leading-snug")
            candidates_col = ui.column().classes("w-full gap-1")
        picker_section.set_visibility(False)

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
        """Pronunciation score per candidate (see modules/pronunciation.py)."""
        from faily.modules.pronunciation import score_model
        best = None
        for epoch, onnx, lbl in rows:
            _set_status(f"Scoring pronunciation — epoch {epoch}…", "text-[#888]")
            try:
                r = await ni_run.io_bound(score_model, onnx)
            except Exception as exc:
                lbl.set_text("score unavailable")
                lbl.tooltip(str(exc)[-300:])
                continue
            if r is None:        # run.io_bound returns None when cancelled / shutting down
                return
            acc = r["accuracy"]
            lbl.set_text(f"pronunciation {acc:.0%} · {r['words_per_sec']:.1f} words/s")
            lbl.classes(replace="font-mono text-[10px] shrink-0 w-52 " + (
                "text-green-400" if acc >= 0.95 else "text-amber-400" if acc >= 0.85 else "text-red-400"))
            tip = "Share of words a speech recogniser gets right on 16 sentences the model never trained on (base voice ≈ 99%)."
            if r["worst"]:
                tip += "  Examples that went wrong: " + " | ".join(f"“{w['heard'][:60]}”" for w in r["worst"][:3])
            lbl.tooltip(tip)
            if best is None or acc > best[0]:
                best = (acc, lbl)
        if best and len(rows) > 1:
            best[1].set_text(best[1].text + "  ★ best")
        try:
            from faily.modules.asr import unload
            unload()
        except Exception:
            pass

    async def _previews():
        state["previewing"] = True
        score_rows: list[tuple[int, Path, ui.label]] = []
        try:
            text = preview_input.value.strip() or _DEFAULT_PREVIEW
            for epoch, onnx in list_candidates(char_dir):
                wav = onnx.with_name(f"preview_epoch_{epoch}.wav")
                if not wav.exists():
                    status_lbl.set_text(f"Generating preview — epoch {epoch}…")
                    try:
                        await ni_run.io_bound(infer, text, onnx, wav)
                    except Exception as exc:
                        show_error(exc)
                        continue
                rel = wav.relative_to(Path("outputs"))
                with candidates_col:
                    with ui.row().classes(
                        "w-full items-center gap-2 px-3 py-1 rounded "
                        "border border-[#1e1e1e] hover:border-[#333]"
                    ):
                        ui.label(f"epoch {epoch}").classes("text-[#888] font-mono text-[10px] w-20 shrink-0")
                        score_lbl = ui.label("scoring…").classes("text-[#555] font-mono text-[10px] shrink-0 w-52")
                        score_rows.append((epoch, onnx, score_lbl))
                        ui.audio(f"/outputs/{rel.as_posix()}").classes("flex-grow").props("controls")
                        btns = []
                        btns.append(ui.button(
                            "SAVE & USE", icon="check", on_click=lambda p=onnx, b=btns: _save(p, True, b),
                        ).props("flat dense color=amber").classes("font-mono text-[10px] shrink-0"))
                        btns.append(ui.button(
                            "SAVE", icon="save", on_click=lambda p=onnx, b=btns: _save(p, False, b),
                        ).props("flat dense color=grey").classes("font-mono text-[10px] shrink-0").tooltip(
                            "Keep this as a model without making it the character's voice"
                        ))
            state["picker"] = True
            await _score_all(score_rows)
        finally:
            state["previewing"] = False

    def _stop():
        st = read_training_status(char_dir) or {}
        proc = sess["proc"][0] if sess else None
        if st.get("stopping") and proc:
            proc.terminate()   # second press: don't wait for the epoch
            sess["log"].append("Killed — checkpoints since the last periodic save are lost.")
        elif st.get("state") == "training" and request_stop(char_dir):
            ui.notify("Stopping after this epoch — a checkpoint of it will be saved", timeout=5000)
        elif proc:
            proc.terminate()
            sess["log"].append(
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
            hist = await ni_run.io_bound(read_loss_history, char_dir)
            state["chart_at"] = time.time()
            if hist["epochs"]:
                loss_chart.options.clear()
                loss_chart.options.update(_loss_chart_options(hist))
                loss_chart.update()
                trend_lbl.set_text(hist["trend"])
                chart_box.set_visibility(True)
        except Exception:
            pass   # the graph is a convenience; never break the dialog over it
        finally:
            state["charting"] = False

    def _set_status(text: str, color: str):
        status_lbl.set_text(text)
        status_lbl.classes(replace=f"font-mono text-[11px] {color}")

    async def _tick():
        if sess:
            for line in sess["log"][cursor[0]:]:
                log.push(line)
            cursor[0] = len(sess["log"])
        st = read_training_status(char_dir) or {}
        s = st.get("state", "none")
        active = s in ("preparing", "training", "exporting")
        stop_btn.set_visibility(s == "training" or (active and bool(sess)))
        stop_btn.set_text("KILL NOW" if st.get("stopping") else "STOP EARLY")
        if s in ("training", "exporting", "ready", "error"):
            await _refresh_chart()
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
            picker_section.set_visibility(True)
            if not state["picker"] and not state["previewing"]:
                candidates_col.clear()
                await _previews()
            if state["picker"]:
                _set_status(f"✓ {st.get('candidates', 0)} checkpoint(s) ready — pick one below", "text-green-400")
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
