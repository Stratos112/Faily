"""Buffer Training wizard — a 4-step dialog (Timbral Direction, Zero-Shot
Backend, Generate, Review & Commit) that bulk-generates zero-shot clips of a
character's own voice to pad out their reference pool before Piper training.
"""
import shutil
from pathlib import Path
from typing import Callable

from nicegui import ui

from faily.core.characters import get_ref_chain, add_ref_clip
from faily.core.model_manager import manager
from faily.modules.vc import BACKENDS
from faily.modules.buffer_training import (
    CATEGORIES, ScriptLine, BufferCandidate,
    load_script_bank, sample_lines, generate_batch,
)
from faily.ui.components import section_label, show_error, model_picker

_STAGING_ROOT = Path("outputs/vc/_buffer_staging")


def _section_row(text: str, tip: str):
    with ui.row().classes("items-center gap-1"):
        section_label(text)
        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(tip)


def _fmt_dur(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def open_buffer_training_dialog(char_name: str, on_committed: Callable[[], None]) -> None:
    """Open the Buffer Training wizard for char_name. on_committed() fires
    after Step 4 commits at least one candidate into the reference pool."""
    chain = get_ref_chain(char_name)
    if not chain:
        ui.notify("No ref clips — add clips from CLONE or TUNE first", type="warning")
        return

    try:
        bank = load_script_bank()
    except FileNotFoundError as exc:
        show_error(exc)
        return

    staging_dir = _STAGING_ROOT / char_name
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    has_transcripts = any(c["transcript"].strip() for c in chain)

    # ── dialog state ─────────────────────────────────────────────────────
    _weights: dict[str, list[float]] = {k: [1.0] for k in CATEGORIES}
    _count: list[int] = [200]
    _backend: list[str] = ["chatterbox"]
    _param1: list[float] = [BACKENDS["chatterbox"]["param1"]["default"]]
    _param2: list[float] = [BACKENDS["chatterbox"]["param2"]["default"]]
    _lines: list[ScriptLine] = []
    _candidates: list[BufferCandidate] = []
    _cancel: list[bool] = [False]
    _started: list[bool] = [False]

    with ui.dialog().classes("w-full max-w-5xl") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full gap-3 p-5"
    ):
        ui.label(f"BUFFER TRAINING — {char_name}").classes(
            "text-white font-mono text-xs tracking-widest"
        )
        ui.label(
            "Bulk-generate zero-shot clips of this character's own voice, across a "
            "diverse script bank, to pad out the reference pool before training."
        ).classes("text-[#444] font-mono text-[10px] leading-snug")
        ui.separator().classes("opacity-20")

        def _ui_safe(fn):
            # Mirrors _do_train's dead-connection guard — a 200-clip batch can
            # run well past ten minutes; a closed browser tab shouldn't abort it.
            try:
                fn()
            except RuntimeError:
                pass

        with ui.stepper().props("flat").classes("w-full bg-transparent") as stepper:

            # ══ STEP 1 — TIMBRAL DIRECTION ══════════════════════════════
            with ui.step("Timbral Direction"):
                _section_row(
                    "TOTAL CLIPS",
                    "How many clips to generate this run. Generation time depends on "
                    "the backend — roughly 15-30+ minutes for 200 clips.",
                )
                with ui.row().classes("w-full items-center gap-3"):
                    count_lbl = ui.label("200").classes(
                        "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                    )
                    def _on_count(e):
                        _count[0] = int(e.value)
                        count_lbl.set_text(str(int(e.value)))
                    ui.slider(min=20, max=500, step=10, value=200, on_change=_on_count).classes(
                        "flex-grow"
                    ).props("color=amber")

                ui.separator().classes("my-1 opacity-10")
                ui.label("CATEGORY WEIGHTS").classes(
                    "text-[#444] font-mono text-[10px] tracking-widest"
                )
                ui.label(
                    "0 excludes a category entirely. Higher values draw more of that "
                    "category's lines relative to the others."
                ).classes("text-[#333] font-mono text-[9px] leading-snug mb-1")

                for key, info in CATEGORIES.items():
                    _section_row(info["label"], info["desc"])
                    with ui.row().classes("w-full items-center gap-3"):
                        w_lbl = ui.label("1.00").classes(
                            "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                        )
                        def _on_weight(e, k=key, lbl=w_lbl):
                            _weights[k][0] = float(e.value)
                            lbl.set_text(f"{e.value:.2f}")
                        ui.slider(
                            min=0.0, max=2.0, step=0.05, value=1.0, on_change=_on_weight,
                        ).classes("flex-grow").props("color=amber")

                def _go_step2():
                    if all(w[0] <= 0 for w in _weights.values()):
                        ui.notify("At least one category needs a weight above 0", type="warning")
                        return
                    stepper.next()

                with ui.stepper_navigation():
                    ui.button("NEXT", on_click=_go_step2).props("color=amber unelevated")

            # ══ STEP 2 — ZERO-SHOT BACKEND ══════════════════════════════
            with ui.step("Zero-Shot Backend"):
                _section_row(
                    "BACKEND",
                    "Which zero-shot model generates the clips, using this character's "
                    "own reference audio as the voice to clone.",
                )
                backend_opts = {k: dict(v) for k, v in BACKENDS.items()}
                if not has_transcripts:
                    backend_opts["f5_tts"]["available"] = False
                    backend_opts["f5_tts"]["unavailable_label"] = "no transcripts"
                    backend_opts["f5_tts"]["desc"] += (
                        "  [unavailable — this character has no transcribed reference "
                        "clips, which F5-TTS requires]"
                    )

                params_col = ui.column().classes("w-full gap-4 mt-2")

                def _rebuild_params():
                    params_col.clear()
                    cfg = BACKENDS[_backend[0]]
                    with params_col:
                        for p, pval in ((cfg["param1"], _param1), (cfg["param2"], _param2)):
                            pval[0] = p["default"]
                            _section_row(p["label"], p["tooltip"])
                            with ui.row().classes("w-full items-center gap-3"):
                                lbl = ui.label(f"{p['default']:.2f}" if p["step"] < 1 else str(int(p["default"]))).classes(
                                    "font-mono text-[10px] text-amber-400 w-10 shrink-0 text-right"
                                )
                                def _on_change(e, pval=pval, lbl=lbl, step=p["step"]):
                                    pval[0] = e.value
                                    lbl.set_text(f"{e.value:.2f}" if step < 1 else str(int(e.value)))
                                ui.slider(
                                    min=p["min"], max=p["max"], step=p["step"],
                                    value=p["default"], on_change=_on_change,
                                ).classes("flex-grow").props("color=amber")

                def _on_backend(key: str):
                    _backend[0] = key
                    _rebuild_params()

                model_picker(backend_opts, "chatterbox", _on_backend)
                _rebuild_params()

                with ui.stepper_navigation():
                    ui.button("BACK", on_click=stepper.previous).props("flat color=grey")
                    ui.button(
                        "GENERATE", on_click=lambda: _start_generation(),
                    ).props("color=amber unelevated")

            # ══ STEP 3 — GENERATE ═══════════════════════════════════════
            with ui.step("Generate"):
                gen_log = ui.log(max_lines=1000).classes(
                    "w-full font-mono text-[10px] rounded border border-[#1a1a1a]"
                ).style("height:260px; background:#050505")
                with ui.row().classes("w-full justify-between items-center mt-2"):
                    gen_status = ui.label("—").classes("text-[#555] font-mono text-[10px]")
                    stop_btn = ui.button("STOP").props("flat dense color=negative")
                gen_progress = ui.linear_progress(value=0.0).classes("w-full mt-1").props("color=amber")

                async def _start_generation():
                    if _started[0]:
                        return
                    _started[0] = True
                    stepper.next()

                    _lines[:] = sample_lines(
                        bank, {k: w[0] for k, w in _weights.items()}, _count[0],
                    )
                    if not _lines:
                        _ui_safe(lambda: gen_status.set_text("Nothing to generate — check category weights"))
                        return

                    def _stop():
                        _cancel[0] = True
                        stop_btn.set_text("STOPPING…")
                        stop_btn.disable()
                    stop_btn.on_click(_stop)

                    if manager.loaded:
                        _ui_safe(lambda: gen_log.push(f"Freeing GPU memory ({', '.join(manager.loaded)})…"))
                        manager.unload_all()

                    _ui_safe(lambda: gen_log.push(f"Generating {len(_lines)} clips with {BACKENDS[_backend[0]]['label']}…"))

                    def _on_progress(i, total, line):
                        _ui_safe(lambda: gen_progress.set_value(i / total))
                        _ui_safe(lambda: gen_status.set_text(f"{i}/{total} — {CATEGORIES[line.category]['label']}"))
                        _ui_safe(lambda: gen_log.push(f"[{i}/{total}] ({line.category}) {line.text[:60]}"))

                    def _on_error(i, line, exc):
                        _ui_safe(lambda: gen_log.push(f"  ✗ failed: {exc}"))

                    try:
                        results = await generate_batch(
                            char_name, _lines, _backend[0], _param1[0], _param2[0],
                            staging_dir, _cancel, _on_progress, _on_error,
                        )
                        _candidates[:] = results
                    except Exception as exc:
                        _ui_safe(lambda: show_error(exc))
                        _ui_safe(lambda: gen_status.set_text("error"))
                        return

                    if _cancel[0]:
                        _ui_safe(lambda: gen_status.set_text(
                            f"{len(_candidates)}/{len(_lines)} generated — stopped early, reviewing what we have"
                        ))
                    else:
                        _ui_safe(lambda: gen_status.set_text(f"✓  {len(_candidates)} clips generated"))
                    _ui_safe(_render_review)
                    _ui_safe(stepper.next)

            # ══ STEP 4 — REVIEW & COMMIT ════════════════════════════════
            with ui.step("Review & Commit"):
                review_header = ui.label("").classes("text-[#888] font-mono text-[10px] tracking-widest")
                review_player = ui.audio("").classes("w-full rounded mt-1")
                review_now_playing = ui.label("").classes("text-[#333] font-mono text-[9px]")
                review_col = ui.column().classes("w-full gap-1 mt-1")

                def _play(u, n):
                    review_player.set_source(u)
                    review_now_playing.set_text(f"▶  {n}")

                def _refresh_header():
                    selected = [c for c in _candidates if c.accepted]
                    total_dur = 0.0
                    import soundfile as sf
                    for c in selected:
                        try:
                            total_dur += sf.info(str(c.path)).duration
                        except Exception:
                            pass
                    review_header.set_text(
                        f"{len(selected)} / {len(_candidates)} selected  ·  {_fmt_dur(total_dur)}"
                    )

                def _render_review():
                    review_col.clear()
                    if not _candidates:
                        with review_col:
                            ui.label("No clips generated.").classes("text-[#444] font-mono text-[10px] py-2")
                        _refresh_header()
                        return
                    groups: dict[str, list[BufferCandidate]] = {}
                    for c in _candidates:
                        groups.setdefault(c.category, []).append(c)

                    with review_col:
                        with ui.scroll_area().classes("w-full").style("height: 360px"):
                            with ui.column().classes("w-full gap-1"):
                                for cat, items in groups.items():
                                    checkboxes: list[ui.checkbox] = []

                                    def _select_all(checkboxes=checkboxes):
                                        # set_value fires each checkbox's own on_value_change,
                                        # which updates .accepted and calls _refresh_header().
                                        for cb in checkboxes:
                                            cb.set_value(True)

                                    def _select_none(checkboxes=checkboxes):
                                        for cb in checkboxes:
                                            cb.set_value(False)

                                    with ui.row().classes("w-full items-center gap-2 mt-2"):
                                        ui.label(
                                            f"{CATEGORIES[cat]['label']}  ({len(items)})"
                                        ).classes("text-amber-500 font-mono text-[10px] tracking-widest flex-grow")
                                        ui.button("all", on_click=_select_all).props(
                                            "flat dense color=grey"
                                        ).classes("font-mono text-[9px]")
                                        ui.button("none", on_click=_select_none).props(
                                            "flat dense color=grey"
                                        ).classes("font-mono text-[9px]")

                                    for c in items:
                                        rel = c.path.relative_to(Path("outputs"))
                                        url = f"/outputs/{rel.as_posix()}"
                                        with ui.row().classes(
                                            "w-full items-center gap-1 px-2 py-1 rounded "
                                            "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
                                        ):
                                            cb = ui.checkbox(value=c.accepted)
                                            def _on_toggle(e, c=c):
                                                c.accepted = bool(e.value)
                                                _refresh_header()
                                            cb.on_value_change(_on_toggle)
                                            checkboxes.append(cb)
                                            ui.button(
                                                icon="play_arrow",
                                                on_click=lambda u=url, n=c.path.stem: _play(u, n),
                                            ).props("flat dense color=amber").classes("shrink-0")
                                            if c.issues:
                                                ui.icon("warning", size="12px").classes(
                                                    "text-amber-500 shrink-0"
                                                ).tooltip(", ".join(c.issues))
                                            with ui.column().classes("gap-0 flex-grow min-w-0"):
                                                ui.label(c.text).classes(
                                                    "text-[#aaa] font-mono text-[10px] truncate"
                                                )
                    _refresh_header()

                async def _commit():
                    commit_btn.disable()
                    ok, failed = 0, 0
                    for c in _candidates:
                        if not c.accepted:
                            continue
                        try:
                            add_ref_clip(
                                char_name, c.path, transcript=" ".join(c.text.split()),
                                source="buffer", category=c.category,
                            )
                            ok += 1
                        except Exception:
                            failed += 1
                    if ok:
                        msg = f"{ok} clip{'s' if ok != 1 else ''} added to the reference pool"
                        if failed:
                            msg += f"  ({failed} failed)"
                        ui.notify(msg, type="warning" if failed else "positive", timeout=4000)
                        on_committed()
                    else:
                        ui.notify("Nothing selected to commit" if not failed else f"All {failed} commits failed", type="warning")
                    dlg.close()

                with ui.stepper_navigation():
                    commit_btn = ui.button("COMMIT SELECTED", on_click=_commit).props(
                        "color=amber unelevated"
                    )
                    ui.button("DISCARD & CLOSE", on_click=dlg.close).props("flat color=grey")

        def _on_hide():
            _cancel[0] = True
            shutil.rmtree(staging_dir, ignore_errors=True)
        dlg.on("hide", _on_hide)

    dlg.open()
