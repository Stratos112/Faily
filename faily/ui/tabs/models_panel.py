"""MODELS tab: every Piper model trained for a character, with how it was made,
how it scores, and what you can do next — play it, make it the voice, continue
training it from where it stopped, start over from the base voice with its
settings, or delete it. Registry in faily/core/piper_models.py.
"""
import time
from pathlib import Path

from nicegui import run as ni_run, ui

from faily.core import piper_models as PM
from faily.core.settings import get_piper_learning_rate, get_piper_match, get_piper_use_padding
from faily.modules.piper import (
    DEFAULT_MATCH, MATCH_ASPECTS, MATCH_CHOICES, infer,
)
from faily.modules.pronunciation import cached_score, score_model
from faily.ui.components import play_url, show_error

_LBL = "text-[#444] font-mono text-[10px] tracking-widest"
_VAL = "text-[#aaa] font-mono text-[11px] [overflow-wrap:anywhere]"
_TRY_DIR = Path("outputs/_model_try")
_DEFAULT_TRY = "Hello, this is a test of the trained voice."
_migrated: set[str] = set()      # characters already checked for a pre-registry model this session


def _fmt_date(iso: str | None) -> str:
    return (iso or "")[:16].replace("T", "  ")


def _fmt_secs(s) -> str:
    if not s:
        return "—"
    s = int(s)
    h, m = s // 3600, (s % 3600) // 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _lr_text(lr) -> str:
    return f"{lr:.2g}" if lr else "the checkpoint's own"


def build_models_panel(name: str, on_change, rebuild, train) -> None:
    """Render the MODELS tab for `name`.
    on_change(): tell the app the character's voice changed.
    rebuild(): rebuild the whole character page (after training starts etc.).
    train(opts): async — start a training run (see characters_tab._do_train)."""
    with ui.column().classes("w-full gap-2"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.label("TRAINED MODELS").classes(_LBL + " flex-grow")
            count_lbl = ui.label("").classes("text-[#333] font-mono text-[10px]")
        try_input = ui.input(label="Try it — text to speak", value=_DEFAULT_TRY).props(
            "outlined dark dense"
        ).classes("w-full font-mono text-[11px]")
        list_col = ui.column().classes("w-full gap-2")

    # ── actions ──────────────────────────────────────────────────────────
    async def play(m: dict):
        text = (try_input.value or "").strip() or _DEFAULT_TRY
        out = _TRY_DIR / f"{name}_{m['id']}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            await ni_run.io_bound(infer, text, m["onnx"], out)
        except Exception as exc:
            show_error(exc)
            return
        play_url(f"/outputs/{out.relative_to(Path('outputs')).as_posix()}?t={int(time.time() * 1000)}")

    def use(m: dict):
        PM.set_active(name, m["id"])
        ui.notify(f"“{m['name']}” is now {name}'s voice", type="positive", timeout=3000)
        on_change()
        refresh_sync()

    async def score(m: dict, btn):
        btn.disable()
        btn.set_text("scoring…")
        try:
            await ni_run.io_bound(score_model, m["onnx"])
        except Exception as exc:
            show_error(exc)
        finally:
            try:
                from faily.modules.asr import unload
                unload()
            except Exception:
                pass
        refresh_sync()

    def confirm_delete(m: dict):
        with ui.dialog() as d, ui.card().classes("bg-[#1a1a1a] border border-[#333] gap-3"):
            ui.label(f"DELETE “{m['name']}”?").classes("text-white font-mono text-xs tracking-widest")
            ui.label(
                "Removes this model and its saved checkpoint. This can't be undone."
                + (" It's the current voice, so the newest remaining model will take over." if m["active"] else "")
            ).classes("text-[#888] font-mono text-[11px] max-w-md")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=d.close).props("flat dense color=grey")

                def go():
                    PM.delete_model(name, m["id"])
                    d.close()
                    on_change()
                    refresh_sync()
                ui.button("DELETE", icon="delete", on_click=go).props("color=negative unelevated dense")
        d.open()

    def details(m: dict):
        sc = cached_score(m["onnx"])
        with ui.dialog().classes("w-full") as d, ui.card().classes(
            "bg-[#0d0d0d] border border-[#252525] w-full max-w-2xl gap-2 p-5"
        ):
            ui.label("MODEL DETAILS").classes("text-white font-mono text-xs tracking-widest")
            name_in = ui.input(label="Name", value=m["name"]).props("outlined dark dense").classes("w-full font-mono")

            def row(label, value):
                with ui.row().classes("w-full gap-3 items-baseline flex-nowrap"):
                    ui.label(label).classes(_LBL + " w-44 shrink-0")
                    ui.label(str(value)).classes(_VAL)

            clips = m.get("clips") or {}
            loss = m.get("loss") or {}
            match = m.get("match")
            row("CREATED", _fmt_date(m.get("created")))
            row("BASE VOICE", (m.get("base_voice_label") or "unknown")
                + (f"  (base epoch {m['base_epoch']})" if m.get("base_epoch") is not None else "")
                + ("  — inferred" if "base_voice" in (m.get("inferred") or []) else ""))
            row("FINAL EPOCH", m.get("epoch", "—"))
            row("EPOCHS TRAINED", (f"{m['epochs_trained']} this run" if m.get("epochs_trained") is not None else "—")
                + (f", {m['epochs_since_base']} since the base voice" if m.get("epochs_since_base") is not None
                   and m.get("parent") else ""))
            row("LEARNING RATE", _lr_text(m.get("learning_rate")) if not m.get("imported") else "not recorded")
            if match:
                row("MATCHED TO", "   ".join(
                    f"{MATCH_ASPECTS[a]['label']}: {MATCH_CHOICES.get(match.get(a), match.get(a))}"
                    for a in MATCH_ASPECTS))
                row("FROZEN PARTS", m.get("frozen") or "none")
            if clips:
                row("TRAINING CLIPS", f"{clips.get('n')} ({clips.get('n_real')} recordings + "
                    f"{clips.get('n_padding')} padding) · {clips.get('minutes')} min of audio")
            if m.get("batch"):
                row("BATCH", f"{m['batch']} (micro-batch × accumulation) · validation split {m.get('validation_split')}")
            row("LOSS", (f"generator {loss['gen']} · validation {loss['val']} (around epoch {loss['epoch']})"
                         if loss.get("gen") is not None else "not available"))
            row("PRONUNCIATION", (f"{sc['accuracy']:.1%} of words right · {sc['words_per_sec']} words/s"
                                  if sc else "not scored yet"))
            if sc and sc.get("worst"):
                row("WENT WRONG", " | ".join(f"“{w['heard'][:70]}”" for w in sc["worst"][:3]))
            row("TRAINING TIME", _fmt_secs(m.get("trained_seconds")))
            if m.get("parent"):
                p = PM.get_model(name, m["parent"])
                row("CONTINUED FROM", p["name"] if p else f"{m['parent']} (deleted)")
            row("CHECKPOINT", f"{m['ckpt_mb']} MB kept — can be continued" if m["ckpt"] else "not kept — can't be continued")
            if m.get("note"):
                ui.label(m["note"]).classes("text-[#666] font-mono text-[10px] leading-snug")
            with ui.row().classes("w-full justify-between mt-2"):
                with ui.row().classes("gap-2"):
                    if m["ckpt"]:
                        def free():
                            PM.delete_checkpoint(name, m["id"])
                            ui.notify("Checkpoint deleted", timeout=2500)
                            d.close()
                            refresh_sync()
                        ui.button("DELETE CHECKPOINT", icon="folder_delete", on_click=free).props(
                            "flat dense color=grey"
                        ).classes("font-mono text-[10px]").tooltip("Frees disk space; the model still works but can't be continued")
                with ui.row().classes("gap-2"):
                    ui.button("Close", on_click=d.close).props("flat dense color=grey")

                    def save_name():
                        new = (name_in.value or "").strip()
                        if new and new != m["name"]:
                            PM.update_meta(name, m["id"], name=new)
                        d.close()
                        refresh_sync()
                    ui.button("SAVE", icon="check", on_click=save_name).props("color=amber unelevated dense")
        d.open()

    def train_from(m: dict):
        can_continue = bool(m["ckpt"]) and bool(m.get("base_voice"))
        can_restart = bool(m.get("base_voice"))
        clips = m.get("clips") or {}
        match0 = {**DEFAULT_MATCH, **(m.get("match") or get_piper_match())}
        with ui.dialog().classes("w-full") as d, ui.card().classes(
            "bg-[#0d0d0d] border border-[#252525] w-full max-w-xl gap-3 p-5"
        ):
            ui.label(f"TRAIN FROM “{m['name']}”").classes("text-white font-mono text-xs tracking-widest")
            ui.label("START FROM").classes(_LBL)
            mode = ui.radio(
                {"continue": f"Continue — resume at epoch {m.get('epoch')} exactly where it stopped",
                 "restart": f"Start over — retrain from the {m.get('base_voice_label') or 'base'} base voice"},
                value="continue" if can_continue else "restart",
            ).props("dense color=amber")
            if not can_continue:
                ui.label(
                    "Can't continue: its checkpoint wasn't kept." if not m["ckpt"]
                    else "Can't continue: its base voice is unknown."
                ).classes("text-amber-500 font-mono text-[10px]")
                mode.set_options({"restart": f"Start over — retrain from the {m.get('base_voice_label') or 'base'} base voice"},
                                 value="restart")
            if not can_restart:
                ui.label("This model's base voice is unknown, so it can't be retrained from here.").classes(
                    "text-red-400 font-mono text-[10px]"
                )
            ui.label("WHEN IT FINISHES").classes(_LBL)
            save_as = ui.radio(
                {"new": "Keep this model and save the result as a new one",
                 "replace": f"Replace “{m['name']}” with the result"},
                value="new",
            ).props("dense color=amber")
            ui.label("SETTINGS (from this model — change if you like)").classes(_LBL)
            lr = ui.number(
                label="Learning rate", value=m.get("learning_rate") or get_piper_learning_rate(),
                format="%.2e", min=1e-7, max=1e-3, step=1e-5,
            ).props("outlined dark dense").classes("w-48 font-mono")
            toggles = {}
            for a, info in MATCH_ASPECTS.items():
                with ui.row().classes("items-center gap-2"):
                    ui.label(info["label"].upper()).classes("text-[#888] font-mono text-[10px] tracking-widest w-28")
                    toggles[a] = ui.toggle(MATCH_CHOICES, value=match0[a]).props(
                        "dense no-caps unelevated toggle-color=amber text-color=grey-6"
                    )
            pad = ui.switch(
                "Include generated padding clips",
                value=(clips.get("n_padding", 0) > 0) if clips else get_piper_use_padding(),
            ).props("dense color=amber").classes("font-mono text-[10px] text-[#888]")
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=d.close).props("flat dense color=grey")

                async def start():
                    if not can_restart:
                        return
                    opts = {
                        "base_voice": m["base_voice"],
                        "learning_rate": lr.value or None,
                        "match": {a: t.value for a, t in toggles.items()},
                        "use_padding": pad.value,
                        "resume_model": m if mode.value == "continue" else None,
                        "replace_id": m["id"] if save_as.value == "replace" else None,
                    }
                    d.close()
                    await train(opts)
                ui.button("START TRAINING", icon="model_training", on_click=start).props(
                    "color=indigo unelevated dense"
                ).classes("font-mono tracking-widest")
        d.open()

    # ── rendering ────────────────────────────────────────────────────────
    def card(m: dict):
        sc = cached_score(m["onnx"])
        loss = m.get("loss") or {}
        with ui.card().classes(
            "w-full gap-1 p-3 bg-[#111] border " + ("border-amber-500/50" if m["active"] else "border-[#222]")
        ):
            with ui.row().classes("w-full items-center gap-2 flex-nowrap"):
                if m["active"]:
                    ui.badge("VOICE", color="amber").props("outline").classes("font-mono text-[9px]")
                ui.label(m["name"]).classes("text-white font-mono text-sm flex-grow [overflow-wrap:anywhere]")
                ui.label(_fmt_date(m.get("created"))).classes("text-[#555] font-mono text-[10px] shrink-0")
            with ui.row().classes("w-full items-center gap-x-4 gap-y-0"):
                ui.label(f"{m.get('base_voice_label') or 'unknown base'} · epoch {m.get('epoch', '—')}").classes(
                    "text-[#999] font-mono text-[10px]"
                )
                if sc:
                    acc = sc["accuracy"]
                    ui.label(f"pronunciation {acc:.0%}").classes(
                        "font-mono text-[10px] " + ("text-green-400" if acc >= 0.95 else
                                                    "text-amber-400" if acc >= 0.85 else "text-red-400")
                    ).tooltip(f"{sc['words_per_sec']} words/s on 16 unseen sentences")
                else:
                    sbtn = ui.button("pronunciation: not scored — SCORE", on_click=lambda m=m: score(m, sbtn)).props(
                        "flat dense no-caps color=grey"
                    ).classes("font-mono text-[10px]")
                if loss.get("gen") is not None:
                    ui.label(f"loss {loss['gen']} / {loss['val']}").classes("text-[#777] font-mono text-[10px]").tooltip(
                        "generator / validation loss near the final epoch (lower is better)"
                    )
                ui.label("checkpoint kept" if m["ckpt"] else "no checkpoint").classes(
                    "font-mono text-[10px] " + ("text-[#555]" if m["ckpt"] else "text-[#a55]")
                )
            with ui.row().classes("w-full items-center gap-1 mt-1"):
                ui.button(icon="play_arrow", on_click=lambda m=m: play(m)).props("round dense color=amber").tooltip(
                    "Speak the “Try it” text with this model"
                )
                if not m["active"]:
                    ui.button("USE", icon="record_voice_over", on_click=lambda m=m: use(m)).props(
                        "flat dense color=amber"
                    ).classes("font-mono text-[10px]").tooltip("Make this the character's voice")
                ui.button("DETAILS", icon="info", on_click=lambda m=m: details(m)).props("flat dense color=grey").classes(
                    "font-mono text-[10px]"
                )
                ui.button("TRAIN FROM THIS…", icon="model_training", on_click=lambda m=m: train_from(m)).props(
                    "flat dense color=indigo"
                ).classes("font-mono text-[10px]")
                ui.space()
                ui.button(icon="delete_outline", on_click=lambda m=m: confirm_delete(m)).props(
                    "flat dense color=grey"
                ).tooltip("Delete this model")

    def refresh_sync():
        models = PM.list_models(name)
        count_lbl.set_text(f"{len(models)} model{'s' if len(models) != 1 else ''}")
        list_col.clear()
        with list_col:
            if not models:
                ui.label(
                    "No trained models yet. Train a voice on the TRAINING tab; every checkpoint you "
                    "save from the audition list shows up here."
                ).classes("text-[#555] font-mono text-[11px] leading-snug py-2")
            for m in models:
                card(m)

    async def first_load():
        if name not in _migrated:
            _migrated.add(name)
            with list_col:
                ui.label("Checking for an existing model…").classes("text-[#555] font-mono text-[11px]")
            try:
                await ni_run.io_bound(PM.ensure_migrated, name)
            except Exception as exc:
                show_error(exc)
        refresh_sync()

    ui.timer(0.05, first_load, once=True)
