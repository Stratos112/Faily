"""ADD CHARACTER (button under the CHARACTERS list): build a character from recordings,
audition it zero-shot, and only then save it.

The clips go into a hidden draft character (importer.draft_name(); "_stage_new_*", so no
list shows it and an abandoned one is purged with the stages). Each upload runs through
split & qualify aimed at the draft; TRY generates with the draft's reference chain, the
same way SPEAK → ZERO SHOT does; SAVE renames the draft (importer.promote_draft), CANCEL
deletes it."""
import uuid

import soundfile as sf
from nicegui import run as ni_run, ui

from faily.core.characters import build_ref_audio, get_character, get_ref_chain, ref_seconds
from faily.core.paths import TMP_DIR
from faily.modules import importer
from faily.modules.vc import BACKENDS, generate as vc_generate
from faily.ui.components import bind_play, model_picker, output_url, overlay_host, section_label, show_error
from faily.ui.tabs.import_dialog import open_import_dialog

_BTN = "font-mono tracking-widest"
_HINT = "text-[#555] font-mono text-[10px] leading-snug"
_UPLOADS = TMP_DIR / "new_character"


def open_new_character_dialog(host=None, on_saved=lambda name: None) -> None:
    """host: fallback parent for the dialogs when the app's overlay host isn't set.
    on_saved(name): called after the character has been saved."""
    host = overlay_host() or host
    draft = importer.draft_name()
    backend = ["xtts_v2"]
    progress: list[float] = [0.0]

    def _discard():
        importer.discard_stage(draft)
        dlg.close()

    def _refresh_clips():
        clip_list.clear()
        chain = get_ref_chain(draft) if get_character(draft) else []
        total = 0.0
        with clip_list:
            if not chain:
                ui.label("No clips yet — upload a recording above.").classes(_HINT + " py-1")
            for c in chain:
                try:
                    dur = sf.info(str(c["audio"])).duration
                except Exception:
                    dur = 0.0
                total += dur
                with ui.row().classes("w-full items-center gap-2 px-2 py-0.5 rounded border border-[#1e1e1e] no-wrap"):
                    btn = ui.button(icon="play_arrow").props("flat dense round size=sm color=amber")
                    bind_play(btn, url=output_url(c["audio"]))
                    ui.label(f"{dur:4.1f} s").classes("font-mono text-[10px] text-[#666] shrink-0")
                    ui.label(c.get("transcript") or "(no transcript)").classes(
                        "font-mono text-[10px] text-[#888] truncate min-w-0 flex-grow")
        clip_count.set_text(f"{len(chain)} clip{'s' if len(chain) != 1 else ''} · {total:.0f} s" if chain else "")
        ready = bool(chain)
        for el in (try_btn, save_btn):
            el.enable() if ready else el.disable()

    async def _on_upload(e):
        _UPLOADS.mkdir(parents=True, exist_ok=True)
        dest = _UPLOADS / f"{uuid.uuid4().hex[:6]}_{e.file.name}"
        dest.write_bytes(await e.file.read())
        upload.reset()
        open_import_dialog(dest, "", host, on_added=_refresh_clips, fixed_target=draft,
                           target_label="the new character", delete_source=True)

    def _on_backend(key: str):
        backend[0] = key
        text_hint.set_text(BACKENDS[key].get("text_hint", ""))

    async def _try():
        text = (text_in.value or "").strip()
        if not text:
            ui.notify("Type a sentence to hear first", type="warning")
            return
        cfg = BACKENDS[backend[0]]
        try_btn.disable(); save_btn.disable()
        progress[0] = 0.0
        bar.set_value(0.0); bar.set_visibility(True)
        status.set_text("loading the model…")
        poll.active = True
        try:
            with build_ref_audio(draft, require_transcript=backend[0] == "f5_tts",
                                 max_duration=ref_seconds(backend[0])) as (ref, transcript):
                path = await ni_run.io_bound(
                    vc_generate, text, ref, progress, None, backend[0],
                    cfg["param1"]["default"], cfg["param2"]["default"], transcript,
                )
            player.set_source(output_url(path))
            player.set_visibility(True)
            player.play()
            status.set_text(f"{cfg['label']} — like it? Name it below and save, or add more clips first.")
        except Exception as exc:
            status.set_text("")
            show_error(exc)
        finally:
            poll.active = False
            bar.set_visibility(False)
            try_btn.enable(); save_btn.enable()

    def _tick():
        if progress[0] > 0:
            status.set_text("generating…")
            bar.set_value(progress[0])

    def _save():
        try:
            name = importer.promote_draft(draft, name_in.value)
        except Exception as exc:
            ui.notify(str(exc), type="warning", timeout=5000)
            return
        dlg.close()
        ui.notify(f"Saved character {name}", type="positive", timeout=3000)
        on_saved(name)

    with host, ui.dialog().props("persistent") as dlg, ui.card().classes(
        "bg-[#0d0d0d] border border-[#252525] w-full max-w-2xl gap-3 p-5"
    ):
        ui.label("NEW CHARACTER").classes("text-white font-mono text-xs tracking-widest")

        section_label("1 · RECORDINGS")
        ui.label(
            "Upload a recording of the voice. It is split at its pauses and every slice is checked "
            "(volume, silence, transcript) before it counts. Repeat with more recordings to give the "
            "voice more to go on — nothing is saved until you press SAVE."
        ).classes(_HINT)
        upload = (
            ui.upload(on_upload=_on_upload, auto_upload=True)
            .props("accept=.wav,.mp3,.flac,.ogg flat dense color=grey label='Upload a recording'")
            .classes("w-full")
        )
        clip_count = ui.label("").classes("text-amber-400 font-mono text-[10px]")
        clip_list = ui.column().classes("w-full gap-1 max-h-48 overflow-y-auto")

        section_label("2 · TRY THE VOICE")
        model_picker(BACKENDS, backend[0], _on_backend)
        text_hint = ui.label(BACKENDS[backend[0]].get("text_hint", "")).classes(_HINT)
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            text_in = ui.input(placeholder="Type a sentence…", value="Hi! This is what my new voice sounds like.") \
                .props("outlined dark dense").classes("flex-grow")
            try_btn = ui.button("TRY", icon="play_arrow", on_click=_try).props("color=grey unelevated dense") \
                .classes(_BTN)
        bar = ui.linear_progress(value=0, show_value=False).props("color=amber").classes("w-full")
        bar.set_visibility(False)
        status = ui.label("").classes("text-[#888] font-mono text-[10px]")
        player = ui.audio("").classes("w-full")
        player.set_visibility(False)

        section_label("3 · SAVE")
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            name_in = ui.input(placeholder="character name").props("outlined dark dense").classes("flex-grow")
            ui.button("CANCEL", on_click=_discard).props("flat dense color=grey").classes(_BTN)
            save_btn = ui.button("SAVE CHARACTER", icon="check", on_click=_save) \
                .props("color=amber unelevated dense").classes(_BTN)
        name_in.on("keydown.enter", lambda: _save())
        poll = ui.timer(0.15, _tick, active=False)

    dlg.on("hide", lambda: dlg.delete())
    _refresh_clips()
    dlg.open()
