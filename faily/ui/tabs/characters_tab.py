import shutil
from pathlib import Path
from nicegui import ui, run as ni_run
from faily.core.characters import (
    list_characters, get_character, get_ref_path, delete_character,
    update_character_metadata,
    list_character_clips, list_character_favorites, rename_character_file,
    get_ref_chain, remove_ref_clip, set_rvc_model, set_piper_model,
    rename_ref_audio, update_ref_clip, CHARACTERS_DIR,
)
from faily.ui.components import (
    SPLIT_COLUMNS,
    section_label, show_error, send_to_edit, bind_play, prime_audio, play_url,
)
from faily.modules.piper import (
    BASE_VOICES, MATCH_ASPECTS, MATCH_CHOICES, training_clips, frozen_modules,
    base_voice_ready, generate_base_voice_sample, SAMPLE_TEXT,
)
from faily.core.model_manager import manager
from faily.core.settings import (
    get_default_base_voice, set_default_base_voice,
    get_piper_learning_rate, set_piper_learning_rate, get_piper_epochs, set_piper_epochs,
    get_piper_match, set_piper_match,
)
from faily.ui.tabs.about_panel import build_about_panel
from faily.ui.tabs.refs_hub import build_refs_hub
from faily.ui.tabs.training_data_panel import build_activation_panel
from faily.ui.tabs.qualify_dialog import open_qualify_dialog
from faily.ui.tabs.import_dialog import open_import_dialog
from faily.ui.tabs.models_panel import build_models_panel
from faily.ui.tabs.piper_training_dialog import (
    open_training_dialog, start_training, training_status_chip,
)
from faily.modules.piper import read_training_status, _ACTIVE_STATES

_BTN = "font-mono tracking-widest"
# wrap long names/transcripts onto the next line instead of widening the row
_WRAP = "break-words [overflow-wrap:anywhere]"


def build_characters_tab(on_speak, on_change, initial: str | None = None):
    """
    on_speak(name): navigate to SPEAK tab with this character pre-selected
    on_change():    notify app that characters were deleted (refresh other tabs)
    initial:        character to have open when the tab is first built (default: none)
    Returns: refresh() callable
    """
    _selected: list[str | None] = [initial]
    # _rebuild_detail() recreates every element, so remember the open sub-tab
    # across rebuilds (the reference list's scroll position is kept in the
    # browser; see below).
    _detail_tab = ["ABOUT"]

    def _grouped() -> tuple[list[dict], dict[str, list[dict]]]:
        all_chars = list_characters()
        by_name = {c["name"]: c for c in all_chars}
        roots = [c for c in all_chars if "parent" not in c or c["parent"] not in by_name]
        children: dict[str, list[dict]] = {}
        for c in all_chars:
            p = c.get("parent")
            if p and p in by_name:
                children.setdefault(p, []).append(c)
        return roots, children

    def _item_row(name: str, depth: int):
        is_active = _selected[0] == name
        indent = f"pl-{depth * 6}" if depth else ""
        active_cls = (
            "border-amber-500/60 text-amber-400 bg-[#1a1500]"
            if is_active
            else "border-[#1e1e1e] text-[#888] hover:border-[#333] hover:bg-[#111]"
        )
        with ui.row().classes(
            f"w-full items-center gap-2 px-3 py-1.5 rounded cursor-pointer border {active_cls} {indent}"
        ) as row:
            icon = "person_outline" if depth else "person"
            icon_cls = "text-[#555]" if depth else "text-amber-500"
            ui.icon(icon, size="14px").classes(f"shrink-0 {icon_cls}")
            prefix = "↳ " * depth
            ui.label(prefix + name).classes("font-mono text-[10px] truncate flex-grow")
            row.on("click", lambda n=name: _select(n))

    def _rebuild_list():
        left_col.clear()
        roots, children = _grouped()

        def _walk(name: str, depth: int):
            _item_row(name, depth=depth)
            for child in children.get(name, []):
                _walk(child["name"], depth + 1)

        with left_col:
            if not roots and not children:
                ui.label("No characters yet — save one from the CLONE tab.").classes(
                    "text-[#444] font-mono text-[10px] px-1 py-4"
                )
                return
            for root in roots:
                _walk(root["name"], depth=0)

    def _select(name: str):
        _selected[0] = name
        _rebuild_list()
        _rebuild_detail()

    def _open_rename_dialog(char_name: str, subfolder: str, clip_path: Path):
        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#333] min-w-[380px] gap-3"
        ):
            ui.label("RENAME CLIP").classes("text-white font-mono text-xs tracking-widest")
            inp = (
                ui.input(value=clip_path.stem)
                .props("outlined dark dense")
                .classes("w-full font-mono")
            )
            ui.label(".wav").classes("text-[#444] font-mono text-[10px]")

            def _do_rename():
                new_stem = inp.value.strip()
                if not new_stem:
                    return
                try:
                    rename_character_file(char_name, subfolder, clip_path.name, new_stem)
                    dlg.close()
                    ui.notify(f"Renamed to {new_stem}.wav", type="positive", timeout=2000)
                    _rebuild_detail()
                except Exception as exc:
                    show_error(exc)

            inp.on("keydown.enter", lambda: _do_rename())
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("Rename", on_click=_do_rename).props("color=amber unelevated dense")
        dlg.open()

    def _split_ref_clip(char_name: str, ref: dict):
        """Run a reference clip through the split & qualify wizard (slice at pauses, then check by
        hand / automatically / automatically with the flagged reviewed). The slices are added to
        this character; the long original can be removed afterwards (a copy goes to refs_trash/)."""
        audio, fk = ref["audio"], ref.get("file_key")
        is_primary = ref["kind"] == "primary"

        def _remove_original():
            if is_primary or not fk:
                return
            trash = CHARACTERS_DIR / char_name / "refs_trash"
            trash.mkdir(exist_ok=True)
            if audio.exists():
                shutil.copy2(audio, trash / audio.name)
            remove_ref_clip(char_name, fk)

        with dialog_host:
            open_import_dialog(
                audio, audio.stem, dialog_host, on_added=_rebuild_detail, fixed_target=char_name,
                remove_original=None if is_primary else _remove_original,
            )

    def _open_ref_edit(char_name: str, audio: Path, is_primary: bool, file_key: str | None, transcript: str):
        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#333] min-w-[440px] gap-3"
        ):
            ui.label("EDIT REFERENCE").classes("text-white font-mono text-xs tracking-widest")

            ui.label("NAME").classes("text-[#444] font-mono text-[10px] tracking-widest")
            name_inp = (
                ui.input(value=audio.stem)
                .props("outlined dark dense")
                .classes("w-full font-mono")
            )

            ui.label("TRANSCRIPT").classes("text-[#444] font-mono text-[10px] tracking-widest mt-2")
            t_inp = (
                ui.textarea(value=transcript)
                .props("outlined dark rows=3")
                .classes("w-full")
            )

            def _save():
                new_stem = name_inp.value.strip()
                new_t = t_inp.value.strip()
                if not new_stem:
                    return
                try:
                    if is_primary:
                        if new_stem != audio.stem:
                            rename_ref_audio(char_name, new_stem)
                        update_character_metadata(char_name, {"transcript": new_t})
                    else:
                        update_ref_clip(
                            char_name, file_key,
                            new_stem if new_stem != audio.stem else None,
                            new_t,
                        )
                    dlg.close()
                    # notify first: _rebuild_detail() deletes the elements this
                    # handler's context belongs to, so ui.notify/show_error after
                    # it raise "parent element ... has been deleted".
                    ui.notify("saved", type="positive", timeout=2000)
                    _rebuild_detail()
                except Exception as exc:
                    show_error(exc)

            name_inp.on("keydown.enter", lambda: _save())
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("Save", on_click=_save).props("color=amber unelevated dense")
        dlg.open()

    def _clip_section(char_name: str, title: str, icon_name: str, icon_color: str,
                      clips: list[Path], subfolder: str):
        with ui.row().classes("items-center gap-2"):
            ui.label(title).classes("text-[#444] font-mono text-[10px] tracking-widest")
            ui.label(str(len(clips))).classes(
                "text-[#333] font-mono text-[10px] bg-[#1a1a1a] px-1.5 rounded"
            )

        for clip in clips:
            rel = clip.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"

            with ui.row().classes(
                "w-full items-center gap-1 px-2 py-1 rounded "
                "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
            ):
                play_btn = ui.button(icon="play_arrow").props(
                    "flat dense color=amber"
                ).classes("shrink-0")
                ui.icon(icon_name, size="12px").classes(f"{icon_color} shrink-0")
                name_lbl = ui.label(clip.stem).classes(
                    "text-[#666] font-mono text-[10px] truncate flex-grow cursor-pointer"
                )
                bind_play(play_btn, name_lbl, url=url)
                ui.label(".wav").classes("text-[#333] font-mono text-[10px] shrink-0")
                ui.button(
                    icon="tune",
                    on_click=lambda c=clip: send_to_edit(c, char_name),
                ).props("flat dense color=grey").classes("shrink-0").tooltip("Send to Edit tab")
                ui.button(
                    icon="drive_file_rename_outline",
                    on_click=lambda c=clip: _open_rename_dialog(char_name, subfolder, c),
                ).props("flat dense color=grey").classes("shrink-0")

    def _rebuild_detail():
        right_col.clear()
        name = _selected[0]
        if not name:
            with right_col:
                ui.label("Select a character from the list.").classes(
                    "text-[#444] font-mono text-[10px] py-4"
                )
            return

        char = get_character(name)
        if not char:
            with right_col:
                ui.label("Character not found.").classes("text-red-500 font-mono text-[10px] py-4")
            return

        is_base = "parent" not in char
        char_dir = CHARACTERS_DIR / name

        clips = list_character_clips(name) if is_base else []
        favs = list_character_favorites(name) if is_base else []
        has_model = bool(char.get("piper_model") or char.get("rvc_model"))

        with right_col:
            # ── header — stays visible no matter which sub-tab is open ──────
            ui.label(name).classes("text-white font-mono text-xl tracking-wide")
            if is_base:
                ui.label("BASE CHARACTER").classes(
                    "text-amber-500 font-mono text-[10px] tracking-widest"
                )
            else:
                ui.label(f"SUB-CHARACTER  ↳  {char['parent']}").classes(
                    "text-[#555] font-mono text-[10px] tracking-widest"
                )
            training_status_chip(name, on_change, _rebuild_detail)
            ui.separator().classes("my-3 opacity-20")

            with ui.tabs(value=_detail_tab[0]).classes("w-full") as detail_tabs:
                tab_about = ui.tab("ABOUT")
                tab_refs = ui.tab("REFERENCES")
                tab_train = ui.tab("TRAINING")
                tab_models = ui.tab("MODELS")
                tab_clips = ui.tab("CLIPS")

            detail_tabs.on_value_change(lambda e: _detail_tab.__setitem__(0, e.value))
            with ui.tab_panels(detail_tabs, value=_detail_tab[0]).classes("w-full bg-transparent"):

                # ══ ABOUT ════════════════════════════════════════════════════
                with ui.tab_panel(tab_about).classes("gap-2 p-0 pt-3"):
                    def _goto(target: str):
                        label = {"references": "REFERENCES", "training": "TRAINING"}[target]
                        _detail_tab[0] = label
                        detail_tabs.set_value(label)

                    build_about_panel(
                        name, goto=_goto, open_character=_select,
                        qualify=lambda n=name: open_qualify_dialog(n, _rebuild_detail),
                        edit=lambda n=name: _open_edit(n), delete=lambda n=name: _confirm_delete(n),
                    )

                # ══ REFERENCES ═══════════════════════════════════════════════
                with ui.tab_panel(tab_refs).classes("gap-2 p-0 pt-3"):
                    def _ref_hooks():
                        def as_ref(row):
                            return {"audio": row["audio"], "file_key": None if row["primary"] else row["key"],
                                    "kind": "primary" if row["primary"] else "clip"}
                        return {
                            "edit": lambda r: _open_ref_edit(name, r["audio"], r["primary"], None if r["primary"] else r["key"], r["transcript"]),
                            "split": lambda r: _split_ref_clip(name, as_ref(r)),
                            "to_edit": lambda r: send_to_edit(r["audio"], name),
                            "open_character": _select,
                        }
                    build_refs_hub(name, dialog_host=dialog_host, rebuild=_rebuild_detail, hooks=_ref_hooks())

                # ══ TRAINING ═════════════════════════════════════════════════
                with ui.tab_panel(tab_train).classes("gap-2 p-0 pt-3"):
                    with ui.row().classes("items-center gap-1"):
                        section_label("TRAINING DATA")
                        ui.icon("info_outline", size="13px").classes("text-[#3a3a3a] cursor-help").tooltip(
                            "A run learns from the ACTIVE clips only. Switch groups on/off here, or clip by clip "
                            "on the REFERENCES tab. Switched-off clips are kept, not deleted.")
                    build_activation_panel(
                        name, on_change=on_change,
                        goto_references=lambda: (_detail_tab.__setitem__(0, "REFERENCES"), detail_tabs.set_value("REFERENCES")),
                    )
                    ui.separator().classes("my-2 opacity-10")


                    def _base_voice_opts() -> dict[str, str]:
                        opts = {}
                        for key, v in BASE_VOICES.items():
                            ready = base_voice_ready(key)
                            suffix = "" if ready else "  (~800 MB download)"
                            opts[key] = f"{v['label']}{suffix}"
                        return opts

                    _default_voice = get_default_base_voice()
                    if _default_voice not in BASE_VOICES:
                        _default_voice = "amy"
                    _base_voice_key: list[str] = [_default_voice]
                    with ui.row().classes("items-center gap-1"):
                        section_label("BASE VOICE")
                        ui.icon("info_outline", size="13px").classes(
                            "text-[#3a3a3a] cursor-help"
                        ).tooltip(
                            "What the fine-tune starts from. With only a handful of ref "
                            "clips, picking a base already close to the target's gender/"
                            "timbre matters more than most other settings. Not-yet-"
                            "downloaded voices download automatically the first time "
                            "you train with them."
                        )

                    def _on_base_voice_change(e):
                        _base_voice_key[0] = e.value
                        base_voice_desc.set_text(BASE_VOICES.get(e.value, {}).get("desc", ""))
                        base_voice_preview_player.set_visibility(False)
                        set_default_btn.set_visibility(e.value != get_default_base_voice())

                    def _set_as_default():
                        set_default_base_voice(_base_voice_key[0])
                        set_default_btn.set_visibility(False)
                        voice_label = BASE_VOICES.get(_base_voice_key[0], {}).get("label", _base_voice_key[0])
                        ui.notify(f"{voice_label} set as default base voice", type="positive", timeout=2500)

                    with ui.row().classes("w-full items-center gap-2"):
                        base_voice_select = (
                            ui.select(options=_base_voice_opts(), value=_default_voice, on_change=_on_base_voice_change)
                            .props("outlined dark dense").classes("flex-grow mb-2")
                        )
                        set_default_btn = ui.button(
                            "SET AS DEFAULT", icon="push_pin", on_click=_set_as_default,
                        ).props("flat dense color=grey").classes("font-mono text-[9px] tracking-widest shrink-0")
                    set_default_btn.set_visibility(_default_voice != get_default_base_voice())
                    base_voice_desc = ui.label(BASE_VOICES[_default_voice]["desc"]).classes(
                        "text-[#444] font-mono text-[10px] tracking-wide mb-2"
                    )

                    # ── what to match: base voice vs. the reference clips ───
                    _match = get_piper_match()
                    with ui.row().classes("items-center gap-1"):
                        section_label("MATCH")
                        ui.icon("info_outline", size="13px").classes(
                            "text-[#3a3a3a] cursor-help"
                        ).tooltip(
                            "For each aspect of the voice, choose whether the trained model "
                            "should match the BASE MODEL (stay exactly as the base voice has "
                            "it — that part of the model is frozen) or the REFERENCE CLIPS "
                            "(learn it from your audio). With a small dataset, learning "
                            "Pronunciation from the clips usually scrambles it (words mangled, "
                            "speech rushed), so Base Model is the safe choice there. Rhythm "
                            "needs a good amount of consistent speech to learn well."
                        )
                    with ui.column().classes("gap-1 mb-2"):
                        for _aspect, _info in MATCH_ASPECTS.items():
                            with ui.row().classes("items-center gap-2"):
                                ui.label(_info["label"].upper()).classes(
                                    "text-[#888] font-mono text-[10px] tracking-widest w-28"
                                ).tooltip(_info["desc"])
                                ui.toggle(
                                    MATCH_CHOICES, value=_match[_aspect],
                                    on_change=lambda e, a=_aspect: set_piper_match(a, e.value),
                                ).props("dense no-caps unelevated toggle-color=amber text-color=grey-6")
                    # ── learning rate ───────────────────────────────────────
                    def _on_lr_change(e):
                        try:
                            if e.value and e.value > 0:
                                set_piper_learning_rate(e.value)
                        except Exception:
                            pass

                    with ui.row().classes("items-center gap-1"):
                        section_label("LEARNING RATE")
                        ui.icon("info_outline", size="13px").classes(
                            "text-[#3a3a3a] cursor-help"
                        ).tooltip(
                            "How big each training step is. Lower = gentler, smoother "
                            "convergence but slower change. The base checkpoint "
                            "is already decayed (lessac sits at about 1.5e-4, amy about 8.7e-5) and decays slightly every epoch; "
                            "the value here replaces that when training starts, then "
                            "keeps the same per-epoch decay. 5e-5 is a good fine-tune "
                            "default; try 2e-5 if the voice has stopped improving. "
                            "Clear it to keep the checkpoint's own rate."
                        )
                    lr_input = (
                        ui.number(value=get_piper_learning_rate(), format="%.2e", min=1e-7, max=1e-3,
                                  step=1e-5, on_change=_on_lr_change)
                        .props("outlined dark dense").classes("w-48 mb-2 font-mono")
                    )

                    # ── epochs ──────────────────────────────────────────────
                    def _on_epochs_change(e):
                        try:
                            if e.value and e.value >= 1:
                                set_piper_epochs(e.value)
                        except Exception:
                            pass

                    with ui.row().classes("items-center gap-1"):
                        section_label("EPOCHS")
                        ui.icon("info_outline", size="13px").classes(
                            "text-[#3a3a3a] cursor-help"
                        ).tooltip(
                            "How many additional epochs to fine-tune for, counted from the "
                            "starting checkpoint. A checkpoint is saved for audition every "
                            "100 epochs (or just once at the end if you ask for fewer than 100), "
                            "so you can pick an earlier one if later ones overfit. You can also stop early at any time."
                        )
                    epochs_input = (
                        ui.number(value=get_piper_epochs(), format="%d", min=1, max=100000, step=100,
                                  on_change=_on_epochs_change)
                        .props("outlined dark dense").classes("w-48 mb-2 font-mono")
                    )

                    # ── base voice preview — hear it before committing ──────
                    async def _preview_base_voice():
                        key = _base_voice_key[0]
                        preview_btn.set_text("LOADING…")
                        preview_btn.disable()
                        try:
                            path = await generate_base_voice_sample(key)
                            rel = path.relative_to(Path("outputs"))
                            play_url(f"/outputs/{rel.as_posix()}")
                        except Exception as exc:
                            show_error(exc)
                        finally:
                            preview_btn.set_text("▶  PREVIEW VOICE")
                            preview_btn.enable()

                    with ui.row().classes("items-center gap-2 mb-1"):
                        preview_btn = ui.button("▶  PREVIEW VOICE", on_click=_preview_base_voice).props(
                            "flat dense color=grey"
                        ).classes("font-mono text-[10px] tracking-widest")
                        ui.label(f'"{SAMPLE_TEXT}"').classes(
                            "text-[#444] font-mono text-[10px] italic"
                        )
                    prime_audio(preview_btn)

                    ui.separator().classes("my-3 opacity-20")

                    async def _do_train(n=name, opts: dict | None = None):
                        """Start training on the ACTIVE clips. opts (from the MODELS tab's "train from this")
                        overrides the settings on this tab: base_voice, learning_rate, match, resume_model,
                        replace_id, epochs."""
                        st = read_training_status(CHARACTERS_DIR / n)
                        if st and st.get("state") in _ACTIVE_STATES:
                            open_training_dialog(n, on_change, _rebuild_detail)
                            return
                        full_chain = get_ref_chain(n)
                        chain = training_clips(full_chain)
                        if not chain:
                            ui.notify(
                                "No reference clips — add some on the REFERENCES tab" if not full_chain else
                                "No clips are switched on for training — activate some above", type="warning")
                            return

                        import soundfile as _sf
                        total_dur = 0.0
                        for c in chain:
                            try:
                                total_dur += _sf.info(str(c["audio"])).duration
                            except Exception:
                                pass
                        # Piper only hard-requires 2 clips with transcripts, but good fine-tunes need real
                        # coverage of the voice; warn on total duration (clip count is a weak proxy).
                        # ~10 min is a rough floor for a recognizable voice. A soft nudge, not a requirement.
                        if total_dur < 600:
                            ui.notify(
                                f"Only {total_dur / 60:.1f} min of active audio ({len(chain)} clips) — Piper fine-tunes "
                                "usually need 10+ min for a recognizable voice, 30-60+ for strong resemblance. "
                                "Training will proceed, but timbre may not match well.",
                                type="warning", timeout=6000,
                            )

                        from faily.modules.piper import diagnose
                        chosen_voice = opts["base_voice"] if opts else _base_voice_key[0]
                        run_match = opts["match"] if opts else get_piper_match()
                        run_lr = opts["learning_rate"] if opts else (lr_input.value or None)
                        run_epochs = int((opts or {}).get("epochs") or epochs_input.value or get_piper_epochs())
                        reason = await ni_run.io_bound(diagnose)
                        if reason:
                            ui.notify(f"Piper not ready — {reason}", type="warning", timeout=8000)
                            return

                        try:
                            frozen_modules(run_match)
                        except ValueError as exc:
                            ui.notify(str(exc), type="warning", timeout=7000)
                            return
                        train_chain = chain          # the active clips
                        if len(train_chain) < 2:
                            ui.notify("Not enough active clips to train on — switch more on above or add clips",
                                      type="warning", timeout=6000)
                            return
                        if len(train_chain) < len(full_chain):
                            ui.notify(f"Training on the {len(train_chain)} active of {len(full_chain)} clips", timeout=4000)
                        if start_training(
                            n, train_chain, chosen_voice, run_lr, run_match,
                            resume_model=(opts or {}).get("resume_model"), replace_id=(opts or {}).get("replace_id"),
                            max_epochs=run_epochs,
                        ):
                            open_training_dialog(n, on_change, _rebuild_detail)

                    train_label = "RETRAIN VOICE" if has_model else "TRAIN VOICE"
                    with ui.row().classes("gap-2 flex-wrap mt-1"):
                        ui.button(train_label, icon="model_training", on_click=_do_train).props(
                            "color=indigo unelevated"
                        ).classes(_BTN)
                        ui.button(
                            "SPEAK", icon="spatial_audio",
                            on_click=lambda n=name: on_speak(n),
                        ).props("color=amber unelevated").classes(_BTN)

                # ══ MODELS ═══════════════════════════════════════════════════
                with ui.tab_panel(tab_models).classes("gap-2 p-0 pt-3"):
                    build_models_panel(
                        name, on_change=on_change, rebuild=_rebuild_detail,
                        train=lambda opts, n=name: _do_train(n, opts),
                    )

                # ══ CLIPS ════════════════════════════════════════════════════
                with ui.tab_panel(tab_clips).classes("gap-2 p-0 pt-3"):
                    if not is_base:
                        ui.label(
                            "Personality clips and favorites are recorded on the base character."
                        ).classes("text-[#444] font-mono text-[10px] py-2")
                    elif not clips and not favs:
                        ui.label("No personality clips or favorites yet.").classes(
                            "text-[#444] font-mono text-[10px] py-2"
                        )
                    else:
                        if clips:
                            _clip_section(
                                name, "PERSONALITY CLIPS", "library_music", "text-amber-400",
                                clips, "clips",
                            )
                        if favs:
                            if clips:
                                ui.separator().classes("my-3 opacity-20")
                            _clip_section(
                                name, "FAVORITES", "favorite", "text-pink-400",
                                favs, "favorites",
                            )

    def _open_edit(name: str):
        char = get_character(name)
        if not char:
            return
        is_base = "parent" not in char
        field_key = "transcript" if is_base else "style_prompt"
        field_label = "TRANSCRIPT" if is_base else "STYLE PROMPT"

        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#333] min-w-[420px] gap-3"
        ):
            ui.label(f"EDIT  {name}").classes("text-white font-mono text-sm tracking-widest")
            ui.label(field_label).classes("text-[#444] font-mono text-[10px] tracking-widest")
            field = (
                ui.textarea(value=char.get(field_key, ""))
                .props("outlined dark rows=4")
                .classes("w-full")
            )

            def _save():
                try:
                    update_character_metadata(name, {field_key: field.value})
                    dlg.close()
                    _rebuild_detail()
                    ui.notify(f"saved  {name}", type="positive", timeout=2000)
                except Exception as exc:
                    show_error(exc)

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("Save", on_click=_save).props("color=amber unelevated dense")
        dlg.open()

    def _confirm_delete(name: str):
        char = get_character(name)
        if not char:
            return
        is_base = "parent" not in char
        _, children_map = _grouped()
        children = children_map.get(name, []) if is_base else []
        child_names = [c["name"] for c in children]

        with ui.dialog() as dlg, ui.card().classes(
            "bg-[#1a1a1a] border border-[#3a1a1a] min-w-[380px] gap-3"
        ):
            ui.label("DELETE CHARACTER").classes("text-red-400 font-mono text-xs tracking-widest")
            ui.label(f'Delete "{name}"?').classes("text-white font-mono text-sm")
            if child_names:
                ui.label(
                    f"This will also delete {len(child_names)} variant(s):"
                ).classes("text-[#888] font-mono text-[10px]")
                for cn in child_names:
                    ui.label(f"  ↳ {cn}").classes("text-[#666] font-mono text-[10px]")
            ui.label("This cannot be undone.").classes("text-[#666] font-mono text-[10px]")

            def _do_delete():
                try:
                    for cn in child_names:
                        delete_character(cn)
                    delete_character(name)
                    _selected[0] = None
                    dlg.close()
                    _rebuild_list()
                    _rebuild_detail()
                    on_change()
                    ui.notify(f"deleted  {name}", timeout=2000)
                except Exception as exc:
                    show_error(exc)

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("Delete", on_click=_do_delete).props("color=negative unelevated dense")
        dlg.open()

    # ── layout ────────────────────────────────────────────────────────────────
    with ui.grid(columns=SPLIT_COLUMNS).classes("w-full h-full gap-0"):
        with ui.column().classes("gap-2 p-6 border-r border-[#252525] overflow-y-auto overflow-x-hidden min-w-0"):
            with ui.row().classes("w-full items-center gap-2"):
                section_label("CHARACTER LIBRARY")
                ui.space()
                ui.button(
                    icon="refresh",
                    on_click=lambda: (_rebuild_list(), _rebuild_detail()),
                ).props("flat dense color=grey").tooltip("Reload")
            left_col = ui.column().classes("w-full gap-1 mt-2")
            # Long-lived parent for the split-and-qualify dialogs: right_col is rebuilt (and
            # everything in it deleted) whenever a clip is edited or added.
            dialog_host = ui.element("div").classes("hidden")

        with ui.column().classes("gap-3 p-6 overflow-y-auto overflow-x-hidden min-w-0"):
            right_col = ui.column().classes("w-full gap-2")

    _rebuild_list()
    _rebuild_detail()

    def refresh():
        _rebuild_list()
        _rebuild_detail()

    return refresh
