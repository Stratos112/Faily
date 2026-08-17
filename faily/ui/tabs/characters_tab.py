from pathlib import Path
from nicegui import ui, run as ni_run
from faily.core.characters import (
    list_characters, get_character, get_ref_path, delete_character,
    update_character_metadata,
    list_character_clips, list_character_favorites, rename_character_file,
    get_ref_chain, remove_ref_clip, set_rvc_model, set_piper_model,
    rename_ref_audio, update_ref_clip, clip_quality_issues,
    CHARACTERS_DIR,
)
from faily.ui.components import section_label, show_error, send_to_edit
from faily.modules.piper import BASE_VOICES, base_voice_ready

_BTN = "font-mono tracking-widest"


def build_characters_tab(on_speak, on_change):
    """
    on_speak(name): navigate to SPEAK tab with this character pre-selected
    on_change():    notify app that characters were deleted (refresh other tabs)
    Returns: refresh() callable
    """
    _selected: list[str | None] = [None]

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
                    _rebuild_detail()
                    ui.notify(f"Renamed to {new_stem}.wav", type="positive", timeout=2000)
                except Exception as exc:
                    show_error(exc)

            inp.on("keydown.enter", lambda: _do_rename())
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat dense color=grey")
                ui.button("Rename", on_click=_do_rename).props("color=amber unelevated dense")
        dlg.open()

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
                    _rebuild_detail()
                    ui.notify("saved", type="positive", timeout=2000)
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

        player = ui.audio("").classes("w-full rounded")
        player.set_visibility(False)
        now_playing = ui.label("").classes("text-[#333] font-mono text-[9px]")

        for clip in clips:
            rel = clip.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"

            def _play(u=url, n=clip.stem):
                player.set_source(u)
                player.set_visibility(True)
                now_playing.set_text(f"▶  {n}")

            with ui.row().classes(
                "w-full items-center gap-1 px-2 py-1 rounded "
                "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
            ):
                ui.button(icon="play_arrow", on_click=_play).props(
                    "flat dense color=amber"
                ).classes("shrink-0")
                ui.icon(icon_name, size="12px").classes(f"{icon_color} shrink-0")
                ui.label(clip.stem).classes(
                    "text-[#666] font-mono text-[10px] truncate flex-grow cursor-pointer"
                ).on("click", _play)
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

        with right_col:
            # ── header ──────────────────────────────────────────────────────
            ui.label(name).classes("text-white font-mono text-xl tracking-wide")
            if is_base:
                ui.label("BASE CHARACTER").classes(
                    "text-amber-500 font-mono text-[10px] tracking-widest"
                )
            else:
                ui.label(f"SUB-CHARACTER  ↳  {char['parent']}").classes(
                    "text-[#555] font-mono text-[10px] tracking-widest"
                )

            ui.separator().classes("my-3 opacity-20")

            # ── metadata ────────────────────────────────────────────────────
            def _meta_row(label: str, value: str):
                with ui.row().classes("items-start gap-3 w-full"):
                    ui.label(label).classes(
                        "text-[#444] font-mono text-[10px] tracking-widest w-28 shrink-0 pt-0.5"
                    )
                    ui.label(value).classes("text-[#aaa] font-mono text-[11px] flex-grow break-all")

            created = char.get("created", "—")[:19].replace("T", "  ")
            _meta_row("CREATED", created)

            if is_base:
                # ── variants ────────────────────────────────────────────────
                _, children_map = _grouped()
                children = children_map.get(name, [])
                if children:
                    ui.separator().classes("my-3 opacity-20")
                    ui.label("VARIANTS").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest"
                    )
                    for sub in children:
                        with ui.row().classes(
                            "items-center gap-2 px-2 py-0.5 rounded cursor-pointer hover:bg-[#111]"
                        ).on("click", lambda n=sub["name"]: _select(n)):
                            ui.icon("person_outline", size="12px").classes("text-[#555]")
                            ui.label(f"↳ {sub['name']}").classes(
                                "text-[#666] font-mono text-[10px] hover:text-amber-400"
                            )
            else:
                _meta_row("PARENT", char.get("parent", "—"))
                if char.get("backend"):
                    _meta_row("BACKEND", char["backend"])
                style = char.get("style_prompt", "").strip()
                if style:
                    ui.label("STYLE PROMPT").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest mt-1"
                    )
                    ui.label(style).classes(
                        "text-[#aaa] font-mono text-[11px] leading-relaxed "
                        "bg-[#1a1a1a] rounded px-3 py-2 w-full"
                    )

            # ── references (unified) ──────────────────────────────────────
            own_refs = []
            if is_base and "ref_audio" in char:
                ra = char_dir / char["ref_audio"]
                if ra.exists():
                    own_refs.append({
                        "audio": ra, "transcript": char.get("transcript", ""),
                        "kind": "primary", "file_key": None,
                    })
            for rc in char.get("ref_clips", []):
                rp = char_dir / rc["file"]
                if rp.exists():
                    own_refs.append({
                        "audio": rp, "transcript": rc.get("transcript", ""),
                        "kind": "clip", "file_key": rc["file"],
                    })

            full_chain = get_ref_chain(name)
            own_paths = {r["audio"] for r in own_refs}
            inherited = [e for e in full_chain if e["audio"] not in own_paths]

            if own_refs or inherited:
                ui.separator().classes("my-3 opacity-20")
                all_refs = own_refs + inherited
                total_dur = 0.0
                try:
                    import soundfile as _sf
                    for _r in all_refs:
                        try:
                            total_dur += _sf.info(str(_r["audio"])).duration
                        except Exception:
                            pass
                except Exception:
                    pass
                dur_str = (f"{total_dur:.0f}s" if total_dur < 60
                           else f"{total_dur / 60:.1f}m")
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("REFERENCES").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest flex-grow"
                    )
                    ui.label(f"{len(all_refs)} clips  ·  {dur_str}").classes(
                        "text-[#333] font-mono text-[10px] bg-[#1a1a1a] px-1.5 rounded"
                    )

                ref_player = ui.audio("").classes("w-full rounded mt-1")
                ref_now_playing = ui.label("").classes("text-[#333] font-mono text-[9px]")

                def _play_ref(u, n):
                    ref_player.set_source(u)
                    ref_now_playing.set_text(f"▶  {n}")

                for ref in own_refs:
                    audio = ref["audio"]
                    rel = audio.relative_to(Path("outputs"))
                    url = f"/outputs/{rel.as_posix()}"
                    is_primary = ref["kind"] == "primary"
                    issues = clip_quality_issues(audio)

                    with ui.row().classes(
                        "w-full items-center gap-1 px-2 py-1 rounded "
                        "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
                    ):
                        ui.button(
                            icon="play_arrow",
                            on_click=lambda u=url, n=audio.stem: _play_ref(u, n),
                        ).props("flat dense color=amber").classes("shrink-0")
                        ui.icon("star" if is_primary else "mic", size="12px").classes(
                            ("text-amber-500" if is_primary else "text-[#555]") + " shrink-0"
                        )
                        if issues:
                            ui.icon("warning", size="12px").classes(
                                "text-amber-500 shrink-0"
                            ).tooltip(
                                f"{', '.join(issues)} — consider re-recording or removing "
                                "this clip; low-quality refs drag down cloning and training."
                            )
                        with ui.column().classes("gap-0 flex-grow min-w-0 cursor-pointer").on(
                            "click", lambda u=url, n=audio.stem: _play_ref(u, n)
                        ):
                            ui.label(audio.stem).classes(
                                "text-[#aaa] font-mono text-[10px] truncate"
                            )
                            t = ref["transcript"]
                            if t:
                                ui.label(f'"{t}"').classes(
                                    "text-[#555] font-mono text-[10px] italic leading-tight truncate"
                                )
                        ui.button(
                            icon="tune",
                            on_click=lambda r=ref: send_to_edit(r["audio"], name),
                        ).props("flat dense color=grey").classes("shrink-0").tooltip("Send to Edit tab")
                        ui.button(
                            icon="edit",
                            on_click=lambda r=ref: _open_ref_edit(
                                name, r["audio"], r["kind"] == "primary",
                                r.get("file_key"), r["transcript"]
                            ),
                        ).props("flat dense color=grey").classes("shrink-0")
                        if not is_primary:
                            def _del(fk=ref["file_key"]):
                                remove_ref_clip(name, fk)
                                _rebuild_detail()
                            ui.button(
                                icon="close", on_click=_del,
                            ).props("flat dense color=grey").classes(
                                "shrink-0 opacity-40 hover:opacity-100"
                            )

                for entry in inherited:
                    audio = entry["audio"]
                    if not audio.exists():
                        continue
                    rel = audio.relative_to(Path("outputs"))
                    url = f"/outputs/{rel.as_posix()}"
                    with ui.row().classes(
                        "w-full items-center gap-1 px-2 py-1 rounded "
                        "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
                    ):
                        ui.button(
                            icon="play_arrow",
                            on_click=lambda u=url, n=audio.stem: _play_ref(u, n),
                        ).props("flat dense color=grey").classes("shrink-0")
                        ui.icon("link", size="12px").classes("text-[#333] shrink-0")
                        with ui.column().classes("gap-0 flex-grow min-w-0 cursor-pointer").on(
                            "click", lambda u=url, n=audio.stem: _play_ref(u, n)
                        ):
                            ui.label(audio.stem).classes(
                                "text-[#555] font-mono text-[10px] truncate"
                            )
                            if entry.get("transcript"):
                                ui.label(f'"{entry["transcript"]}"').classes(
                                    "text-[#333] font-mono text-[10px] italic leading-tight truncate"
                                )
                        ui.label("↑ parent").classes("text-[#333] font-mono text-[9px] shrink-0")

            # ── personality clips ────────────────────────────────────────
            if is_base:
                clips = list_character_clips(name)
                if clips:
                    ui.separator().classes("my-3 opacity-20")
                    _clip_section(
                        name, "PERSONALITY CLIPS", "library_music", "text-amber-400",
                        clips, "clips",
                    )

                favs = list_character_favorites(name)
                if favs:
                    ui.separator().classes("my-3 opacity-20")
                    _clip_section(
                        name, "FAVORITES", "favorite", "text-pink-400",
                        favs, "favorites",
                    )

            # ── actions ──────────────────────────────────────────────────────
            ui.separator().classes("mt-4 mb-3 opacity-20")

            has_model = bool(char.get("piper_model") or char.get("rvc_model"))

            # ── base voice picker — what the Piper fine-tune starts from ────
            def _base_voice_opts() -> dict[str, str]:
                opts = {}
                for key, v in BASE_VOICES.items():
                    ready = base_voice_ready(key)
                    suffix = "" if ready else "  (~800 MB download)"
                    opts[key] = f"{v['label']}{suffix}"
                return opts

            _base_voice_key: list[str] = ["lessac"]
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

            base_voice_select = (
                ui.select(options=_base_voice_opts(), value="lessac", on_change=_on_base_voice_change)
                .props("outlined dark dense").classes("w-full mb-2")
            )
            base_voice_desc = ui.label(BASE_VOICES["lessac"]["desc"]).classes(
                "text-[#444] font-mono text-[10px] tracking-wide mb-2"
            )

            with ui.row().classes("gap-2 flex-wrap"):

                async def _do_train(n=name):
                    chain = get_ref_chain(n)
                    if not chain:
                        ui.notify("No ref clips — add clips from CLONE or TUNE first", type="warning")
                        return
                    # Piper only hard-requires 2 clips with transcripts, but good
                    # fine-tunes typically need 30-200+ (several minutes of audio).
                    # Warn below a lower floor without blocking — this is a soft
                    # nudge, not a hard requirement.
                    if len(chain) < 20:
                        ui.notify(
                            f"Only {len(chain)} reference clip{'s' if len(chain) != 1 else ''} — "
                            "Piper fine-tunes usually need 30-200+ for good results. "
                            "Training will proceed, but quality may suffer.",
                            type="warning", timeout=6000,
                        )

                    from faily.modules.piper import (
                        train, diagnose, finalize_piper_model, infer, download_base_voice,
                    )
                    chosen_voice = _base_voice_key[0]
                    reason = await ni_run.io_bound(diagnose)
                    if reason:
                        ui.notify(f"Piper not ready — {reason}", type="warning", timeout=8000)
                        return

                    _proc_ref: list = [None]
                    _log_lines: list[str] = []
                    _candidates: list[tuple[int, Path]] = []
                    char_dir = CHARACTERS_DIR / n

                    with ui.dialog().classes("w-full max-w-3xl") as dlg, ui.card().classes(
                        "bg-[#0d0d0d] border border-[#252525] w-full gap-3 p-5"
                    ):
                        ui.label(f"TRAINING  {n}").classes(
                            "text-white font-mono text-xs tracking-widest"
                        )
                        log = ui.log(max_lines=2000).classes(
                            "w-full font-mono text-[10px] rounded border border-[#1a1a1a]"
                        ).style("height:420px; background:#050505")
                        with ui.row().classes("w-full justify-between items-center"):
                            status_lbl = ui.label("running…").classes(
                                "text-[#555] font-mono text-[10px]"
                            )
                            with ui.row().classes("gap-2"):
                                ui.button(
                                    "Copy log", icon="content_copy",
                                    on_click=lambda: ui.clipboard.write("\n".join(_log_lines)),
                                ).props("flat dense color=amber")
                                ui.button(
                                    "STOP",
                                    on_click=lambda: _proc_ref[0].terminate() if _proc_ref[0] else None,
                                ).props("flat dense color=negative")
                                close_btn = (
                                    ui.button("CLOSE", on_click=dlg.close)
                                    .props("flat dense color=grey")
                                )
                                close_btn.set_visibility(False)

                        ui.separator().classes("opacity-20")
                        ui.label(
                            "PREVIEW LINE — used to audition each saved checkpoint once "
                            "training finishes. Edit it any time before then."
                        ).classes("text-[#444] font-mono text-[10px] leading-snug")
                        preview_text_input = (
                            ui.input(value="Hello, this is a test of the trained voice.")
                            .props("outlined dark dense")
                            .classes("w-full font-mono text-[11px]")
                        )

                        # ── checkpoint picker — populated once training finishes ────
                        with ui.column().classes("w-full gap-2") as picker_section:
                            ui.separator().classes("opacity-20")
                            ui.label("PICK A CHECKPOINT").classes(
                                "text-amber-400 font-mono text-[10px] tracking-widest"
                            )
                            ui.label(
                                "Each saved checkpoint is a full, independent snapshot — the "
                                "last one isn't always the best-sounding. Listen and pick "
                                "your favorite; the rest are discarded."
                            ).classes("text-[#444] font-mono text-[10px] leading-snug")
                            candidates_col = ui.column().classes("w-full gap-1")
                        picker_section.set_visibility(False)
                    dlg.open()

                    def _ui_safe(fn):
                        # If the browser tab/connection dies mid-training, NiceGUI
                        # raises RuntimeError on any attempt to touch its elements.
                        # Training itself (a separate OS process) keeps running
                        # regardless — don't let a dead UI client abort the actual
                        # training/export logic or the file writes that follow it.
                        try:
                            fn()
                        except RuntimeError:
                            pass

                    def _log(line: str):
                        _log_lines.append(line)
                        _ui_safe(lambda: log.push(line))

                    async def _choose(chosen_onnx: Path):
                        try:
                            final_path = await ni_run.io_bound(finalize_piper_model, char_dir, chosen_onnx)
                            set_piper_model(n, str(final_path))
                            ui.notify(f"Piper model set for {n}", type="positive", timeout=4000)
                            on_change()
                            _rebuild_detail()
                            dlg.close()
                        except Exception as exc:
                            show_error(exc)

                    async def _generate_previews():
                        candidates_col.clear()
                        text = preview_text_input.value.strip() or "Hello, this is a test of the trained voice."
                        for epoch, onnx_path in _candidates:
                            _ui_safe(lambda e=epoch: status_lbl.set_text(f"Generating preview — epoch {e}…"))
                            preview_path = onnx_path.with_name(f"preview_epoch_{epoch}.wav")
                            try:
                                await ni_run.io_bound(infer, text, onnx_path, preview_path)
                            except Exception as exc:
                                show_error(exc)
                                continue
                            rel = preview_path.relative_to(Path("outputs"))
                            with candidates_col:
                                with ui.row().classes(
                                    "w-full items-center gap-2 px-3 py-1 rounded "
                                    "border border-[#1e1e1e] hover:border-[#333]"
                                ):
                                    ui.label(f"epoch {epoch}").classes(
                                        "text-[#888] font-mono text-[10px] w-20 shrink-0"
                                    )
                                    ui.audio(f"/outputs/{rel.as_posix()}").classes("flex-grow").props("controls")
                                    ui.button(
                                        "USE THIS", icon="check",
                                        on_click=lambda p=onnx_path: _choose(p),
                                    ).props("flat dense color=amber").classes("font-mono text-[10px] shrink-0")

                    try:
                        if not base_voice_ready(chosen_voice):
                            _log(f"Downloading base voice: {BASE_VOICES.get(chosen_voice, {}).get('label', chosen_voice)}…")
                            await download_base_voice(chosen_voice, _log)
                        _candidates.extend(
                            await train(chain, char_dir, _log, _proc_ref, base_voice=chosen_voice)
                        )
                        _ui_safe(lambda: picker_section.set_visibility(True))
                        await _generate_previews()
                        _ui_safe(lambda: status_lbl.set_text(f"✓  {len(_candidates)} checkpoint(s) ready — pick one below"))
                        _ui_safe(lambda: status_lbl.classes(remove="text-[#555]", add="text-green-400"))
                    except Exception as exc:
                        _ui_safe(lambda: status_lbl.set_text("error"))
                        _ui_safe(lambda: status_lbl.classes(remove="text-[#555]", add="text-red-400"))
                        _ui_safe(lambda: show_error(exc))
                    finally:
                        _ui_safe(lambda: close_btn.set_visibility(True))

                train_label = "RETRAIN VOICE" if has_model else "TRAIN VOICE"
                ui.button(train_label, icon="model_training", on_click=_do_train).props(
                    "color=indigo unelevated"
                ).classes(_BTN)

                ui.button(
                    "SPEAK", icon="spatial_audio",
                    on_click=lambda n=name: on_speak(n),
                ).props("color=amber unelevated").classes(_BTN)

                ui.button("EDIT", icon="edit", on_click=lambda n=name: _open_edit(n)).props(
                    "flat color=grey"
                )
                (
                    ui.button(
                        "DELETE", icon="delete_outline",
                        on_click=lambda n=name: _confirm_delete(n),
                    )
                    .props("flat color=negative")
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
    with ui.grid(columns="2fr 3fr").classes("w-full h-full gap-0"):
        with ui.column().classes("gap-2 p-8 border-r border-[#252525] overflow-y-auto"):
            with ui.row().classes("w-full items-center gap-2"):
                section_label("CHARACTER LIBRARY")
                ui.space()
                ui.button(
                    icon="refresh",
                    on_click=lambda: (_rebuild_list(), _rebuild_detail()),
                ).props("flat dense color=grey").tooltip("Reload")
            left_col = ui.column().classes("w-full gap-1 mt-2")

        with ui.column().classes("gap-3 p-8 overflow-y-auto"):
            right_col = ui.column().classes("w-full gap-2")

    _rebuild_list()
    _rebuild_detail()

    def refresh():
        _rebuild_list()
        _rebuild_detail()

    return refresh
