from pathlib import Path
from nicegui import ui, run as ni_run
from faily.core.characters import (
    list_characters, get_character, get_ref_path, delete_character,
    update_character_metadata,
    list_character_clips, list_character_favorites, rename_character_file,
    get_ref_chain, remove_ref_clip, set_rvc_model,
    CHARACTERS_DIR,
)
from faily.ui.components import section_label, show_error

_BTN = "font-mono tracking-widest"


def build_characters_tab(on_speak, on_change):
    """
    on_speak(name): navigate to TUNE tab with this character pre-selected
    on_change():    notify app that characters were deleted (refresh other tabs)
    Returns: refresh() callable
    """
    _selected: list[str | None] = [None]

    def _grouped() -> tuple[list[dict], dict[str, list[dict]]]:
        """Return (roots, children_by_parent). Works at arbitrary depth."""
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

    def _clip_section(char_name: str, title: str, icon_name: str, icon_color: str,
                      clips: list[Path], subfolder: str):
        """Build a browseable, renameable clip list section inside the current context."""
        with ui.row().classes("items-center gap-2"):
            ui.label(title).classes("text-[#444] font-mono text-[10px] tracking-widest")
            ui.label(str(len(clips))).classes(
                "text-[#333] font-mono text-[10px] bg-[#1a1a1a] px-1.5 rounded"
            )

        player = ui.audio("").classes("w-full rounded")
        player.set_visibility(False)

        for clip in clips:
            rel = clip.relative_to(Path("outputs"))
            url = f"/outputs/{rel.as_posix()}"

            with ui.row().classes(
                "w-full items-center gap-1 px-2 py-1 rounded "
                "hover:bg-[#1a1a1a] border border-transparent hover:border-[#2a2a2a]"
            ):
                ui.icon(icon_name, size="12px").classes(
                    f"{icon_color} shrink-0 cursor-pointer"
                ).on("click", lambda u=url: (
                    player.set_source(u),
                    player.set_visibility(True),
                ))
                ui.label(clip.stem).classes(
                    "text-[#666] font-mono text-[10px] truncate flex-grow cursor-pointer"
                ).on("click", lambda u=url: (
                    player.set_source(u),
                    player.set_visibility(True),
                ))
                ui.label(".wav").classes("text-[#333] font-mono text-[10px] shrink-0")
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
                ref = get_ref_path(name)
                ref_label = ref.name if (ref and ref.exists()) else "⚠  missing"
                _meta_row("REF AUDIO", ref_label)

                transcript = char.get("transcript", "").strip()
                if transcript:
                    ui.label("TRANSCRIPT").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest mt-1"
                    )
                    ui.label(transcript).classes(
                        "text-[#aaa] font-mono text-[11px] leading-relaxed "
                        "bg-[#1a1a1a] rounded px-3 py-2 w-full"
                    )

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

                # ── ref audio player ─────────────────────────────────────────
                if ref and ref.exists():
                    ui.separator().classes("my-3 opacity-20")
                    ui.label("REFERENCE AUDIO").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest"
                    )
                    rel = ref.relative_to(Path("outputs"))
                    ui.audio(f"/outputs/{rel.as_posix()}").classes("w-full rounded mt-1")

                # ── personality clips ────────────────────────────────────────
                clips = list_character_clips(name)
                if clips:
                    ui.separator().classes("my-3 opacity-20")
                    _clip_section(
                        name, "PERSONALITY CLIPS", "library_music", "text-amber-400",
                        clips, "clips",
                    )

                # ── favorites ────────────────────────────────────────────────
                favs = list_character_favorites(name)
                if favs:
                    ui.separator().classes("my-3 opacity-20")
                    _clip_section(
                        name, "FAVORITES", "favorite", "text-pink-400",
                        favs, "favorites",
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

            # ── added ref clips (base and sub) ────────────────────────────────
            own_refs = char.get("ref_clips", [])
            if own_refs:
                ui.separator().classes("my-3 opacity-20")
                ui.label("ADDED REF CLIPS").classes(
                    "text-[#444] font-mono text-[10px] tracking-widest"
                )
                for rc in own_refs:
                    file_key = rc["file"]
                    t = rc.get("transcript", "").strip()
                    with ui.row().classes("items-center gap-2 w-full px-1 py-0.5"):
                        ui.icon("mic", size="12px").classes("text-amber-500 shrink-0")
                        with ui.column().classes("gap-0 flex-grow min-w-0"):
                            ui.label(file_key.split("/")[-1]).classes(
                                "text-[#aaa] font-mono text-[10px] truncate"
                            )
                            if t:
                                ui.label(f'"{t}"').classes(
                                    "text-[#555] font-mono text-[10px] italic leading-tight"
                                )
                        def _remove_clip(fk=file_key, n=name):
                            remove_ref_clip(n, fk)
                            _rebuild_detail()
                        ui.button(
                            icon="close", on_click=_remove_clip,
                        ).props("flat dense color=grey").classes("shrink-0 opacity-40 hover:opacity-100")

            # ── reference chain ───────────────────────────────────────────────
            chain = get_ref_chain(name)
            if chain:
                ui.separator().classes("my-3 opacity-20")
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label(f"REFERENCE CHAIN  ({len(chain)} clips)").classes(
                        "text-[#444] font-mono text-[10px] tracking-widest flex-grow"
                    )
                    ui.button("RELOAD", icon="refresh", on_click=_rebuild_detail).props(
                        "flat dense color=amber"
                    ).classes("font-mono text-[10px] tracking-widest shrink-0")

                for entry in chain:
                    audio: Path = entry["audio"]
                    t: str = entry.get("transcript", "").strip()
                    with ui.row().classes("items-start gap-2 w-full px-1 py-0.5"):
                        ui.icon("mic", size="12px").classes("text-[#555] mt-0.5 shrink-0")
                        with ui.column().classes("gap-0 flex-grow min-w-0"):
                            ui.label(audio.name).classes(
                                "text-[#666] font-mono text-[10px] truncate"
                            )
                            if t:
                                ui.label(f'"{t}"').classes(
                                    "text-[#444] font-mono text-[10px] italic leading-tight"
                                )

            # ── actions ──────────────────────────────────────────────────────
            ui.separator().classes("mt-4 mb-3 opacity-20")

            rvc_status = char.get("rvc_model", "")
            if rvc_status:
                ui.label("MODEL TRAINED").classes(
                    "text-indigo-400 font-mono text-[9px] tracking-widest"
                )

            with ui.row().classes("gap-2 flex-wrap"):
                (
                    ui.button("TUNE", icon="record_voice_over", on_click=lambda n=name: on_speak(n))
                    .props("color=amber unelevated")
                    .classes(_BTN)
                )

                train_btn_holder: list = []

                async def _do_train(n=name):
                    chain = get_ref_chain(n)
                    if not chain:
                        ui.notify("No ref clips — add clips from CLONE or TUNE first", type="warning")
                        return
                    btn = train_btn_holder[0]
                    btn.props("loading")
                    btn.disable()
                    try:
                        from faily.modules.rvc import train_rvc
                        clips = [e["audio"] for e in chain]
                        model_path = await ni_run.io_bound(
                            train_rvc, n, clips, CHARACTERS_DIR / n, None
                        )
                        set_rvc_model(n, str(model_path))
                        ui.notify(
                            f"RVC model trained  ·  {len(clips)} clip{'s' if len(clips) != 1 else ''}",
                            type="positive", timeout=3000,
                        )
                        on_change()
                        _rebuild_detail()
                    except Exception as exc:
                        show_error(exc)
                        btn.props(remove="loading")
                        btn.enable()

                train_btn = (
                    ui.button("TRAIN", icon="model_training", on_click=_do_train)
                    .props("color=indigo unelevated")
                    .classes(_BTN)
                )
                train_btn_holder.append(train_btn)

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
            section_label("CHARACTER LIBRARY")
            left_col = ui.column().classes("w-full gap-1 mt-2")

        with ui.column().classes("gap-3 p-8 overflow-y-auto"):
            right_col = ui.column().classes("w-full gap-2")

    _rebuild_list()
    _rebuild_detail()

    def refresh():
        _rebuild_list()
        _rebuild_detail()

    return refresh
