"""Renders the preprocessing tool chain: add/remove/reorder steps, per-step settings + preview.

The tool chain list lives in the Streamlit sidebar (st.sidebar), not the main content
area. It scrolls independently and stays visible regardless of chain length, so picking
a step never requires scrolling down to its settings and back up to pick the next one.
The main content area holds the settings panel and a 3-column before/after preview.

The in-progress step chain is persisted to project.draft_pipeline_path on every change,
not just kept in st.session_state — session_state is wiped on a browser refresh or a new
tab, so without this the user would lose their chain and have to rebuild it from scratch.

Reordering uses up/down buttons rather than a drag-and-drop component — simpler and more
reliable than depending on a third-party Streamlit component, at the cost of being less
slick. The step list, once configured, composes with `preprocessing.pipeline` exactly the
same way regardless of how it was arranged.
"""
import json
import uuid

import streamlit as st

from image_processing.core import repository as repo
from image_processing.core import thumbnails
from image_processing.core.config_schema import PipelineStep, PreprocessConfig, ScopeRule
from image_processing.core.hashing import stable_hash
from image_processing.core.store import read_json, write_json
from image_processing.preprocessing import align, crop, pipeline
from image_processing.preprocessing.registry import TOOL_LABELS, TOOLS, get_tool
from image_processing.preprocessing.ui_widgets import dynamic_bounded_slider

STEPS_KEY = "pp_steps"
SCOPE_KEY = "pp_scope"
SELECTED_KEY = "pp_selected_index"
SAMPLE_KEY = "pp_sample_image_id"
MAX_SCOPE_THUMBNAILS = 48


def _new_rule(all_of: list[str] | None = None) -> dict:
    # Draft rules carry an id (unlike ScopeRule) so their widgets get stable keys across
    # add/delete — see the explicit-key note in _render_scope_builder.
    return {"id": uuid.uuid4().hex[:8], "all_of": list(all_of or []), "none_of": []}


def _load_draft(project) -> dict:
    draft = read_json(project.draft_pipeline_path, default={"steps": []})
    if "scope" not in draft:
        # Drafts saved before scope rules existed hold a flat "has all of" category list.
        legacy = draft.get("categories", [])
        draft["scope"] = [_new_rule(legacy)] if legacy else []
    return draft


def _save_draft(project, steps: list[dict]) -> None:
    # Reads the scope from session_state rather than taking it as a parameter, so every
    # existing _save_draft(project, steps) call site (add/remove/reorder step, landmark
    # edits, param changes) keeps working unchanged — the scope is only ever mutated by
    # _render_scope_builder, never by any of those other call sites.
    scope = st.session_state.get(SCOPE_KEY, [])
    write_json(project.draft_pipeline_path, {"steps": steps, "scope": scope})


def _steps(project) -> list[dict]:
    if STEPS_KEY not in st.session_state:
        draft = _load_draft(project)
        st.session_state[STEPS_KEY] = draft["steps"]
        st.session_state[SCOPE_KEY] = draft["scope"]
    return st.session_state[STEPS_KEY]


def _scope_rules(project) -> list[dict]:
    if SCOPE_KEY not in st.session_state:
        _steps(project)  # loads both keys together from the same draft file
    return st.session_state[SCOPE_KEY]


def _to_scope(rules: list[dict]) -> list[ScopeRule]:
    # Going through PreprocessConfig's validator drops empty rules, same as on save.
    return PreprocessConfig(scope=[ScopeRule(all_of=r["all_of"], none_of=r["none_of"]) for r in rules]).scope


def _known_risky_order_warning(steps: list[dict]) -> str | None:
    types = [s["type"] for s in steps]
    if "align" in types:
        align_idx = types.index("align")
        if "crop" in types[:align_idx]:
            return "A crop step runs before align — it may crop away a needed landmark region."
        if "resize" in types[:align_idx]:
            return "A resize step runs before align — landmark coordinates were picked at a different scale."
    return None


def _sample_image(project, scope: list[ScopeRule]) -> tuple[str | None, "np.ndarray | None"]:
    ids = repo.filter_by_scope(project, [img["id"] for img in repo.list_images(project)], scope)
    if not ids:
        return None, None
    current = st.session_state.get(SAMPLE_KEY, ids[0])
    if current not in ids:
        # The remembered sample was deleted (Import Images) or left the scope — reset the
        # widget's own state too, not just the local, or the selectbox below would be
        # handed a session_state value that isn't among its options.
        current = ids[0]
        st.session_state[SAMPLE_KEY] = current
    current_index = ids.index(current)

    chosen = st.selectbox("Preview sample image", ids, index=current_index, key=SAMPLE_KEY)

    # Previous/Next mutate SAMPLE_KEY via on_click callbacks rather than an inline
    # `if st.button(): ...`. Callbacks run in Streamlit's pre-script phase, before the
    # script body (and therefore the selectbox above) re-executes — an inline check would
    # instead run *after* the selectbox has already been instantiated this same script
    # pass (since the buttons are placed below it), and Streamlit forbids writing to a
    # widget's session_state key once that widget is already instantiated in the same run.
    def _step(offset: int) -> None:
        new_index = current_index + offset
        if 0 <= new_index < len(ids):
            st.session_state[SAMPLE_KEY] = ids[new_index]

    prev_col, next_col = st.columns(2)
    with prev_col:
        st.button(
            "Previous", icon=":material/chevron_left:", key="pp_sample_prev", width="stretch",
            disabled=current_index == 0, on_click=_step, args=(-1,),
        )
    with next_col:
        st.button(
            "Next", icon=":material/chevron_right:", key="pp_sample_next", width="stretch",
            disabled=current_index == len(ids) - 1, on_click=_step, args=(1,),
        )

    image = pipeline.load_image(repo.image_path(project, chosen))
    return chosen, image


def _render_align_landmark_manager(project, step: dict, steps: list[dict], sample_image) -> None:
    """Mutates step['params']['landmarks'] directly (step is a live entry in `steps`,
    which is the same list object backing session_state) so removals/additions are never
    lost to a discarded local copy when a button handler triggers an immediate rerun."""
    st.markdown("**Manage landmarks**")
    images = repo.list_images(project)
    ids = [img["id"] for img in images]
    if not ids:
        st.info("No images in the project — import some to pick landmarks from.")
        return
    if st.session_state.get("align_ref_image") not in (None, *ids):
        st.session_state["align_ref_image"] = ids[0]  # remembered reference image was deleted
    ref_id = st.selectbox("Reference image", ids, key="align_ref_image")
    ref_image = pipeline.load_image(repo.image_path(project, ref_id))
    rh, rw = ref_image.shape[:2]

    # Tracks the in-progress landmark pick across reruns, independent of the widgets' own
    # session_state keys. dynamic_bounded_slider needs this: since the position sliders'
    # bounds are recomputed from Width/Height every render, their `value=` fallback must
    # come from a source that isn't itself reset by that bounds change — see its docstring
    # for why reading the widgets' own keys back wouldn't work here.
    pending_key = f"lm_pending_{step['id']}"
    pending = st.session_state.setdefault(
        pending_key, {"w": max(8, min(40, rw)), "h": max(8, min(40, rh)), "cx": rw // 2, "cy": rh // 2}
    )

    col1, col2 = st.columns(2)
    with col2:
        # step["id"]-scoped prefixes so two separate align steps don't share landmark-picker
        # state.
        #
        # Width/Height are sized FIRST, against the full image dimensions — independent of
        # position. Only after the size is set is the center X/Y position constrained —
        # using crop.center_bounds/crop_box, the exact same region-positioning logic as the
        # Crop tool, so a landmark region behaves identically and can never end up
        # off-image. This one-way dependency means resizing can nudge position back
        # on-image if it no longer fits, but repositioning never touches size.
        w = dynamic_bounded_slider("Crop region width", 8, max(8, rw), pending["w"], 1, f"lm_w_{step['id']}")
        h = dynamic_bounded_slider("Crop region height", 8, max(8, rh), pending["h"], 1, f"lm_h_{step['id']}")
        min_cx, max_cx = crop.center_bounds(rw, w)
        min_cy, max_cy = crop.center_bounds(rh, h)
        cx = dynamic_bounded_slider(
            "Region X position (center)", min_cx, max_cx, pending["cx"], 1, f"lm_cx_{step['id']}"
        )
        cy = dynamic_bounded_slider(
            "Region Y position (center)", min_cy, max_cy, pending["cy"], 1, f"lm_cy_{step['id']}"
        )
        st.session_state[pending_key] = {"w": w, "h": h, "cx": cx, "cy": cy}

    box_params = {"cx": cx, "cy": cy, "w": w, "h": h}
    left, top, w, h = crop.crop_box((rh, rw), box_params)
    with col1:
        # Same overlay function the Crop tool uses on its Input image preview, reused here
        # so a landmark region is outlined identically.
        st.image(crop.draw_overlay(ref_image, box_params), caption="Reference image", width="stretch")
    with col2:
        st.image(ref_image[top : top + h, left : left + w], caption="Landmark patch preview")
        if st.button("Add landmark"):
            # Landmarks are still stored top-left-corner-based (align.make_landmark's
            # existing signature, and every downstream match_landmarks/apply consumer) —
            # only this picker's UI is center-based, converted here at the point of adding.
            landmark = align.make_landmark(ref_image, left, top, w, h)
            step["params"]["landmarks"] = [*step["params"].get("landmarks", []), landmark]
            _save_draft(project, steps)
            st.success(f"Added landmark at ({left}, {top}), size {w}x{h}")

    landmarks = step["params"].get("landmarks", [])
    if landmarks:
        st.write(f"{len(landmarks)} landmark(s) defined:")
        for i, lm in enumerate(landmarks):
            lc1, lc2 = st.columns([4, 1])
            lc1.write(f"Landmark {i}: pos=({lm['x']}, {lm['y']}), size={lm['w']}x{lm['h']}")
            if lc2.button("Remove", key=f"rm_lm_{i}"):
                step["params"]["landmarks"] = [l for j, l in enumerate(landmarks) if j != i]
                _save_draft(project, steps)
                st.rerun()

    if sample_image is not None and landmarks:
        low_conf = align.low_confidence_landmarks(sample_image, step["params"])
        if low_conf:
            st.warning(f"Low-confidence match on landmark(s) {low_conf} for the current preview image.")


def _render_sidebar_chain(project, steps: list[dict]) -> int:
    """Renders the add/reorder/remove/select tool chain in the sidebar, which scrolls
    independently of the main content — so a long chain never forces the user to scroll
    down to settings after picking a step, then back up to pick the next one."""
    st.sidebar.subheader("Tool chain")
    new_type = st.sidebar.selectbox("Add a tool", list(TOOLS.keys()), format_func=lambda t: TOOL_LABELS[t])
    if st.sidebar.button("+ Add Step", width="stretch"):
        steps.append({"id": uuid.uuid4().hex[:8], "type": new_type, "params": get_tool(new_type).default_params()})
        st.session_state[SELECTED_KEY] = len(steps) - 1
        _save_draft(project, steps)
        st.rerun()

    if not steps:
        st.sidebar.info("No steps yet — add a tool above.")
        return 0

    warning = _known_risky_order_warning(steps)
    if warning:
        st.sidebar.warning(warning)

    st.sidebar.divider()
    selected = st.session_state.get(SELECTED_KEY, 0)
    selected = min(selected, len(steps) - 1)

    for i, step in enumerate(steps):
        up_col, down_col, del_col, label_col = st.sidebar.columns([1, 1, 1, 4])
        # Icons are passed via `icon=` (Streamlit's bundled Material Symbols font) rather
        # than as raw Unicode glyphs (↑ ↓ ✕) in the label — those depend on the button's
        # text font having a matching glyph, which isn't guaranteed and rendered blank here.
        with up_col:
            if st.button(
                "", icon=":material/keyboard_arrow_up:", key=f"up_{step['id']}", disabled=(i == 0), help="Move up"
            ):
                steps[i - 1], steps[i] = steps[i], steps[i - 1]
                st.session_state[SELECTED_KEY] = i - 1
                _save_draft(project, steps)
                st.rerun()
        with down_col:
            if st.button(
                "",
                icon=":material/keyboard_arrow_down:",
                key=f"down_{step['id']}",
                disabled=(i == len(steps) - 1),
                help="Move down",
            ):
                steps[i + 1], steps[i] = steps[i], steps[i + 1]
                st.session_state[SELECTED_KEY] = i + 1
                _save_draft(project, steps)
                st.rerun()
        with del_col:
            if st.button("", icon=":material/close:", key=f"del_{step['id']}", help="Remove step"):
                steps.pop(i)
                st.session_state[SELECTED_KEY] = max(0, i - 1)
                _save_draft(project, steps)
                st.rerun()
        with label_col:
            label = f"{i + 1}. {TOOL_LABELS[step['type']]}"
            if st.button(label, key=f"select_{step['id']}", type="primary" if i == selected else "secondary", width="stretch"):
                # Unlike the up/down/delete buttons above, this one used to update
                # `selected` without an st.rerun() — but this button's own `type=`
                # (primary/secondary) is computed and sent to the browser *before* Streamlit
                # even knows it was clicked, so the just-clicked step stayed unhighlighted
                # and the previously-selected one stayed highlighted until a second click
                # forced a fresh pass. Rerunning immediately, like its sibling buttons do,
                # makes the highlight correct on the very next render.
                st.session_state[SELECTED_KEY] = i
                st.rerun()

    return selected


def _render_saved_config_list(project) -> None:
    """Lists every applied preprocessing config in the sidebar, below the tool chain, so
    users can give each one a memorable nickname (the on-disk id stays the content hash —
    the nickname is a separate display-only label, see repository.set_config_name) or
    delete configs they no longer need."""
    st.sidebar.divider()
    st.sidebar.subheader("Saved preprocessing configs")

    config_files = sorted(project.preprocessing_configs_dir.glob("*.json"))
    if not config_files:
        st.sidebar.caption("None yet — build a chain above and click 'Apply to project'.")
        return

    names = repo.load_config_names(project)
    runs = repo.list_runs(project)

    for config_file in config_files:
        preproc_hash = config_file.stem
        nickname = names.get(preproc_hash, "")
        config_scope = PreprocessConfig.model_validate(json.loads(config_file.read_text())).scope
        with st.sidebar.expander(nickname or preproc_hash):
            st.caption(f"ID: `{preproc_hash}`")
            st.caption(f"Scope: {repo.format_scope(config_scope)}")
            new_name = st.text_input("Nickname", value=nickname, key=f"nickname_{preproc_hash}")
            if st.button("Save name", key=f"save_name_{preproc_hash}"):
                repo.set_config_name(project, preproc_hash, new_name)
                st.rerun()

            used_by = sum(1 for r in runs if r["config"].get("preproc_hash") == preproc_hash)
            if used_by:
                st.caption(f"⚠️ Used by {used_by} existing run(s) — deleting will break viewing them.")
            if st.button("Delete config", key=f"delete_config_{preproc_hash}"):
                pipeline.delete_config(project, preproc_hash)
                st.rerun()


def _render_scope_builder(project, steps: list[dict]) -> list[ScopeRule]:
    """Renders the rule list that decides which images this config (and training on it)
    uses: an image is in scope if it matches ANY rule, and a rule matches images carrying
    ALL of its first list and NONE of its second (see repository.categories_match_scope).
    Returns the effective scope, with empty rules dropped."""
    rules = _scope_rules(project)
    options = [*repo.list_categories(project), repo.UNTAGGED]
    all_ids = [img["id"] for img in repo.list_images(project)]
    labels = repo.get_all_labels(project)
    image_categories = {iid: labels.get(iid, {}).get("categories", []) for iid in all_ids}

    st.markdown("**Images this config applies to**")
    st.caption(
        "Add rules to specify what images to apply the config to. "
        "Within 1 rule, you can intersect categories. "
        "Across 2+ rules, you can union categories."
    )

    for i, rule in enumerate(rules):
        with st.container(border=True):
            # Explicit, rule-id-scoped keys (not derived from `default=`) so a widget's
            # identity survives reruns and deleting another rule — otherwise edits get
            # dropped on the first change, the same bug fixed for the preprocessing sliders.
            all_col, none_col, count_col, del_col = st.columns([4, 4, 1, 1], vertical_alignment="bottom")
            with all_col:
                all_of = st.multiselect(
                    "Intersect Categories",
                    options,
                    default=[c for c in rule["all_of"] if c in options],
                    key=f"scope_all_{rule['id']}",
                )
            with none_col:
                none_of = st.multiselect(
                    "Subtract Categories",
                    options,
                    default=[c for c in rule["none_of"] if c in options],
                    key=f"scope_none_{rule['id']}",
                )
            if all_of != rule["all_of"] or none_of != rule["none_of"]:
                rule["all_of"], rule["none_of"] = all_of, none_of
                _save_draft(project, steps)
            rule_scope = _to_scope([rule])
            with count_col:
                if rule_scope:
                    matched = sum(repo.categories_match_scope(c, rule_scope) for c in image_categories.values())
                    st.metric("Images", matched, label_visibility="collapsed")
                else:
                    st.caption("empty")
            with del_col:
                if st.button("", icon=":material/close:", key=f"scope_del_{rule['id']}", help="Remove rule"):
                    rules.pop(i)
                    _save_draft(project, steps)
                    st.rerun()

    if st.button("Add rule", icon=":material/add:"):
        rules.append(_new_rule())
        _save_draft(project, steps)
        st.rerun()

    scope = _to_scope(rules)
    matching = [iid for iid in all_ids if repo.categories_match_scope(image_categories[iid], scope)]
    st.caption(f"**{len(matching)} of {len(all_ids)} images** match: {repo.format_scope(scope)}")
    if scope and matching:
        with st.expander("Show matching images"):
            shown = matching[:MAX_SCOPE_THUMBNAILS]
            st.image(
                [str(thumbnails.get_thumbnail_path(project, iid, repo.image_path(project, iid))) for iid in shown],
                caption=shown,
                width=120,
            )
            if len(matching) > len(shown):
                st.caption(f"…and {len(matching) - len(shown)} more.")
    return scope


def render(project) -> dict | None:
    """Renders the full pipeline builder UI. Returns the build_cache report after 'Apply', else None.

    The saved-config sidebar list is rendered *last*, after 'Apply to project' has had a
    chance to run build_cache() earlier in this same pass — rendering it first (as
    originally written) meant a newly-applied config wouldn't show up until the next
    rerun, since the list would already have been drawn from the pre-build state of disk.
    """
    steps = _steps(project)
    selected = _render_sidebar_chain(project, steps)

    scope = _render_scope_builder(project, steps)

    report = None
    if not steps:
        st.info("No steps yet — add a tool in the sidebar to start building your preprocessing chain.")
    else:
        sample_id, sample_image = _sample_image(project, scope)
        if sample_image is None:
            if scope and repo.list_images(project):
                st.info(
                    "No images match the scope rules yet — assign categories on the Label "
                    "Images page, or remove the rules above to preview all images."
                )
            else:
                st.info("Import images to enable preview.")
        else:
            step = steps[selected]
            tool = get_tool(step["type"])

            st.subheader(f"Settings — {TOOL_LABELS[step['type']]} (step {selected + 1})")
            settings_col, input_col, output_col = st.columns([1, 1, 1])

            image_before = pipeline.apply_up_to(sample_image, [PipelineStep(**s) for s in steps], selected)

            with settings_col:
                new_params = tool.render_controls(
                    step["params"], context={"sample_image": image_before, "step_id": step["id"]}
                )
                if new_params != step["params"]:
                    step["params"] = new_params
                if step["type"] == "align":
                    _render_align_landmark_manager(project, step, steps, image_before)

            image_after = tool.apply(image_before, step["params"])
            with input_col:
                st.caption("Input to this step")
                if step["type"] == "crop":
                    # Outlines where the crop will land, using the exact same box-resolving
                    # logic apply() uses — so the overlay can never drift out of sync with
                    # what actually gets cropped. Overlay only ever affects this preview
                    # copy, never the real image_before/image_after data.
                    st.image(crop.draw_overlay(image_before, step["params"]), width="stretch")
                else:
                    st.image(image_before, width="stretch")
                st.caption(f"{image_before.shape[1]} × {image_before.shape[0]} px (width × height)")
            with output_col:
                st.caption("Output of this step")
                st.image(image_after, width="stretch")
                st.caption(f"{image_after.shape[1]} × {image_after.shape[0]} px (width × height)")

            # Catches param-only changes (sliders, landmark additions) that don't go
            # through an explicit st.rerun() above and so wouldn't otherwise hit a save point.
            _save_draft(project, steps)

            st.divider()
            # The nickname is scoped to this exact config's would-be hash (not a fixed
            # key), so the box shows that config's existing nickname (if it was already
            # applied before) rather than leftover text typed for a different chain — and
            # so an unrelated previous config's name is never blanked out by accident.
            # This is the same tool chain/hash regardless of which step type is currently
            # selected, so one nickname box covers every tool, not a per-tool control.
            pending_hash = stable_hash(
                PreprocessConfig(steps=[PipelineStep(**s) for s in steps], scope=scope).model_dump()
            )
            existing_nickname = repo.get_config_name(project, pending_hash) or ""
            nickname_key = f"pending_nickname_{pending_hash}"
            # The button is rendered above the nickname box (per user request), so its
            # on-click branch runs before the text_input() line below it in this same
            # script pass. That's fine — the key's value already lives in session_state
            # from the box's last render, same as any other keyed widget read back after
            # the fact — but it does mean we must read st.session_state[nickname_key]
            # directly here rather than a local variable the text_input call would return.
            if st.button("Apply to project", type="primary", width="stretch"):
                config = PreprocessConfig(steps=[PipelineStep(**s) for s in steps], scope=scope)
                report = pipeline.build_cache(project, config)
                repo.set_config_name(project, report["preproc_hash"], st.session_state.get(nickname_key, ""))
            st.text_input(
                "Nickname",
                value=existing_nickname,
                key=nickname_key,
                placeholder="e.g. fork-v1",
                help="Optional memorable name, assigned when you click Apply. Editable "
                "later from 'Saved preprocessing configs' in the sidebar.",
            )

    _render_saved_config_list(project)
    return report
