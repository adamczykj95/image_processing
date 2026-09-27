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
from image_processing.core.config_schema import PipelineStep, PreprocessConfig
from image_processing.core.hashing import stable_hash
from image_processing.core.store import read_json, write_json
from image_processing.preprocessing import align, pipeline
from image_processing.preprocessing.registry import TOOL_LABELS, TOOLS, get_tool
from image_processing.preprocessing.ui_widgets import slider_with_input

STEPS_KEY = "pp_steps"
CATEGORIES_KEY = "pp_categories"
SELECTED_KEY = "pp_selected_index"
SAMPLE_KEY = "pp_sample_image_id"


def _load_draft(project) -> dict:
    return read_json(project.draft_pipeline_path, default={"steps": [], "categories": []})


def _save_draft(project, steps: list[dict]) -> None:
    # Reads categories from session_state rather than taking them as a parameter, so every
    # existing _save_draft(project, steps) call site (add/remove/reorder step, landmark
    # edits, param changes) keeps working unchanged — categories are only ever mutated by
    # their own multiselect in render(), never by any of those other call sites.
    categories = st.session_state.get(CATEGORIES_KEY, [])
    write_json(project.draft_pipeline_path, {"steps": steps, "categories": categories})


def _steps(project) -> list[dict]:
    if STEPS_KEY not in st.session_state:
        draft = _load_draft(project)
        st.session_state[STEPS_KEY] = draft["steps"]
        st.session_state[CATEGORIES_KEY] = draft.get("categories", [])
    return st.session_state[STEPS_KEY]


def _categories(project) -> list[str]:
    if CATEGORIES_KEY not in st.session_state:
        _steps(project)  # loads both keys together from the same draft file
    return st.session_state[CATEGORIES_KEY]


def _known_risky_order_warning(steps: list[dict]) -> str | None:
    types = [s["type"] for s in steps]
    if "align" in types:
        align_idx = types.index("align")
        if "crop" in types[:align_idx]:
            return "A crop step runs before align — it may crop away a needed landmark region."
        if "resize" in types[:align_idx]:
            return "A resize step runs before align — landmark coordinates were picked at a different scale."
    return None


def _sample_image(project, categories: list[str]) -> tuple[str | None, "np.ndarray | None"]:
    images = [
        img for img in repo.list_images(project) if repo.image_matches_categories(project, img["id"], categories)
    ]
    if not images:
        return None, None
    ids = [img["id"] for img in images]
    current = st.session_state.get(SAMPLE_KEY, ids[0])
    if current not in ids:
        current = ids[0]
    chosen = st.selectbox("Preview sample image", ids, index=ids.index(current), key=SAMPLE_KEY)
    image = pipeline.load_image(repo.image_path(project, chosen))
    return chosen, image


def _render_align_landmark_manager(project, step: dict, steps: list[dict], sample_image) -> None:
    """Mutates step['params']['landmarks'] directly (step is a live entry in `steps`,
    which is the same list object backing session_state) so removals/additions are never
    lost to a discarded local copy when a button handler triggers an immediate rerun."""
    st.markdown("**Manage landmarks**")
    images = repo.list_images(project)
    ids = [img["id"] for img in images]
    ref_id = st.selectbox("Reference image", ids, key="align_ref_image")
    ref_image = pipeline.load_image(repo.image_path(project, ref_id))
    rh, rw = ref_image.shape[:2]

    col1, col2 = st.columns(2)
    with col1:
        st.image(ref_image, caption="Reference image", width="stretch")
    with col2:
        # step["id"]-scoped prefixes so two separate align steps don't share landmark-picker
        # state, and slider_with_input for a linked text-entry box on each — see
        # preprocessing/crop.py's render_controls for why the key needs to be stable.
        #
        # Width/height are sized FIRST, against the full image dimensions — independent of
        # X/Y. Only after the size is set do X/Y get constrained (max = image size minus
        # the chosen width/height) so the box can't be dragged off-image. The previous
        # order did this backwards (width's max was `image width - x`), so moving X shrank
        # the max allowed width and Streamlit silently clamped an already-chosen width down
        # to fit — the box's *size* would change just from repositioning it. This way the
        # dependency only ever runs one direction: resizing can nudge position back
        # on-image if needed, but repositioning never touches size.
        w = slider_with_input("Landmark Width", 8, max(8, rw), max(8, min(40, rw)), 1, f"lm_w_{step['id']}")
        h = slider_with_input("Landmark Height", 8, max(8, rh), max(8, min(40, rh)), 1, f"lm_h_{step['id']}")
        x = slider_with_input(
            "Landmark X", 0, max(0, rw - w), min(rw // 4, max(0, rw - w)), 1, f"lm_x_{step['id']}"
        )
        y = slider_with_input(
            "Landmark Y", 0, max(0, rh - h), min(rh // 4, max(0, rh - h)), 1, f"lm_y_{step['id']}"
        )
        st.image(ref_image[y : y + h, x : x + w], caption="Landmark patch preview")
        if st.button("Add landmark"):
            landmark = align.make_landmark(ref_image, x, y, w, h)
            step["params"]["landmarks"] = [*step["params"].get("landmarks", []), landmark]
            _save_draft(project, steps)
            st.success(f"Added landmark at ({x}, {y}), size {w}x{h}")

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
        config_categories = json.loads(config_file.read_text()).get("categories", [])
        with st.sidebar.expander(nickname or preproc_hash):
            st.caption(f"ID: `{preproc_hash}`")
            st.caption(f"Categories: {', '.join(config_categories) if config_categories else '(all images)'}")
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


def render(project) -> dict | None:
    """Renders the full pipeline builder UI. Returns the build_cache report after 'Apply', else None.

    The saved-config sidebar list is rendered *last*, after 'Apply to project' has had a
    chance to run build_cache() earlier in this same pass — rendering it first (as
    originally written) meant a newly-applied config wouldn't show up until the next
    rerun, since the list would already have been drawn from the pre-build state of disk.
    """
    steps = _steps(project)
    categories = _categories(project)
    selected = _render_sidebar_chain(project, steps)

    all_categories = repo.list_categories(project)
    new_categories = st.multiselect(
        "Categories this config applies to",
        all_categories,
        default=[c for c in categories if c in all_categories],
        key="pp_categories_widget",
        help="Leave empty to apply this config to all images. Select one or more to scope "
        "it (and, later, training) to only images tagged with those categories — e.g. a "
        "separate preprocessing chain for each product type. Create categories on the "
        "Label Images page.",
    )
    if new_categories != categories:
        st.session_state[CATEGORIES_KEY] = new_categories
        _save_draft(project, steps)
        categories = new_categories

    report = None
    if not steps:
        st.info("No steps yet — add a tool in the sidebar to start building your preprocessing chain.")
    else:
        sample_id, sample_image = _sample_image(project, categories)
        if sample_image is None:
            if categories and repo.list_images(project):
                st.info(
                    "No images match the selected categories yet — assign categories on "
                    "the Label Images page, or clear the selection above to preview all images."
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
                st.image(image_before, width="stretch")
            with output_col:
                st.caption("Output of this step")
                st.image(image_after, width="stretch")

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
                PreprocessConfig(steps=[PipelineStep(**s) for s in steps], categories=categories).model_dump()
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
                config = PreprocessConfig(steps=[PipelineStep(**s) for s in steps], categories=categories)
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
