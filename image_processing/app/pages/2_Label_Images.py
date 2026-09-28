import random

import streamlit as st

from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css
from image_processing.core import repository as repo
from image_processing.core import thumbnails

st.set_page_config(page_title="Label Images", page_icon="🏷️", layout="wide")
inject_global_css()
project = require_project()

st.title("Label Images")

images = repo.list_images(project)
if not images:
    st.info("No images yet — import some on the Import Images page first.")
    st.stop()


def _selection_key(image_id: str) -> str:
    return f"select_{image_id}"


def _is_selected(image_id: str) -> bool:
    return st.session_state.get(_selection_key(image_id), False)


def _clear_selection() -> None:
    # Direct session_state writes (no value= passed to the checkbox itself) rather than
    # relying on a value/key combination — same pattern as the linked slider/number-input
    # widgets elsewhere in this app, and for the same reason: a widget that takes both a
    # `value=` derived from external state AND a fixed `key=` fights its own cached state
    # instead of being reliably programmatically controlled between reruns.
    for img in images:
        st.session_state[_selection_key(img["id"])] = False


# Bumped whenever a bulk operation (auto-assign split, bulk label/split/category assign, or
# deleting a category) rewrites data out from under the per-image widgets below. Baked into
# their `key`s so Streamlit treats them as fresh widgets on the next render instead of
# reusing a stale cached value and — since the gallery loop's change-detection can't tell
# "stale cache" from "user edited it" — silently writing that stale value straight back over
# the bulk update in the same rerun. Deleting a category additionally *requires* this: a
# multiselect whose cached value references an option no longer in its options list raises
# an error, not just a display glitch. Selection checkboxes don't need this — they're pure
# UI state independent of label/split/category data, and are cleared via direct
# session_state writes (_clear_selection) rather than key-rotation.
generation = st.session_state.setdefault("label_gen", 0)

st.subheader("Image categories")
st.caption(
    "Optional tags independent of good/anomaly — e.g. distinguish product types (forks "
    "vs. spoons) so each can get its own preprocessing config and model."
)
all_categories = repo.list_categories(project)

new_cat_col, add_cat_col = st.columns([3, 1])
with new_cat_col:
    new_category_name = st.text_input(
        "New category name", key="new_category_name", label_visibility="collapsed", placeholder="e.g. fork"
    )
with add_cat_col:
    if st.button("Add category", width="stretch"):
        if new_category_name.strip():
            repo.create_category(project, new_category_name)
            st.rerun()

if all_categories:
    for category in all_categories:
        count = repo.count_images_with_category(project, category)
        cat_col, count_col, del_col = st.columns([3, 2, 1])
        cat_col.write(f"**{category}**")
        count_col.caption(f"{count} image(s)")
        if del_col.button("Delete", key=f"delete_category_{category}"):
            repo.delete_category(project, category)
            st.session_state["label_gen"] += 1
            st.rerun()
else:
    st.caption("No categories yet — add one above.")

st.divider()

st.subheader("Quick train/val split")
st.caption(
    "PatchCore trains on normal images only. A common split: most 'good' images go to "
    "train, the rest of 'good' plus all 'anomaly' images go to val."
)
train_fraction = st.slider("Fraction of 'good' images assigned to train", 0.1, 0.95, 0.8, step=0.05)
if st.button("Auto-assign split from current labels"):
    good_ids = repo.list_by_label(project, "good")
    anomaly_ids = repo.list_by_label(project, "anomaly")
    random.Random(0).shuffle(good_ids)
    n_train = max(1, int(len(good_ids) * train_fraction)) if good_ids else 0
    for iid in good_ids[:n_train]:
        repo.set_label(project, iid, split="train")
    for iid in good_ids[n_train:]:
        repo.set_label(project, iid, split="val")
    for iid in anomaly_ids:
        repo.set_label(project, iid, split="val")
    st.session_state["label_gen"] += 1
    st.success(f"Assigned {n_train} good images to train, the rest to val.")
    st.rerun()

st.divider()

filter_choice = st.radio(
    "Show", ["All", "Unlabeled", "Good", "Anomaly"], horizontal=True
)

# One catalog read for the whole page, not one per image (get_label() reloads the whole
# file from disk on every call — fine for a single lookup, expensive called in a loop
# over every image shown here, and the main reason each rerun visibly took a while).
all_labels = repo.get_all_labels(project)

_DEFAULT_LABEL_ENTRY = {"label": "unlabeled", "split": "unassigned", "labeled_at": None, "notes": "", "categories": []}

filtered = []
for img in images:
    label_entry = all_labels.get(img["id"], _DEFAULT_LABEL_ENTRY)
    if filter_choice == "All" or filter_choice.lower() == label_entry["label"] or (
        filter_choice == "Unlabeled" and label_entry["label"] == "unlabeled"
    ):
        filtered.append((img, label_entry))

st.write(f"{len(filtered)} image(s)")

st.subheader("Bulk actions")
st.caption(
    "Check the images you want below (or use the buttons here to select in bulk), then "
    "apply a label, split, or category to all of them at once. Much faster than clicking "
    "through every image individually once you're past a handful of them."
)

selected_ids = [img["id"] for img in images if _is_selected(img["id"])]

select_all_col, select_none_col, count_col = st.columns([1, 1, 2])
with select_all_col:
    if st.button("Select all shown", width="stretch"):
        for img, _ in filtered:
            st.session_state[_selection_key(img["id"])] = True
        st.rerun()
with select_none_col:
    if st.button("Select none", width="stretch"):
        _clear_selection()
        st.rerun()
with count_col:
    st.write(f"**{len(selected_ids)}** image(s) currently selected")

# These rows are always rendered (never conditionally inserted/removed based on
# selected_ids) — only their `disabled` state changes. Streamlit reflows the whole layout
# below whenever elements are inserted or removed mid-page, which caused a visible flash
# (briefly oversized images under the "running" overlay) the instant selection went from
# empty to non-empty, and again after a bulk-apply cleared it back to empty. Keeping the
# element structure constant across reruns avoids that reflow entirely.
bulk_label_col, bulk_label_btn_col = st.columns([3, 1])
with bulk_label_col:
    bulk_label = st.selectbox(
        "Set label for selected", ["unlabeled", "good", "anomaly"], key="bulk_label_choice"
    )
with bulk_label_btn_col:
    st.write("")
    if st.button(
        f"Apply to {len(selected_ids)}", key="bulk_apply_label", width="stretch", disabled=not selected_ids
    ):
        repo.apply_label_updates(project, {iid: {"label": bulk_label} for iid in selected_ids})
        _clear_selection()
        st.session_state["label_gen"] += 1
        st.success(f"Set label to '{bulk_label}' for {len(selected_ids)} image(s).")
        st.rerun()

bulk_split_col, bulk_split_btn_col = st.columns([3, 1])
with bulk_split_col:
    bulk_split = st.selectbox(
        "Set split for selected", ["unassigned", "train", "val"], key="bulk_split_choice"
    )
with bulk_split_btn_col:
    st.write("")
    if st.button(
        f"Apply to {len(selected_ids)}", key="bulk_apply_split", width="stretch", disabled=not selected_ids
    ):
        repo.apply_label_updates(project, {iid: {"split": bulk_split} for iid in selected_ids})
        _clear_selection()
        st.session_state["label_gen"] += 1
        st.success(f"Set split to '{bulk_split}' for {len(selected_ids)} image(s).")
        st.rerun()

if all_categories:
    bulk_cat_col, bulk_cat_btn_col = st.columns([3, 1])
    with bulk_cat_col:
        bulk_category = st.selectbox("Add category to selected", all_categories, key="bulk_cat_choice")
    with bulk_cat_btn_col:
        st.write("")
        if st.button(
            f"Add to {len(selected_ids)}", key="bulk_apply_category", width="stretch", disabled=not selected_ids
        ):
            repo.add_category_to_images(project, bulk_category, selected_ids)
            _clear_selection()
            st.session_state["label_gen"] += 1
            st.success(f"Added '{bulk_category}' to {len(selected_ids)} image(s).")
            st.rerun()

st.divider()

# Collected across the *entire* gallery in this one pass, then written as a single batch
# after the loop — not saved (and rerun) the moment the first changed widget is found.
# That earlier pattern meant only one image's edit was ever persisted per script run, so a
# batch of edits needed several uninterrupted reruns in a row to fully catch up, and a new
# click arriving before that catch-up finished would reset it, silently dropping whichever
# edits it hadn't reached yet. Doing it this way, any single rerun that completes — even if
# several before it were cancelled by rapid clicking — saves every pending edit at once.
pending_updates: dict[str, dict] = {}

cols_per_row = 4
for row_start in range(0, len(filtered), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, (img, label_entry) in zip(cols, filtered[row_start : row_start + cols_per_row]):
        with col:
            source_path = project.images_raw_dir / img["relpath"]
            thumb_path = thumbnails.get_thumbnail_path(project, img["id"], source_path)
            st.image(str(thumb_path), width="stretch")
            st.checkbox("Select", key=_selection_key(img["id"]))
            label = st.radio(
                "Label",
                ["unlabeled", "good", "anomaly"],
                index=["unlabeled", "good", "anomaly"].index(label_entry["label"]),
                key=f"label_{img['id']}_{generation}",
                horizontal=True,
            )
            split = st.selectbox(
                "Split",
                ["unassigned", "train", "val"],
                index=["unassigned", "train", "val"].index(label_entry["split"]),
                key=f"split_{img['id']}_{generation}",
            )
            categories_selected = st.multiselect(
                "Categories",
                all_categories,
                default=[c for c in label_entry["categories"] if c in all_categories],
                key=f"categories_{img['id']}_{generation}",
            )
            image_updates: dict = {}
            if label != label_entry["label"] or split != label_entry["split"]:
                image_updates["label"] = label
                image_updates["split"] = split
            if categories_selected != label_entry["categories"]:
                image_updates["categories"] = categories_selected
            if image_updates:
                pending_updates[img["id"]] = image_updates

if pending_updates:
    repo.apply_label_updates(project, pending_updates)
    st.rerun()
