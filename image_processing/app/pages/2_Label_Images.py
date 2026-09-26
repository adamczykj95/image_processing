import random

import streamlit as st

from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css
from image_processing.core import repository as repo

st.set_page_config(page_title="Label Images", page_icon="🏷️", layout="wide")
inject_global_css()
project = require_project()

st.title("Label Images")

images = repo.list_images(project)
if not images:
    st.info("No images yet — import some on the Import Images page first.")
    st.stop()

# Bumped whenever a bulk operation (auto-assign split, bulk category assign, or deleting a
# category) rewrites data out from under the per-image widgets below. Baked into their
# `key`s so Streamlit treats them as fresh widgets on the next render instead of reusing a
# stale cached value and — since the gallery loop's change-detection can't tell "stale
# cache" from "user edited it" — silently writing that stale value straight back over the
# bulk update in the same rerun. Deleting a category additionally *requires* this: a
# multiselect whose cached value references an option no longer in its options list raises
# an error, not just a display glitch.
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

filtered = []
for img in images:
    label_entry = repo.get_label(project, img["id"])
    if filter_choice == "All" or filter_choice.lower() == label_entry["label"] or (
        filter_choice == "Unlabeled" and label_entry["label"] == "unlabeled"
    ):
        filtered.append((img, label_entry))

if all_categories:
    bulk_cat_col, bulk_btn_col = st.columns([3, 1])
    with bulk_cat_col:
        bulk_category = st.selectbox("Bulk-assign category to images shown below", all_categories)
    with bulk_btn_col:
        st.write("")
        if st.button(f"Add to {len(filtered)} shown image(s)", width="stretch"):
            repo.add_category_to_images(project, bulk_category, [img["id"] for img, _ in filtered])
            st.session_state["label_gen"] += 1
            st.success(f"Added '{bulk_category}' to {len(filtered)} image(s).")
            st.rerun()

st.write(f"{len(filtered)} image(s)")

cols_per_row = 4
for row_start in range(0, len(filtered), cols_per_row):
    cols = st.columns(cols_per_row)
    for col, (img, label_entry) in zip(cols, filtered[row_start : row_start + cols_per_row]):
        with col:
            st.image(str(project.images_raw_dir / img["relpath"]), width="stretch")
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
            if label != label_entry["label"] or split != label_entry["split"]:
                repo.set_label(project, img["id"], label=label, split=split)
                st.rerun()
            if categories_selected != label_entry["categories"]:
                repo.set_image_categories(project, img["id"], categories_selected)
                st.rerun()
