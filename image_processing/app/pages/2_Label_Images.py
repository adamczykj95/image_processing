import random

import streamlit as st

from image_processing.app.state import require_project
from image_processing.core import repository as repo

st.set_page_config(page_title="Label Images", page_icon="🏷️", layout="wide")
project = require_project()

st.title("Label Images")

images = repo.list_images(project)
if not images:
    st.info("No images yet — import some on the Import Images page first.")
    st.stop()

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
                key=f"label_{img['id']}",
                horizontal=True,
            )
            split = st.selectbox(
                "Split",
                ["unassigned", "train", "val"],
                index=["unassigned", "train", "val"].index(label_entry["split"]),
                key=f"split_{img['id']}",
            )
            if label != label_entry["label"] or split != label_entry["split"]:
                repo.set_label(project, img["id"], label=label, split=split)
                st.rerun()
