from PIL import Image

import streamlit as st

from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css
from image_processing.core import repository as repo

st.set_page_config(page_title="Import Images", page_icon="📥", layout="wide")
inject_global_css()
project = require_project()

st.title("Import Images")

# Bumped after a successful import so the file_uploader below gets a fresh `key` on the
# next render — Streamlit otherwise keeps the previously-selected files in the widget
# indefinitely (its state is tied to the key, not to whether they've been processed), so
# the only way to make the upload tray clear itself is to make it a new widget instance.
uploader_gen = st.session_state.setdefault("uploader_gen", 0)

uploaded = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    key=f"uploader_{uploader_gen}",
)

if uploaded and st.button("Import selected files", type="primary"):
    imported = 0
    for file in uploaded:
        image = Image.open(file).convert("RGB")
        image_id = project.new_id()
        ext = "png"
        relpath = f"{image_id}.{ext}"
        image.save(project.images_raw_dir / relpath)
        repo.add_image(project, image_id, relpath, image.width, image.height)
        imported += 1
    st.session_state["uploader_gen"] += 1
    st.success(f"Imported {imported} image(s).")
    st.rerun()

st.divider()
images = repo.list_images(project)
st.subheader(f"Project images ({len(images)})")

if images:
    cols = st.columns(6)
    for i, img in enumerate(images):
        with cols[i % 6]:
            st.image(str(project.images_raw_dir / img["relpath"]), width="stretch")
            label = repo.get_label(project, img["id"])["label"]
            st.caption(f"{img['relpath']} · {label}")
else:
    st.info("No images imported yet.")
