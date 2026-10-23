from PIL import Image

import streamlit as st

from image_processing.app.components import image_select_grid
from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css
from image_processing.core import repository as repo
from image_processing.core import thumbnails

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

if flash := st.session_state.pop("import_flash", None):
    st.success(flash)

if not images:
    st.info("No images imported yet.")
    st.stop()


GRID_KEY = "import_grid"


def _set_selection(image_ids, value: bool) -> None:
    selection = set(image_select_grid.get_selection(GRID_KEY))
    image_select_grid.set_selection(GRID_KEY, selection | set(image_ids) if value else selection - set(image_ids))


def _invert_selection(image_ids) -> None:
    image_select_grid.set_selection(GRID_KEY, image_select_grid.get_selection(GRID_KEY) ^ set(image_ids))


all_labels = repo.get_all_labels(project)
filter_choice = st.radio("Show", ["All", "Unlabeled", "Good", "Anomaly"], horizontal=True, key="import_filter")
filtered = [
    img
    for img in images
    if filter_choice == "All" or all_labels.get(img["id"], {}).get("label", "unlabeled") == filter_choice.lower()
]
shown_ids = [img["id"] for img in filtered]
# Counted across *all* images, not just those shown — a selection survives changing the
# filter, and the delete button must say exactly how many images it will remove.
selection = image_select_grid.get_selection(GRID_KEY)
selected_ids = [img["id"] for img in images if img["id"] in selection]


CONFIRM_KEY = "import_pending_delete"


def _delete_pending() -> None:
    image_ids = st.session_state.pop(CONFIRM_KEY, [])
    deleted = repo.delete_images(project, image_ids)
    _set_selection(image_ids, False)
    st.session_state["import_flash"] = f"Deleted {deleted} image(s)."


# Always rendered (only `disabled` changes) to avoid the layout reflow flash described in
# Label Images' bulk-actions section.
all_col, none_col, invert_col, delete_col = st.columns(4)
all_col.button("Select all shown", width="stretch", on_click=_set_selection, args=(shown_ids, True))
none_col.button("Select none", width="stretch", on_click=_set_selection, args=([img["id"] for img in images], False))
invert_col.button("Invert shown", width="stretch", on_click=_invert_selection, args=(shown_ids,))
if delete_col.button(
    f"Delete {len(selected_ids)} selected",
    type="primary",
    icon=":material/delete:",
    width="stretch",
    disabled=not selected_ids,
):
    # Snapshot the ids now: the confirm step deletes exactly what was selected when
    # Delete was clicked, even if the selection changes before confirming.
    st.session_state[CONFIRM_KEY] = selected_ids

if pending := st.session_state.get(CONFIRM_KEY):
    with st.container(border=True):
        st.warning(
            f"Permanently delete **{len(pending)}** image(s)? This removes their files, labels, "
            "categories, and processed copies in every preprocessing config. It cannot be undone."
        )
        st.caption("Existing training runs keep their recorded metrics; deleted images just no longer appear in their results.")
        confirm_col, cancel_col = st.columns(2)
        confirm_col.button(f"Yes, delete {len(pending)}", type="primary", width="stretch", on_click=_delete_pending)
        cancel_col.button("Cancel", width="stretch", on_click=st.session_state.pop, args=(CONFIRM_KEY, None))

st.write(f"{len(filtered)} shown · **{len(selected_ids)}** selected")

st.caption("Click an image to select it; Shift+click another to select everything in between.")
image_select_grid.render(
    GRID_KEY,
    [
        (
            img["id"],
            thumbnails.get_thumbnail_path(project, img["id"], project.images_raw_dir / img["relpath"]),
            f"{img['relpath']} · {all_labels.get(img['id'], {}).get('label', 'unlabeled')}",
        )
        for img in filtered
    ],
)
