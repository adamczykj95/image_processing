"""Shared app-wide CSS.

Streamlit rounds image corners by default, which crops a few pixels of the actual image
data out of view — unacceptable here since users need to see the full image (e.g. to spot
a defect right at an edge, or judge an alignment/crop box precisely). This strips that
rounding globally so every st.image() call in the app shows the complete, unclipped image.
"""
import streamlit as st

_CSS = """
<style>
img, [data-testid="stImage"] img, [data-testid="stImageContainer"] img {
    border-radius: 0 !important;
}
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
