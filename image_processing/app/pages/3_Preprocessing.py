import streamlit as st

from image_processing.app.components import pipeline_builder
from image_processing.app.state import require_project
from image_processing.app.style import inject_global_css

st.set_page_config(page_title="Preprocessing", page_icon="🧰", layout="wide")
inject_global_css()
project = require_project()

st.title("Preprocessing Pipeline Builder")
st.caption(
    "Build your preprocessing chain by adding tools in any order. Select a step to "
    "adjust its settings and see a before/after preview on a sample image."
)

report = pipeline_builder.render(project)

if report is not None:
    st.session_state["last_preproc_hash"] = report["preproc_hash"]
    if report["warnings"]:
        for w in report["warnings"]:
            st.warning(w)
    else:
        st.success(
            f"Cache built successfully. preproc_hash = `{report['preproc_hash']}`, "
            f"output size = {report['output_sizes']}"
        )
