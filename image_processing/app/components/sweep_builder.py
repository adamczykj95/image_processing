"""Sweep grid definition UI + free-form-text parsing helpers.

Grid values are entered as free-form comma-separated text rather than fixed preset
checklists (explicit user choice) — more flexible, at the cost of needing clear parse
error surfacing rather than a widget that can't be malformed.
"""
import streamlit as st

from image_processing.core import repository as repo

VALID_LAYERS = {"layer1", "layer2", "layer3", "layer4"}
VALID_BACKBONES = ["wide_resnet50_2", "resnet18", "wide_resnet101_2"]


def parse_float_list(text: str) -> tuple[list[float], str | None]:
    text = text.strip()
    if not text:
        return [], "Enter at least one value."
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return [], f"'{part}' is not a valid number."
    if not values:
        return [], "Enter at least one value."
    return values, None


def parse_int_list(text: str, *, require_odd: bool = False) -> tuple[list[int], str | None]:
    text = text.strip()
    if not text:
        return [], "Enter at least one value."
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            return [], f"'{part}' is not a valid integer."
        if require_odd and value % 2 == 0:
            return [], f"{value} must be odd."
        values.append(value)
    if not values:
        return [], "Enter at least one value."
    return values, None


def parse_layer_sets(text: str) -> tuple[list[list[str]], str | None]:
    text = text.strip()
    if not text:
        return [], "Enter at least one layer set (one per line)."
    layer_sets = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        layers = [layer.strip() for layer in line.split(",") if layer.strip()]
        invalid = [layer for layer in layers if layer not in VALID_LAYERS]
        if invalid:
            return [], f"Unknown layer name(s) {invalid} in '{line}'. Valid: {sorted(VALID_LAYERS)}."
        if layers:
            layer_sets.append(layers)
    if not layer_sets:
        return [], "Enter at least one layer set (one per line)."
    return layer_sets, None


def render_grid_form(project) -> dict | None:
    """Renders the sweep grid definition form. Returns a dict of parsed grid dimensions
    plus the total combination count, or None if inputs have validation errors or no
    preprocessing configs exist yet to sweep over."""
    config_files = sorted(project.preprocessing_configs_dir.glob("*.json"))
    if not config_files:
        st.warning("No preprocessing config found yet — build and apply one on the Preprocessing page first.")
        return None
    hash_options = [f.stem for f in config_files]
    config_names = repo.load_config_names(project)

    st.subheader("Sweep grid")
    preproc_hashes = st.multiselect(
        "Preprocessing configs to sweep",
        hash_options,
        default=hash_options[:1],
        format_func=lambda h: config_names.get(h, h),
    )
    backbones = st.multiselect("Backbones", VALID_BACKBONES, default=["wide_resnet50_2"])
    layers_text = st.text_area(
        "Feature layer sets — one comma-separated set per line",
        value="layer2,layer3",
        help="Each line is one combination to try, e.g. 'layer2,layer3'.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        coreset_text = st.text_input("Coreset sampling ratios (comma-separated)", value="0.1")
    with c2:
        neighbors_text = st.text_input("Num neighbors (comma-separated)", value="9")
    with c3:
        patch_text = st.text_input("Patch sizes, odd only (comma-separated)", value="3")

    errors: list[str] = []
    if not preproc_hashes:
        errors.append("Select at least one preprocessing config.")
    if not backbones:
        errors.append("Select at least one backbone.")

    layer_sets, err = parse_layer_sets(layers_text)
    if err:
        errors.append(f"Layers: {err}")
    coreset_ratios, err = parse_float_list(coreset_text)
    if err:
        errors.append(f"Coreset ratios: {err}")
    num_neighbors_list, err = parse_int_list(neighbors_text)
    if err:
        errors.append(f"Num neighbors: {err}")
    patch_sizes, err = parse_int_list(patch_text, require_odd=True)
    if err:
        errors.append(f"Patch sizes: {err}")

    for e in errors:
        st.error(e)
    if errors:
        return None

    total = (
        len(preproc_hashes)
        * len(backbones)
        * len(layer_sets)
        * len(coreset_ratios)
        * len(num_neighbors_list)
        * len(patch_sizes)
    )
    st.info(f"This sweep will run **{total}** combination(s).")
    if total > 50:
        st.warning(
            f"{total} combinations is a lot — each one trains and validates a full "
            "PatchCore model. Consider narrowing the grid before starting."
        )

    return {
        "preproc_hashes": preproc_hashes,
        "backbones": backbones,
        "layer_sets": layer_sets,
        "coreset_ratios": coreset_ratios,
        "num_neighbors_list": num_neighbors_list,
        "patch_sizes": patch_sizes,
        "total": total,
    }
