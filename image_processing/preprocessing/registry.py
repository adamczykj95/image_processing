"""Tool registry: maps a step type name to its apply/default_params/render_controls functions.

Adding a new preprocessing tool is additive: implement the three-function interface in
its own module and add one entry here. The pipeline executor never changes.
"""
from image_processing.preprocessing import align, brightness_contrast, color, crop, resize

TOOLS = {
    "align": align,
    "crop": crop,
    "resize": resize,
    "brightness_contrast": brightness_contrast,
    "color": color,
}

TOOL_LABELS = {
    "align": "Align",
    "crop": "Crop",
    "resize": "Resize",
    "brightness_contrast": "Brightness / Contrast",
    "color": "Color",
}


def get_tool(step_type: str):
    if step_type not in TOOLS:
        raise KeyError(f"Unknown preprocessing tool type: {step_type}")
    return TOOLS[step_type]
