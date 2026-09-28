"""On-disk cache of downscaled gallery thumbnails, keyed by image id.

Source images under images/raw/ are imported once and never mutated (see
Project.images_raw_dir's callers) — so unlike the preprocessing cache, a thumbnail never
goes stale once built and needs no content-hash invalidation, just a one-time build keyed
on image_id.

JPEG (not PNG) is deliberate: these are gallery previews, not anything downstream needs
pixel-exact, and JPEG's lossy compression is far smaller than PNG for photographic content
at these dimensions — the whole point of this cache is to shrink what the Label Images
page re-renders on every click.
"""
from pathlib import Path

from PIL import Image

from image_processing.core.project import Project

MAX_DIMENSION = 320
JPEG_QUALITY = 80


def get_thumbnail_path(project: Project, image_id: str, source_path: Path) -> Path:
    """Returns a cached thumbnail's path, building it first if this is the first request."""
    thumb_path = project.thumbnails_dir / f"{image_id}.jpg"
    if not thumb_path.exists():
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(source_path).convert("RGB")
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))
        image.save(thumb_path, format="JPEG", quality=JPEG_QUALITY)
    return thumb_path
