"""Detect faces in input photos, crop to ICAO 35x45mm passport spec,
and tile them on A4 pages (one packed PDF for all inputs).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ICAO 35x45 mm photo spec.
PHOTO_W_MM = 35.0
PHOTO_H_MM = 45.0
HEAD_H_MM = 34.0          # target chin-to-crown height within the tile
EYE_FROM_BOTTOM_MM = 30.0  # eye line measured from bottom of the tile

# A4 portrait.
A4_W_MM = 210.0
A4_H_MM = 297.0
PAGE_MARGIN_MM = 5.0
TILE_GUTTER_MM = 2.0

# MediaPipe Face Mesh landmark indices (same for Tasks API FaceLandmarker).
LM_CHIN = 152
LM_FOREHEAD = 10
LM_NOSE = 1
LM_EYE_R = 33   # right eye outer corner
LM_EYE_L = 263  # left eye outer corner

# Crown sits above forehead landmark; extrapolate by this fraction of (chin-forehead).
CROWN_EXTRAPOLATION = 0.25

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"


def ensure_model() -> Path:
    if not MODEL_CACHE.exists():
        MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {MODEL_CACHE} …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
        print("Download complete.")
    return MODEL_CACHE


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


@dataclass
class TileLayout:
    dpi: int
    tile_w: int
    tile_h: int
    head_h: int
    eye_from_bottom: int
    page_w: int
    page_h: int
    margin: int
    gutter: int
    cols: int
    rows: int

    @classmethod
    def for_dpi(cls, dpi: int) -> "TileLayout":
        tile_w = mm_to_px(PHOTO_W_MM, dpi)
        tile_h = mm_to_px(PHOTO_H_MM, dpi)
        head_h = mm_to_px(HEAD_H_MM, dpi)
        eye_from_bottom = mm_to_px(EYE_FROM_BOTTOM_MM, dpi)
        page_w = mm_to_px(A4_W_MM, dpi)
        page_h = mm_to_px(A4_H_MM, dpi)
        margin = mm_to_px(PAGE_MARGIN_MM, dpi)
        gutter = mm_to_px(TILE_GUTTER_MM, dpi)
        usable_w = page_w - 2 * margin
        usable_h = page_h - 2 * margin
        cols = max(1, (usable_w + gutter) // (tile_w + gutter))
        rows = max(1, (usable_h + gutter) // (tile_h + gutter))
        return cls(dpi, tile_w, tile_h, head_h, eye_from_bottom,
                   page_w, page_h, margin, gutter, int(cols), int(rows))

    @property
    def per_page(self) -> int:
        return self.cols * self.rows


def load_image_rgb(path: Path) -> Image.Image:
    """Load image, apply EXIF rotation, return PIL RGB."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def detect_landmarks(pil_img: Image.Image, landmarker) -> np.ndarray | None:
    """Return Nx3 array of (x_px, y_px, z) landmarks or None if no/many faces."""
    w, h = pil_img.size
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(pil_img))
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    if len(result.face_landmarks) > 1:
        return None
    lms = result.face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in lms], dtype=np.float32)
    return pts


def compute_crop_box(landmarks: np.ndarray, src_w: int, src_h: int,
                     layout: TileLayout) -> tuple[float, float, float, float] | None:
    """Compute (left, top, right, bottom) in source-image pixels.
    Returns None if the face is too close to an edge to satisfy the spec.
    """
    chin_y = landmarks[LM_CHIN, 1]
    forehead_y = landmarks[LM_FOREHEAD, 1]
    nose_x = landmarks[LM_NOSE, 0]
    eye_y = (landmarks[LM_EYE_R, 1] + landmarks[LM_EYE_L, 1]) / 2.0

    face_h = chin_y - forehead_y
    if face_h <= 0:
        return None
    crown_y = forehead_y - CROWN_EXTRAPOLATION * face_h
    head_h_src = chin_y - crown_y

    # Scale: head height in source -> head height in tile.
    scale = head_h_src / layout.head_h  # source px per tile px
    crop_w = layout.tile_w * scale
    crop_h = layout.tile_h * scale

    # Position: eye line at (tile_h - eye_from_bottom) from top of tile.
    eye_y_in_tile_px = layout.tile_h - layout.eye_from_bottom
    top = eye_y - eye_y_in_tile_px * scale
    left = nose_x - crop_w / 2.0
    right = left + crop_w
    bottom = top + crop_h

    if left < 0 or top < 0 or right > src_w or bottom > src_h:
        return None
    return (left, top, right, bottom)


def make_tile(src_rgb: Image.Image, box: tuple[float, float, float, float],
              layout: TileLayout) -> Image.Image:
    crop = src_rgb.crop(box)
    return crop.resize((layout.tile_w, layout.tile_h), Image.LANCZOS)


def render_page(tiles: list[Image.Image], layout: TileLayout) -> Image.Image:
    """Render a single A4 page with up to layout.per_page tiles, centered."""
    page = Image.new("RGB", (layout.page_w, layout.page_h), "white")
    n = len(tiles)
    if n == 0:
        return page
    cols = min(layout.cols, n)
    rows = (n + layout.cols - 1) // layout.cols
    grid_w = cols * layout.tile_w + (cols - 1) * layout.gutter
    grid_h = rows * layout.tile_h + (rows - 1) * layout.gutter
    x0 = (layout.page_w - grid_w) // 2
    y0 = (layout.page_h - grid_h) // 2

    draw = ImageDraw.Draw(page)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, layout.cols)
        x = x0 + c * (layout.tile_w + layout.gutter)
        y = y0 + r * (layout.tile_h + layout.gutter)
        page.paste(tile, (x, y))
        draw.rectangle([x, y, x + layout.tile_w - 1, y + layout.tile_h - 1],
                       outline=(180, 180, 180), width=1)
    return page


def iter_inputs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in IMAGE_EXTS and child.is_file():
                    out.append(child)
        elif p.is_file():
            out.append(p)
        else:
            print(f"WARN: input not found: {p}", file=sys.stderr)
    return out


def process_one(path: Path, layout: TileLayout, landmarker) -> Image.Image | None:
    try:
        pil = load_image_rgb(path)
    except Exception as exc:
        print(f"SKIP {path.name}: cannot read ({exc})", file=sys.stderr)
        return None
    lms = detect_landmarks(pil, landmarker)
    if lms is None:
        print(f"SKIP {path.name}: need exactly one face", file=sys.stderr)
        return None
    box = compute_crop_box(lms, pil.width, pil.height, layout)
    if box is None:
        print(f"SKIP {path.name}: face too close to edge for 35x45mm crop",
              file=sys.stderr)
        return None
    return make_tile(pil, box, layout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="Image files or directories")
    ap.add_argument("-o", "--output", type=Path, default=Path("passport_photos.pdf"))
    ap.add_argument("--copies", type=int, default=4,
                    help="Copies per face on the A4 sheet (default: 4)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--debug", action="store_true",
                    help="Also save individual tile PNGs next to the PDF")
    args = ap.parse_args(argv)

    if args.copies < 1:
        ap.error("--copies must be >= 1")

    layout = TileLayout.for_dpi(args.dpi)
    print(f"Layout: {layout.cols}x{layout.rows} = {layout.per_page} tiles/page "
          f"at {args.dpi} DPI")

    inputs = iter_inputs(args.inputs)
    if not inputs:
        print("No input images found.", file=sys.stderr)
        return 2

    model_path = ensure_model()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    tiles_stream: list[Image.Image] = []
    failures = 0
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for path in inputs:
            tile = process_one(path, layout, landmarker)
            if tile is None:
                failures += 1
                continue
            if args.debug:
                dbg = args.output.with_name(f"{args.output.stem}_{path.stem}.png")
                tile.save(dbg)
            tiles_stream.extend([tile] * args.copies)
            print(f"OK   {path.name}: +{args.copies} tiles")

    if not tiles_stream:
        print("No tiles produced.", file=sys.stderr)
        return 1

    pages: list[Image.Image] = []
    for i in range(0, len(tiles_stream), layout.per_page):
        chunk = tiles_stream[i:i + layout.per_page]
        pages.append(render_page(chunk, layout))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(args.output, save_all=True, append_images=pages[1:],
                  resolution=float(args.dpi), format="PDF")
    print(f"Wrote {args.output} ({len(pages)} page(s), {len(tiles_stream)} tile(s), "
          f"{failures} skipped)")
    return 0 if pages else 1


if __name__ == "__main__":
    sys.exit(main())
