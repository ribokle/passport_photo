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

# Crop mode presets — margins inside the tile (mm above crown / mm below chin).
# "passport": tight head crop.
# "id": head + shoulders (head fills ~56% of tile, ~19mm below chin).
MODES: dict[str, dict[str, float]] = {
    "passport": {"top_mm": 1.0, "bottom_mm": 7.0},
    "id":       {"top_mm": 1.0, "bottom_mm": 19.0},
}

# A4 portrait.
A4_W_MM = 210.0
A4_H_MM = 297.0
PAGE_MARGIN_MM = 5.0
TILE_GUTTER_MM = 2.0

# MediaPipe Face Mesh landmark indices (same for Tasks API FaceLandmarker).
LM_CHIN = 152
LM_FOREHEAD = 10
LM_EYE_R = 33   # right eye outer corner
LM_EYE_L = 263  # left eye outer corner

# Crown sits above forehead landmark (lm 10 = hairline area).
# Extrapolate upward by this fraction of (chin-to-forehead) distance to estimate hair crown.
# Increase if the top of the head is still being cut; decrease for shaved/very short hair.
CROWN_EXTRAPOLATION = 0.65

# If the computed crop top lands within this many source pixels of the image top,
# snap to the image edge so no hair is ever missed by a sliver.
SNAP_TO_EDGE_PX = 30

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
    top_margin: int    # px above estimated crown
    bottom_margin: int # px below chin
    page_w: int
    page_h: int
    margin: int
    gutter: int
    cols: int
    rows: int

    @classmethod
    def for_dpi(cls, dpi: int, top_mm: float = 1.0, bottom_mm: float = 7.0) -> "TileLayout":
        tile_w = mm_to_px(PHOTO_W_MM, dpi)
        tile_h = mm_to_px(PHOTO_H_MM, dpi)
        top_margin = mm_to_px(top_mm, dpi)
        bottom_margin = mm_to_px(bottom_mm, dpi)
        page_w = mm_to_px(A4_W_MM, dpi)
        page_h = mm_to_px(A4_H_MM, dpi)
        margin = mm_to_px(PAGE_MARGIN_MM, dpi)
        gutter = mm_to_px(TILE_GUTTER_MM, dpi)
        usable_w = page_w - 2 * margin
        usable_h = page_h - 2 * margin
        cols = max(1, (usable_w + gutter) // (tile_w + gutter))
        rows = max(1, (usable_h + gutter) // (tile_h + gutter))
        return cls(dpi, tile_w, tile_h, top_margin, bottom_margin,
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
                     layout: TileLayout,
                     crown_extrapolation: float = CROWN_EXTRAPOLATION,
                     ) -> tuple[float, float, float, float] | None:
    """Compute (left, top, right, bottom) in source-image pixels.

    Anchors the crop on the estimated crown (top) and chin (bottom) so the full
    head is always visible.  Horizontal centre comes from the face bounding box
    so left and right margins are equal relative to the face edges.
    Overflows beyond the image edge are handled with white padding in make_tile.
    """
    chin_y = float(landmarks[LM_CHIN, 1])
    forehead_y = float(landmarks[LM_FOREHEAD, 1])
    face_h = chin_y - forehead_y
    if face_h <= 0:
        return None

    # Estimate crown: hair sits above the forehead (hairline) landmark.
    crown_y = forehead_y - crown_extrapolation * face_h  # may go above image top

    # Scale: map crown-to-chin (head height) to the tile height minus margins.
    head_h_src = chin_y - crown_y
    head_h_tile = layout.tile_h - layout.top_margin - layout.bottom_margin
    if head_h_tile <= 0:
        return None
    scale = head_h_src / head_h_tile  # source px per tile px

    crop_w = layout.tile_w * scale
    crop_h = layout.tile_h * scale

    # Vertical: crown lands at top_margin from tile top.
    top = crown_y - layout.top_margin * scale

    # Snap to image edge if the crop top is only a sliver away — avoids
    # cutting a few pixels of hair when the face is close to the frame top.
    if 0 < top < SNAP_TO_EDGE_PX:
        top = 0.0

    bottom = top + crop_h

    # Horizontal: face bounding-box centre gives equal margins either side.
    center_x = float((landmarks[:, 0].min() + landmarks[:, 0].max()) / 2.0)
    left = center_x - crop_w / 2.0
    right = left + crop_w

    # Only reject if the bottom (chin + margin) would go below the image.
    if bottom > src_h + src_h * 0.05:   # 5% tolerance for slight chin overshoot
        return None
    return (left, top, right, bottom)


def make_tile(src_rgb: Image.Image, box: tuple[float, float, float, float],
              layout: TileLayout) -> Image.Image:
    left, top, right, bottom = box
    src_w, src_h = src_rgb.size

    # Pad with white wherever the crop box extends outside the image.
    pad_left  = max(0.0, -left)
    pad_top   = max(0.0, -top)
    pad_right = max(0.0, right - src_w)
    pad_bot   = max(0.0, bottom - src_h)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bot > 0:
        new_w = int(src_w + pad_left + pad_right)
        new_h = int(src_h + pad_top + pad_bot)
        padded = Image.new("RGB", (new_w, new_h), "white")
        padded.paste(src_rgb, (int(pad_left), int(pad_top)))
        shifted = (left + pad_left, top + pad_top,
                   right + pad_left, bottom + pad_top)
        crop = padded.crop(shifted)
    else:
        crop = src_rgb.crop((left, top, right, bottom))

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
    x0 = (layout.page_w - grid_w) // 2  # still centred horizontally
    y0 = layout.margin                   # anchored to top margin

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


def process_one(path: Path, layout: TileLayout, landmarker,
                crown_extrapolation: float = CROWN_EXTRAPOLATION) -> Image.Image | None:
    try:
        pil = load_image_rgb(path)
    except Exception as exc:
        print(f"SKIP {path.name}: cannot read ({exc})", file=sys.stderr)
        return None
    lms = detect_landmarks(pil, landmarker)
    if lms is None:
        print(f"SKIP {path.name}: need exactly one face", file=sys.stderr)
        return None
    box = compute_crop_box(lms, pil.width, pil.height, layout,
                           crown_extrapolation=crown_extrapolation)
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
    ap.add_argument("--mode", choices=list(MODES), default="passport",
                    help="Crop preset: 'passport' (head only) or 'id' (head + shoulders)")
    ap.add_argument("--crown", type=float, default=CROWN_EXTRAPOLATION, metavar="FLOAT",
                    help=f"Crown extrapolation factor (default {CROWN_EXTRAPOLATION}). "
                         "Fraction of face height added above the forehead landmark to "
                         "estimate the hair crown. Increase (e.g. 0.8) if the top of "
                         "the head is still cut; decrease (e.g. 0.4) for shaved/very "
                         "short hair.")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--debug", action="store_true",
                    help="Also save individual tile PNGs next to the PDF")
    args = ap.parse_args(argv)

    if args.copies < 1:
        ap.error("--copies must be >= 1")
    if not 0.0 <= args.crown <= 2.0:
        ap.error("--crown must be between 0.0 and 2.0")

    margins = MODES[args.mode]
    layout = TileLayout.for_dpi(args.dpi, top_mm=margins["top_mm"], bottom_mm=margins["bottom_mm"])
    print(f"Mode: {args.mode}  |  Layout: {layout.cols}x{layout.rows} = {layout.per_page} tiles/page "
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
            tile = process_one(path, layout, landmarker,
                               crown_extrapolation=args.crown)
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
