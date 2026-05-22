# passport_photo

CLI that turns front-facing photos into print-ready A4 passport photo sheets.

- Detects a single face per input with MediaPipe Face Mesh.
- Crops to the ICAO/EU 35×45 mm spec (head ~34 mm tall, eyes ~30 mm from bottom).
- Tiles each face N times on A4 and packs multiple inputs onto the same page when there is room.
- Outputs a single multi-page PDF at 300 DPI.

## Install

```
pip install -r requirements.txt
```

## Use

```
python passport_photo.py INPUT [INPUT ...] -o out.pdf [--copies 4] [--dpi 300] [--debug]
```

- `INPUT` may be image files or directories (recurses one level, picks `.jpg/.jpeg/.png/.bmp/.webp/.tif/.tiff`).
- `--copies` repeats each face that many times (default 4; try 6, 8, etc.).
- A 5×6 grid (≈30 tiles) fits on one A4 at 35×45 mm, so multiple inputs typically share the same page.

Example:

```
python passport_photo.py ./photos -o sheet.pdf --copies 6
```

## Notes / out of scope

- Background is left untouched — strict ICAO submissions need a plain background; pre-process the photo first.
- No compliance check for expression, glasses, shadows, or lighting.
- US 2×2 inch / other specs: change the `PHOTO_W_MM`, `PHOTO_H_MM`, `HEAD_H_MM`, and `EYE_FROM_BOTTOM_MM` constants at the top of `passport_photo.py`.
- An input is skipped (with a warning) if it has zero or multiple faces, or the face is too close to an edge to satisfy the spec.

## Printing

Print the PDF at **100 % / actual size** (no fit-to-page) on A4. Each tile measures exactly 35×45 mm; cut along the gray guides.
