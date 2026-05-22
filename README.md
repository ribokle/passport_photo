# passport_photo

CLI that turns front-facing photos into print-ready A4 passport photo sheets.

- Detects a single face per input with MediaPipe Face Mesh.
- Crops to ICAO/EU 35×45 mm (passport mode) or a zoomed-out head+shoulders view (id mode).
- Tiles each face N times on A4 and packs multiple inputs onto the same page when there is room.
- Outputs a single multi-page PDF at 300 DPI, ready to print and cut.

## Install

```bash
pip install -r requirements.txt
```

The MediaPipe face landmarker model (~3.5 MB) is downloaded automatically on first run.

## Quick start

Drop your photos into `input_photos/` and run:

```bash
python passport_photo.py input_photos/ -o sheet.pdf
```

## Examples

**Single photo, 4 copies (default):**
```bash
python passport_photo.py input_photos/photo.jpg -o sheet.pdf
```

**Single photo, 6 copies:**
```bash
python passport_photo.py input_photos/photo.jpg -o sheet.pdf --copies 6
```

**Multiple photos packed onto the same sheet:**
```bash
python passport_photo.py input_photos/alice.jpg input_photos/bob.jpg -o sheet.pdf
```

**Whole directory, 4 copies each:**
```bash
python passport_photo.py input_photos/ -o sheet.pdf
```

**Head + shoulders mode (ID card / visa style):**
```bash
python passport_photo.py input_photos/ -o sheet.pdf --mode id
```

**Passport mode explicitly stated:**
```bash
python passport_photo.py input_photos/ -o sheet.pdf --mode passport
```

**Higher DPI (e.g. 600) for sharper prints:**
```bash
python passport_photo.py input_photos/ -o sheet.pdf --dpi 600
```

**Debug: also save each cropped tile as a PNG for inspection:**
```bash
python passport_photo.py input_photos/ -o sheet.pdf --debug
```

## All options

| Option | Default | Description |
|---|---|---|
| `INPUT` | — | Image files or directories (`.jpg .jpeg .png .bmp .webp .tif .tiff`) |
| `-o / --output` | `passport_photos.pdf` | Output PDF path |
| `--mode` | `passport` | `passport` — tight head crop; `id` — head + shoulders |
| `--copies` | `4` | How many times to repeat each face on the sheet |
| `--dpi` | `300` | Print resolution (300 is standard; 600 for premium prints) |
| `--debug` | off | Save individual cropped tile PNGs alongside the PDF |

## Modes

| Mode | Tile size | Head height | Below chin |
|---|---|---|---|
| `passport` | 35×45 mm | ~37 mm (82 % of tile) | 7 mm |
| `id` | 35×45 mm | ~25 mm (56 % of tile) | 19 mm (shoulders visible) |

## Sheet layout

Photos are arranged in a 5×6 grid (up to 30 tiles per A4 page at 300 DPI) and aligned to the top of the sheet. Multiple inputs are packed together; a new page is only started when the current one is full.

## Printing

Print the PDF at **100 % / actual size** (disable "fit to page") on A4. Each tile is exactly 35×45 mm; cut along the gray guides.

## Notes

- Background is left untouched — ICAO strictly requires a plain light background; pre-process the source photo if needed.
- No compliance checks for expression, glasses, shadows, or lighting.
- An input is skipped (with a warning) if it contains zero or multiple faces.
- To adapt to other specs (e.g. US 2×2 inch), edit `PHOTO_W_MM`, `PHOTO_H_MM`, and the `MODES` dict at the top of `passport_photo.py`.
