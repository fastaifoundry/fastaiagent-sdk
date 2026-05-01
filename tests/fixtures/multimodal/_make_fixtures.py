"""Generate multimodal test fixtures.

Idempotent: re-running overwrites with byte-identical output (modulo timestamps
in PDF metadata). Run from the repo root::

    python tests/fixtures/multimodal/_make_fixtures.py

Produces:
* ``cat.jpg``       — 200x200 JPEG with the literal text "CAT" rendered. Vision
                       LLMs reliably read the text, making e2e assertions
                       deterministic without committing a real photo.
* ``receipt.png``   — 600x800 PNG with "TOTAL $47.50" rendered as readable text.
* ``contract.pdf``  — 2-page PDF, one sentence per page, generated via pymupdf.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best-effort load of a system font; fall back to PIL's default bitmap."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def make_cat_jpg(out: Path) -> None:
    img = PILImage.new("RGB", (200, 200), color=(245, 222, 179))
    draw = ImageDraw.Draw(img)
    font = _load_font(72)
    draw.text((35, 60), "CAT", fill=(40, 40, 40), font=font)
    img.save(out, format="JPEG", quality=85)


def make_receipt_png(out: Path) -> None:
    img = PILImage.new("RGB", (600, 800), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_h = _load_font(36)
    font_b = _load_font(24)
    draw.text((40, 40), "ACME COFFEE", fill=(0, 0, 0), font=font_h)
    draw.text((40, 100), "Order #12345", fill=(0, 0, 0), font=font_b)
    draw.text((40, 160), "Espresso        $3.50", fill=(0, 0, 0), font=font_b)
    draw.text((40, 200), "Pastry          $4.00", fill=(0, 0, 0), font=font_b)
    draw.text((40, 240), "Sandwich       $40.00", fill=(0, 0, 0), font=font_b)
    draw.text((40, 320), "TOTAL $47.50", fill=(0, 0, 0), font=font_h)
    img.save(out, format="PNG")


def make_contract_pdf(out: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Page 1: This Service Agreement is entered into between Acme Corp and the Client.",
        fontsize=14,
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "Page 2: The agreement has a term of two years from the effective date.",
        fontsize=14,
    )
    doc.save(str(out))
    doc.close()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    make_cat_jpg(HERE / "cat.jpg")
    make_receipt_png(HERE / "receipt.png")
    make_contract_pdf(HERE / "contract.pdf")
    print(f"wrote fixtures to {HERE}")


if __name__ == "__main__":
    main()
