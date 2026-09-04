"""Generate multimodal test fixtures.

Idempotent: re-running overwrites with byte-identical output (modulo timestamps
in PDF metadata). Run from the repo root::

    python tests/fixtures/multimodal/_make_fixtures.py

Produces:
* ``cat.jpg``       — 200x200 JPEG with the literal text "CAT" rendered. Vision
                       LLMs reliably read the text, making e2e assertions
                       deterministic without committing a real photo.
* ``receipt.png``   — 600x800 PNG with "TOTAL $47.50" rendered as readable text.
* ``contract.pdf``  — 2-page PDF, one sentence per page, written as raw PDF
                       syntax with no third-party dependency.
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


_CONTRACT_PAGES = [
    "Page 1: This Service Agreement is entered into between Acme Corp and the Client.",
    "Page 2: The agreement has a term of two years from the effective date.",
]


def _pdf_escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string literal."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_contract_pdf(out: Path) -> None:
    """Write a minimal multi-page PDF by hand.

    Deliberately dependency-free. The SDK's PDF engine (``pypdfium2``) only
    *reads* PDFs — it cannot author them — and the engine it replaced
    (``pymupdf``) is AGPL-licensed, which must not reappear anywhere in this
    repo. A two-page text-only PDF is about forty lines of raw syntax, so this
    writes the bytes directly rather than taking a dependency to make a fixture.

    Structure: catalog -> pages tree -> one page + content stream each, plus a
    shared Helvetica font. Object offsets are collected as we go and written
    into the xref table at the end, which is what makes the file valid.
    """
    objects: list[bytes] = []

    def add(body: str) -> int:
        """Append an object, returning its 1-based object number."""
        objects.append(body.encode("latin-1"))
        return len(objects)

    n_pages = len(_CONTRACT_PAGES)
    # Object numbers are allocated up front so the tree can reference forward.
    catalog_num, pages_num, font_num = 1, 2, 3
    first_page_num = 4
    page_nums = [first_page_num + 2 * i for i in range(n_pages)]

    add(f"<< /Type /Catalog /Pages {pages_num} 0 R >>")
    kids = " ".join(f"{n} 0 R" for n in page_nums)
    add(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>")
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(_CONTRACT_PAGES):
        content_num = page_nums[i] + 1
        # A4 at 72 dpi = 595x842 pt; text baseline at (72, 770) ~= 1in margins.
        add(
            f"<< /Type /Page /Parent {pages_num} 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        )
        stream = f"BT /F1 14 Tf 72 770 Td ({_pdf_escape(text)}) Tj ET"
        add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")

    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for num, body in enumerate(objects, start=1):
        offsets.append(len(buf))
        buf += f"{num} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(buf)
    buf += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    buf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        buf += f"{off:010d} 00000 n \n".encode("latin-1")
    buf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_num} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")

    out.write_bytes(bytes(buf))


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    make_cat_jpg(HERE / "cat.jpg")
    make_receipt_png(HERE / "receipt.png")
    make_contract_pdf(HERE / "contract.pdf")
    print(f"wrote fixtures to {HERE}")


if __name__ == "__main__":
    main()
