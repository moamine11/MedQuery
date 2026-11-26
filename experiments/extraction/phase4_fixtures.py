"""Create neutral, local PDF fixtures and Phase 2 inventory for extraction checks."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ingestion.adapters import canonical
from ingestion.pipeline import intake


def raster(lines, width=1050, height=380):
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 35)
    for i, line in enumerate(lines):
        draw.text((35, 32 + i * 62), line, font=font, fill="black")
    target = BytesIO()
    image.save(target, format="PNG")
    return target.getvalue()


def create_pdf(path):
    source_lines = ["Independent source paragraph with reliable text.", "Second line of the synthetic paragraph."]
    image_lines = ["SYNTHETIC IMAGE ONLY", "The image contains the number 42.", "A second line verifies local OCR reading."]
    with pymupdf.open() as pdf:
        p = pdf.new_page(width=600, height=720)
        p.insert_text((70, 130), "\n".join(source_lines), fontsize=15)
        pdf.new_page(width=600, height=720)
        p = pdf.new_page(width=600, height=720)
        p.insert_image(p.rect, stream=raster(image_lines, height=1260))
        p = pdf.new_page(width=600, height=720)
        p.insert_image(p.rect, stream=raster(image_lines, height=1260))
        p.insert_text((20, 105), "The image contains the number 74.", fontsize=20, render_mode=3)
        p = pdf.new_page(width=600, height=720)
        p.insert_text((100, 170), "ROTATED CROPBOX TEST", fontsize=22)
        p.set_cropbox(pymupdf.Rect(45, 50, 570, 675))
        p.set_rotation(90)
        p = pdf.new_page(width=600, height=720)
        p.insert_text((50, 85), "\n".join(source_lines), fontsize=15)
        p.insert_image(pymupdf.Rect(75, 280, 520, 540), stream=raster(
            ["IMAGE REGION HAS TEXT", "The fixture label is P-71.", "Regional OCR validation."]))
        p = pdf.new_page(width=600, height=720)
        p.draw_rect(p.rect, fill=(.13, .32, .45))
        p = pdf.new_page(width=600, height=720)
        p.insert_text((60, 100), "Table 1 Example grid", fontsize=15)
        for x in (60, 180, 310, 490):
            p.draw_line((x, 130), (x, 290))
        for y in (130, 170, 210, 250, 290):
            p.draw_line((60, y), (490, y))
        for n, text in enumerate(("Column A", "Cell 42*", "Cell 17", "Cell 15")):
            p.insert_text((70, 155 + n * 40), text, fontsize=14)
        p.insert_text((70, 315), "* Example footnote without a verified relationship.", fontsize=14)
        p.insert_text((60, 411), "Separate paragraph outside the table region.", fontsize=14)
        pdf.save(path)


def main():
    home = ROOT / "data/ingestion/phase4/fixtures"
    root = home / "source"
    root.mkdir(parents=True, exist_ok=True)
    create_pdf(root / "synthetic.pdf")
    row = {"schema_version": 1, "record_id": "SYN-EXTRACT", "title": "Synthetic extraction fixture",
           "issuer": {"name": "Synthetic publisher", "kind": "unknown"}, "language": "en",
           "categories": ["treatment_protocols"], "document_type": "educational_material",
           "authority": "educational", "assets": [{"path": "synthetic.pdf", "role": "primary"}]}
    source = root / "synthetic.jsonl"
    source.write_text(json.dumps(row) + "\n")
    checked = intake(root, "treatment_protocols", canonical, source, shutil.which("pdfinfo"), workers=1)
    if checked["summary"]["error_count"]:
        raise ValueError("Fixture intake failed")
    (home / "inventory.jsonl").write_text("".join(json.dumps(item) + "\n" for item in checked["inventory"]))
    print(home)


if __name__ == "__main__":
    main()
