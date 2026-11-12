from pathlib import Path

from docx import Document
from pypdf import PdfReader


root = Path(r"D:\Project\RobustNAS\DynamicBN\DynamicBN20251203\robust_metric")
docx_path = root / "docs" / "experiments" / "RobustMetric_研究进展跟进说明_20261016.docx"
pdf_path = root / ".tmp" / "research_progress_render" / "RobustMetric_研究进展跟进说明_20261016.pdf"
render_dir = root / ".tmp" / "research_progress_render"

assert docx_path.exists() and docx_path.stat().st_size > 10_000
document = Document(docx_path)
text = "\n".join(paragraph.text for paragraph in document.paragraphs)
for table in document.tables:
    for row in table.rows:
        text += "\n" + "\t".join(cell.text for cell in row.cells)

required = [
    "47,600",
    "80 个指标",
    "560 条",
    "feature_measure_multiview_max_delta",
    "feature_measure_multiview_rms_delta",
    "严格发布",
    "后续建议",
    "可直接使用的汇报口径",
]
missing = [item for item in required if item not in text]
assert not missing, f"missing required content: {missing}"

headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
assert len(headings) == 11, f"expected 11 headings, found {len(headings)}"
assert len(document.tables) == 8, f"expected 8 tables, found {len(document.tables)}"

assert pdf_path.exists() and pdf_path.stat().st_size > 10_000
pdf = PdfReader(str(pdf_path))
pngs = sorted(render_dir.glob("page-*.png"))
assert len(pdf.pages) == 4, f"expected 4 PDF pages, found {len(pdf.pages)}"
assert len(pngs) == 4 and all(path.stat().st_size > 100_000 for path in pngs)

print(
    f"VERIFY PASS: docx={docx_path.stat().st_size} bytes, "
    f"headings={len(headings)}, tables={len(document.tables)}, "
    f"pdf_pages={len(pdf.pages)}, rendered_pngs={len(pngs)}"
)
