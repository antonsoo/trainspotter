"""Check real SVG 1.1 conversion independently of the browser renderers."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("browser_output", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--baseline", type=Path, default=Path(__file__).parent / "fixtures/unresolved-palette.svg")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
ns = {"svg": "http://www.w3.org/2000/svg"}


def render(file: Path, name: str, raw: bytes | None = None) -> dict[str, object]:
    raw = file.read_bytes() if raw is None else raw
    root = ET.fromstring(raw)
    inner = root.find("svg:svg", ns)
    assert inner is not None
    header = int(inner.attrib["y"])
    image_path = args.output / (name + ".png")
    cairosvg.svg2png(bytestring=raw, write_to=str(image_path))
    with Image.open(image_path) as decoded:
        image = decoded.convert("RGBA")
        background = image.getpixel((0, 0))
        caption_bright = sum(p[3] == 255 and min(p[:3]) > 150
                             for p in image.crop((0, 0, image.width, header)).get_flattened_data())
        axis_bright = sum(p[3] == 255 and min(p[:3]) > 50
                          for p in image.crop((0, header, 52, image.height - 30)).get_flattened_data())
        size = image.size
        painted = sum(p[3] > 0 for p in image.get_flattened_data())
    return {"file": file.name, "svgBytes": len(raw), "svgSha256": hashlib.sha256(raw).hexdigest(),
            "pngBytes": image_path.stat().st_size,
            "pngSha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "dimensions": size, "background": background, "nonTransparentPixels": painted,
            "brightCaptionPixels": caption_bright, "brightAxisPixels": axis_bright,
            "palettePreserved": background == (16, 21, 26, 255) and caption_bright > 0 and axis_bright > 0}


before = render(args.baseline, "before-unresolved")
assert before["palettePreserved"] is False
sizing_control = render(args.baseline, "before-without-responsive-rule",
                        args.baseline.read_bytes().replace(
                            b'svg:root { max-width: 100%; height: auto; }', b''))
assert sizing_control["nonTransparentPixels"] > 0
assert sizing_control["palettePreserved"] is False
rows = []
for file in sorted(args.browser_output.glob("chromium-1440-*.svg")):
    result = render(file, file.stem)
    assert result["palettePreserved"] is True, result
    rows.append(result)
assert len(rows) == 11
source = args.browser_output / "chromium-1440-recorded-divergence-0.svg"
pdf = args.output / 'recorded-divergence.pdf'
cairosvg.svg2pdf(bytestring=source.read_bytes(), write_to=str(pdf))
receipt = {"renderer": "CairoSVG", "version": cairosvg.__version__,
           "scope": "Eleven unique controlled/retained SVG payloads, converted to PNG independently of Chromium/Firefox. Exact dark background and opaque caption/axis paint are checked; font rasterization is not expected to match browsers pixel for pixel. PDF conversion is retained separately, without a physical-printer claim.",
           "baselineSource": "1993888a27d93c6f09558f61fc09b3c84c1a590b", "before": before,
           "controlledSizingIsolation": {"operation": "Remove only the responsive root rule from the original fixture; retains unresolved palette variables.", "result": sizing_control},
           "after": rows, "convertedPdf": {"file": pdf.name, "bytes": pdf.stat().st_size,
             "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}}
if all(shutil.which(tool) for tool in ("pdfinfo", "pdfimages", "pdftotext")):
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    images = subprocess.check_output(["pdfimages", "-list", str(pdf)], text=True)
    text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
    assert re.search(r"Pages:\s+1\b", info)
    assert len(images.strip().splitlines()) == 2
    assert "Source: examples/divergence.trainer_state.json" in text
    assert "Horizontal axis: steps" in text
    assert "54.88" in text and "55.49" in text
    receipt["pdfInspection"] = {"state": "verified", "pages": 1, "rasterImages": 0,
                                "sourceAndAxisText": True, "bothOffScaleValues": ["54.88", "55.49"]}
else:
    receipt["pdfInspection"] = {"state": "unavailable", "reason": "Poppler inspection tools not installed"}
(args.output / 'cairo.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({"renderer": cairosvg.__version__, "beforePalettePreserved": before['palettePreserved'], "after": len(rows), "allPalettesPreserved": all(r['palettePreserved'] for r in rows)}))
