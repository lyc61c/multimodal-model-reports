"""Copy explicitly selected original website figures and record their provenance."""
import argparse
import hashlib
import os
import re
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from paper_sources import CACHE, PLAN, ROOT, HEADERS, dump, read


def build(entry_id, specs_path):
    row = next(r for r in read(PLAN)["entries"] if r["id"] == entry_id)
    source = read(CACHE / entry_id / "source.json")
    assert source["status"] == "fetched"
    target = ROOT / row["destination"]
    folder = target / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    assets = []
    for spec in read(specs_path):
        with urllib.request.urlopen(urllib.request.Request(spec["url"], headers=HEADERS), timeout=60) as response:
            data = response.read(25 * 1024 * 1024)
            final_url = response.geturl()
        path = folder / spec["filename"]
        path.write_bytes(data)
        preview = None
        if path.suffix.lower() == '.svg':
            svg = ET.fromstring(data)
            assert svg.tag.rsplit('}', 1)[-1] == 'svg', 'Source is not an SVG image'
            viewbox = [float(x) for x in re.split(r'[ ,]+', svg.get('viewBox', '').strip()) if x]
            width = float(re.match(r'[\d.]+', svg.get('width', '0')).group())
            height = float(re.match(r'[\d.]+', svg.get('height', '0')).group())
            if (not width or not height) and len(viewbox) == 4:
                width, height = viewbox[2:]
            node = os.environ.get('PAPER_NODE_RUNTIME', 'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe')
            preview = path.with_name(path.stem + '-preview.png')
            subprocess.run([node, str(ROOT / 'scripts/render_original_svg.cjs'), str(path), str(preview)], check=True)
        else:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                width, height = im.size
        assets.append({"asset_path": "figures/" + spec["filename"],
                       "original_label": spec["label"], "source_url": source["source_url"],
                       "image_url": spec["url"], "image_final_url": final_url,
                       "source_version": source["source_version"], "source_sha256": source["sha256"],
                       "asset_sha256": hashlib.sha256(data).hexdigest(), "width": width, "height": height,
                       "processing": {"method": "original_web_asset_copy", "note": "Original bytes; no redraw or synthetic image"}})
        if preview:
            assets[-1]['preview_path'] = preview.relative_to(target).as_posix()
            assets[-1]['preview_sha256'] = hashlib.sha256(preview.read_bytes()).hexdigest()
            assets[-1]['preview_note'] = 'Faithful raster rendering on a white page for visual inspection; original SVG retained without edits.'
        print(path.relative_to(ROOT).as_posix())
    dump(target / "figures-manifest.json", {"paper_id": entry_id, "source_version": source["source_version"], "assets": assets})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--specs", required=True)
    args = parser.parse_args()
    build(args.id, args.specs)
