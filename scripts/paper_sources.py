"""Prepare the retained article queue and cache authoritative source material.

This script does not generate technical prose. Authors read the cached sources,
write each introduction, and choose original figure/table regions to illustrate it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/article-plan.json"
CACHE = ROOT / "source/papers"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PaperReadingRepository/1.0)"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def arxiv_id(url):
    match = re.search(r"arxiv\.org/(?:pdf|abs|html)/(\d{4}\.\d+(?:v\d+)?)", url or "")
    return match.group(1).removesuffix(".pdf") if match else None


def prepare():
    entries = read(ROOT / "data/catalogue.json")["entries"]
    rows = []
    arxiv_count = 0
    for entry in sorted(entries, key=lambda e: (e["family_slug"], e["date"], e["title"])):
        if entry["catalogue_group"] == "excluded_low_star":
            continue
        aid = arxiv_id(entry.get("paper_url"))
        if aid:
            version = aid if re.search(r"v\d+$", aid) else aid + "v1"
            paper_slug = "arxiv-" + aid.split("v")[0]
            owner = "writer_" + str(arxiv_count % 3 + 1)
            arxiv_count += 1
            source_url = "https://arxiv.org/pdf/" + version
        else:
            version = "official-source-snapshot-2026-10-08"
            paper_slug = "document-" + entry["id"]
            owner = "root"
            source_url = entry.get("paper_url") or entry.get("github_code_url")
        if aid and aid.split("v")[0] == "2502.13923":
            paper_slug, owner = "qwen2.5-vl", "existing"
        destination = f"papers/{entry['family_slug']}/{paper_slug}"
        row = {"id": entry["id"], "title": entry["title"], "family": entry["family"],
               "family_slug": entry["family_slug"], "organization": entry["organization"],
               "date": entry["date"], "paper_url": entry.get("paper_url"),
               "github_code_url": entry.get("github_code_url"), "repo": entry.get("repo"),
               "catalogue_group": entry["catalogue_group"], "document_kind": entry["scope"]["kind"],
               "source_url": source_url, "source_version": version, "arxiv_id": aid,
               "destination": destination, "owner": owner}
        rows.append(row)
        target = ROOT / destination
        target.mkdir(parents=True, exist_ok=True)
        metadata_path = target / "metadata.json"
        if not metadata_path.exists():
            dump(metadata_path, {"paper_id": entry["id"], "catalogue_entry_id": entry["id"],
                "family_slug": entry["family_slug"], "title": entry["title"],
                "article_file": "README.md", "article_status": "planned", "source_version": version,
                "source_url": source_url, "document_kind": row["document_kind"],
                "catalogue_group": row["catalogue_group"], "article_template_version": "1.1"})
    dump(PLAN, {"snapshot_date": "2026-10-08", "scope": "All 272 entries surviving the original star filter", "entries": rows})
    print(json.dumps({"total": len(rows), "owners": dict(Counter(r["owner"] for r in rows)),
                      "groups": dict(Counter(r["catalogue_group"] for r in rows))}, ensure_ascii=True))


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.text = []
        self.images = []
        self.links = []
        self.in_title = False
        self.title = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style", "noscript", "svg"):
            self.skip += 1
        if tag == "title":
            self.in_title = True
        if tag == "img":
            src = attrs.get("src") or attrs.get("data-src")
            if src:
                self.images.append({"src": src, "alt": attrs.get("alt", "")})
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        if tag in ("p", "div", "section", "h1", "h2", "h3", "li", "tr", "br"):
            self.text.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self.skip:
            self.skip -= 1
        if tag == "title":
            self.in_title = False

    def handle_data(self, value):
        if self.in_title:
            self.title.append(value)
        if not self.skip:
            self.text.append(value)


def public_url(url):
    if not url:
        return None
    if "github.com/" in url and "/blob/" in url:
        url = url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    if "huggingface.co/" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    return url


def fetch(row, retry=False):
    folder = CACHE / row["id"]
    status_path = folder / "source.json"
    if status_path.exists() and not retry:
        previous = read(status_path)
        if previous.get("status") == "fetched":
            return {"id": row["id"], "status": "cached", "kind": previous["kind"]}
    folder.mkdir(parents=True, exist_ok=True)
    url = public_url(row.get("source_url"))
    result = {"id": row["id"], "title": row["title"], "source_url": row.get("source_url"),
              "request_url": url, "source_version": row["source_version"], "observed_at": "2026-10-08"}
    try:
        if not url:
            raise ValueError("Upstream provides no source URL")
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
            data = response.read(90 * 1024 * 1024)
            result["final_url"] = response.geturl()
            result["content_type"] = response.headers.get("Content-Type", "")
        result.update({"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        if data.startswith(b"%PDF"):
            from pypdf import PdfReader
            pdf_path = folder / "source.pdf"
            pdf_path.write_bytes(data)
            reader = PdfReader(pdf_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            (folder / "source.txt").write_text("\n\n".join(f"===== PDF PAGE {i + 1} =====\n{text}" for i, text in enumerate(pages)), encoding="utf-8")
            dump(folder / "pages.json", [{"pdf_page": i + 1, "text": text} for i, text in enumerate(pages)])
            result.update({"kind": "pdf", "pages": len(pages), "text_characters": sum(map(len, pages))})
        else:
            html = data.decode("utf-8", errors="replace")
            (folder / "source.html").write_text(html, encoding="utf-8")
            parser = PageParser()
            parser.feed(html)
            plain = re.sub(r"[ \t]+", " ", "".join(parser.text))
            plain = re.sub(r"\n\s*\n+", "\n\n", plain).strip()
            (folder / "source.txt").write_text(plain, encoding="utf-8")
            result.update({"kind": "html", "html_title": "".join(parser.title).strip(),
                           "text_characters": len(plain), "images": [{"url": urllib.parse.urljoin(result["final_url"], image["src"]), "alt": image["alt"]} for image in parser.images],
                           "links": [urllib.parse.urljoin(result["final_url"], link) for link in parser.links]})
        result["status"] = "fetched"
    except Exception as error:
        result.update({"status": "fetch_error", "error": str(error)})
    dump(status_path, result)
    return {key: result.get(key) for key in ("id", "status", "kind", "pages", "text_characters", "error")}


def fetch_many(args):
    rows = read(PLAN)["entries"]
    if args.owner:
        rows = [r for r in rows if r["owner"] == args.owner]
    if args.ids:
        rows = [r for r in rows if r["id"] in set(args.ids.split(","))]
    if args.limit:
        rows = rows[:args.limit]
    if args.url:
        assert len(rows) == 1, "An alternate source URL requires exactly one entry"
        row = dict(rows[0])
        previous_path = CACHE / row["id"] / "source.json"
        history_path = CACHE / row["id"] / "source-attempts.json"
        history = read(history_path) if history_path.exists() else []
        if previous_path.exists():
            history.append(read(previous_path))
        dump(history_path, history)
        row["source_url"] = args.url
        row["source_version"] = args.version or row["source_version"]
        rows = [row]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda row: fetch(row, args.retry), rows):
            print(json.dumps(result, ensure_ascii=True), flush=True)


def assets(args):
    import pdfplumber
    row = next(r for r in read(PLAN)["entries"] if r["id"] == args.id)
    source = read(CACHE / row["id"] / "source.json")
    assert source["kind"] == "pdf" and source["status"] == "fetched"
    target = ROOT / row["destination"]
    figures = target / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # Publication assets must select a figure/table region explicitly.
    # Whole pages belong in the ignored reading cache, not an introduction.
    if not args.specs:
        raise ValueError("Figure/table assets require --specs with explicit crop bbox coordinates; full-page publication screenshots are disabled")
    specs = read(args.specs)
    if not specs or any(not spec.get("bbox") for spec in specs):
        raise ValueError("Every publication asset must provide an explicit figure/table crop bbox")
    records = []
    with pdfplumber.open(CACHE / row["id"] / "source.pdf") as pdf:
        for index, spec in enumerate(specs):
            page = pdf.pages[spec["page"] - 1]
            x0, top, x1, bottom = spec["bbox"]
            if (x1-x0)/page.width >= 0.98 and (bottom-top)/page.height >= 0.98:
                raise ValueError("Select only the required figure/table, not the whole PDF page")
            rendered = page.crop(spec["bbox"])
            filename = spec.get("filename") or f"original-p{spec['page']:03d}-region-{index + 1:02d}.png"
            file = figures / filename
            rendered.to_image(resolution=args.dpi, antialias=True).save(file, quantize=False)
            record = {"asset_path": "figures/" + filename, "original_label": spec["label"],
                      "pdf_page": spec["page"], "source_url": source["source_url"] + ("#page=" + str(spec["page"])),
                      "source_version": source["source_version"], "source_sha256": source["sha256"],
                      "asset_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                      "processing": {"method": "pdf_region_render", "dpi": args.dpi,
                                     "crop_bbox": spec["bbox"], "page_size": [page.width, page.height], "note": "Original figure/table region rendered from the fixed PDF; surrounding prose excluded"}}
            records.append(record)
            print(json.dumps({"asset": str(file.relative_to(ROOT)), "page": spec["page"], "label": spec["label"]}, ensure_ascii=True))
    dump(target / "figures-manifest.json", {"paper_id": row["id"], "source_version": source["source_version"], "source_pdf_sha256": source["sha256"], "assets": records})


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("--owner")
    fetch_parser.add_argument("--ids")
    fetch_parser.add_argument("--workers", type=int, default=3)
    fetch_parser.add_argument("--limit", type=int)
    fetch_parser.add_argument("--retry", action="store_true")
    fetch_parser.add_argument("--url", help="Explicitly selected authoritative alternate source")
    fetch_parser.add_argument("--version", help="Version of the selected alternate source")
    asset_parser = sub.add_parser("assets")
    asset_parser.add_argument("--id", required=True)
    asset_parser.add_argument("--specs", required=True, help="JSON array of explicit page, label, bbox and optional filename for each figure/table")
    asset_parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    {"prepare": lambda: prepare(), "fetch": lambda: fetch_many(args), "assets": lambda: assets(args)}[args.command]()


if __name__ == "__main__":
    main()
