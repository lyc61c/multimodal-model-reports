"""Render explicitly selected PDF figure/table regions and register their provenance.

Plans and draft images live in ignored tmp/. Applying a plan requires visual-review
notes. This tool never silently keeps a full page when a region is unavailable.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

from paper_sources import ROOT, dump, read


def safe_path(relative):
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(plan_path, only=None, allow_pending=False):
    import pdfplumber
    plan = read(plan_path)
    pdfs = {}
    try:
        for item in plan["items"]:
            if only and item["id"] not in only:
                continue
            if item.get("action") == "drop":
                continue
            if not item.get("regions") and allow_pending:
                print(json.dumps({"id": item["id"], "page": item["pdf_page"], "status": "NEEDS_EXPLICIT_REGION"}), flush=True)
                continue
            assert item.get("regions"), item["old_asset_path"]
            pdf_path = safe_path(item["pdf_path"])
            if pdf_path not in pdfs:
                assert digest(pdf_path) == item["source_sha256"], pdf_path
                pdfs[pdf_path] = pdfplumber.open(pdf_path)
            page = pdfs[pdf_path].pages[item["pdf_page"] - 1]
            item["page_size"] = [page.width, page.height]
            for region in item["regions"]:
                box = region["bbox"]
                assert len(box) == 4 and 0 <= box[0] < box[2] <= page.width + 0.01
                assert 0 <= box[1] < box[3] <= page.height + 0.01
                assert not ((box[2]-box[0])/page.width >= 0.98 and (box[3]-box[1])/page.height >= 0.98), "Full-page or near-full-page screenshots are not publication assets"
                draft = safe_path(region["draft_path"])
                draft.parent.mkdir(parents=True, exist_ok=True)
                page.crop(box).to_image(resolution=region.get("dpi", 180), antialias=True).save(draft, quantize=False)
                region["draft_sha256"] = digest(draft)
            print(json.dumps({"id": item["id"], "page": item["pdf_page"], "regions": len(item["regions"])}, ensure_ascii=True), flush=True)
    finally:
        for pdf in pdfs.values():
            pdf.close()
    dump(plan_path, plan)


def tidy_full_page_wording(body):
    substitutions = [
        (r"配图为原文整页截图，保留图表与周围正文", "配图为原文图表区域裁切"),
        (r"配图为原文整页截图", "配图为原文图表区域裁切"),
        (r"图表为原文整页截图", "图表为原文图表区域裁切"),
        (r"本资产为完整页面截图[。.]?", "仅保留所引用的原文图表区域。"),
        (r"整页保留了", "裁图保留了"),
        (r"完整页面截图|完整页截图|整页截图|整页渲染|完整页面|完整页渲染", "图表区域裁切"),
    ]
    for pattern, replacement in substitutions:
        body = re.sub(pattern, replacement, body)
    return body


def apply(plan_path):
    plan = read(plan_path)
    groups = {}
    for item in plan["items"]:
        assert item.get("visual_review"), "Each selected asset needs actual visual review notes"
        assert item.get("selected_labels_verified"), "Verify all needed target figures/tables before applying"
        assert not item.get("unresolved_targets"), "Missing target figures/tables need resolution before applying"
        assert item.get("action") in {"crop", "drop"}, "Unknown crop action"
        assert bool(item.get("regions")) == (item["action"] == "crop"), "Crop actions need regions; dropped images must not have regions"
        groups.setdefault(item["destination"], []).append(item)
    # Preflight every article before any files are replaced.
    for destination, items in groups.items():
        folder = safe_path(destination)
        review = read(folder / "review.json")
        assert isinstance(read(folder / "metadata.json"), dict), destination
        assert isinstance(read(folder / "evidence.json"), dict), destination
        assert isinstance(review.get("checks", []), (list, dict)), destination
        assert review.get("artifact_checks") is None or isinstance(review["artifact_checks"], dict), destination
        records = {a["asset_path"]: a for a in read(folder / "figures-manifest.json")["assets"]}
        body = (folder / "README.md").read_text(encoding="utf-8-sig")
        new_paths = []
        for item in items:
            old = item["old_asset_path"]
            assert old in records and digest(folder / old) == records[old]["asset_sha256"], (destination, old)
            assert re.search(r"!\[([^\]]*)\]\(" + re.escape(old) + r"\)", body), (destination, old)
            for edit in item.get("body_edits", []):
                assert edit["old"] in body, (destination, "Expected caption wording was not found")
            for region in item.get("regions", []):
                target = folder / "figures" / region["filename"]
                target.resolve().relative_to(folder.resolve())
                assert not target.exists(), (destination, "New crop already exists", target.name)
                assert digest(safe_path(region["draft_path"])) == region["draft_sha256"], region["filename"]
                width, height = item["page_size"]
                x0, top, x1, bottom = region["bbox"]
                assert 0 <= x0 < x1 <= width + 0.01 and 0 <= top < bottom <= height + 0.01, destination
                assert not ((x1-x0)/width >= 0.98 and (bottom-top)/height >= 0.98), "Near-full-page crop rejected"
                new_paths.append(target)
        assert len(new_paths) == len(set(new_paths)), (destination, "Duplicate crop filenames")
    for destination, items in groups.items():
        folder = safe_path(destination)
        manifest_path = folder / "figures-manifest.json"
        manifest = read(manifest_path)
        records = {a["asset_path"]: a for a in manifest["assets"]}
        body_path = folder / "README.md"
        body = body_path.read_text(encoding="utf-8-sig")
        backups = safe_path("tmp/full-page-backups/" + destination)
        backups.mkdir(parents=True, exist_ok=True)
        for name in ["README.md", "figures-manifest.json", "metadata.json", "review.json", "evidence.json"]:
            if (folder / name).exists() and not (backups / name).exists():
                shutil.copy2(folder / name, backups / name)
        for item in items:
            old = item["old_asset_path"]
            assert old in records, (destination, old)
            old_record = records[old]
            old_file = folder / old
            assert digest(old_file) == old_record["asset_sha256"], old_file
            backup_image = backups / old
            backup_image.parent.mkdir(parents=True, exist_ok=True)
            if not backup_image.exists():
                shutil.copy2(old_file, backup_image)
            image_pattern = r"!\[([^\]]*)\]\(" + re.escape(old) + r"\)"
            assert re.search(image_pattern, body), (destination, old)
            replacements = []
            additions = []
            for region in item.get("regions", []):
                draft = safe_path(region["draft_path"])
                assert digest(draft) == region["draft_sha256"]
                new_path = "figures/" + region["filename"]
                target = folder / new_path
                target.resolve().relative_to(folder.resolve())
                shutil.copy2(draft, target)
                record = copy.deepcopy(old_record)
                record.update(asset_path=new_path, original_label=region["label"], asset_sha256=digest(target))
                record["processing"] = {
                    "method": "pdf_region_render", "dpi": region.get("dpi", 180),
                    "crop_bbox": region["bbox"], "page_size": item["page_size"],
                    "note": "Original figure/table pixels rendered from the fixed PDF; surrounding prose excluded.",
                    "visual_review": item["visual_review"],
                }
                additions.append(record)
                replacements.append(f'![原文 {region["label"]}]({new_path})')
            if item.get("action") == "drop":
                assert not additions
                replacement = item.get("replacement_text", "")
            else:
                assert additions
                replacement = "\n\n".join(replacements)
            body = re.sub(image_pattern, lambda m: replacement, body)
            for edit in item.get("body_edits", []):
                assert edit["old"] in body, (destination, "Expected caption wording was not found")
                body = body.replace(edit["old"], edit["new"])
            manifest["assets"] = [a for a in manifest["assets"] if a["asset_path"] != old] + additions
            old_file.unlink()
        body = tidy_full_page_wording(body)
        body_path.write_text(body, encoding="utf-8")
        dump(manifest_path, manifest)
        metadata = read(folder / "metadata.json")
        metadata.update(chinese_characters=len(re.findall(r"[\u4e00-\u9fff]", body)), figure_revision_date="2026-10-09")
        dump(folder / "metadata.json", metadata)
        review_path = folder / "review.json"
        review = read(review_path)
        review["article_sha256"] = digest(body_path)
        revision_check = "2026-10-09: Replaced full-page screenshots with visually checked, explicitly selected original figure/table crops; non-figure pages removed. Technical claims unchanged."
        checks = review.setdefault("checks", [])
        if isinstance(checks, dict):
            checks["figure_region_revision"] = revision_check
        elif isinstance(checks, list):
            checks.append(revision_check)
        else:
            raise TypeError("review.checks must be a list or dictionary")
        if review.get("artifact_checks") is None:
            review["artifact_checks"] = {}
        review.setdefault("artifact_checks", {})["original_assets"] = len(manifest["assets"])
        review["artifact_checks"]["no_full_page_screenshots"] = all(a.get("processing", {}).get("method") != "full_pdf_page_render" for a in manifest["assets"])
        review["figure_revision"] = {"date": "2026-10-09", "regions_and_sources": "figures-manifest.json", "visual_checks": [x["visual_review"] for x in items]}
        dump(review_path, review)
        evidence_path = folder / "evidence.json"
        evidence = read(evidence_path)
        evidence["figure_revision"] = {"date": "2026-10-09", "method": "Original PDF region extraction", "unchanged_source_versions": True}
        dump(evidence_path, evidence)
        print(json.dumps({"destination": destination, "replaced_pages": len(items), "current_assets": len(manifest["assets"])}, ensure_ascii=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["render", "apply"])
    parser.add_argument("plan")
    parser.add_argument("--ids", help="Comma-separated article IDs to rerender after adjusting their crop coordinates")
    parser.add_argument("--allow-pending", action="store_true", help="Render available draft regions while leaving unresolved items visibly pending")
    args = parser.parse_args()
    plan = Path(args.plan).resolve()
    plan.relative_to(ROOT)
    if args.mode == "render":
        render(plan, set(args.ids.split(",")) if args.ids else None, args.allow_pending)
    else:
        apply(plan)


if __name__ == "__main__":
    main()
