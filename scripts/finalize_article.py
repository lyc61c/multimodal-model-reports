"""Check and register an authored article; never generates technical prose."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from paper_sources import CACHE, PLAN, ROOT, dump, read


def finalize(entry_id, title, checks):
    row = next(r for r in read(PLAN)["entries"] if r["id"] == entry_id)
    folder = ROOT / row["destination"]
    article = folder / "README.md"
    body = article.read_text(encoding="utf-8")
    expected = ["论文信息与模型定位", "快速概览", "整体架构", "核心创新与数学原理", "数据构建与处理", "训练流程与目标", "实验设置与主要结果"]
    headings = re.findall(r"^## \d[.、] ?([^\n]+)$", body, re.M)
    assert headings == expected, headings
    assert not re.search(r"〈|〉|TODO|待填写|TBD", body)
    assert "\ufffd" not in body
    assert body.count("$$") % 2 == 0
    source = read(CACHE / entry_id / "source.json")
    assert source["status"] == "fetched"
    manifest_path = folder / "figures-manifest.json"
    manifest = read(manifest_path) if manifest_path.exists() else {"paper_id": entry_id, "source_version": source["source_version"], "assets": []}
    for asset in manifest["assets"]:
        file = folder / asset["asset_path"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == asset["asset_sha256"]
        assert asset["asset_path"] in body
    for link in re.findall(r"\]\(([^)]+)\)", body):
        if "://" not in link and not link.startswith("#"):
            assert (folder / link.split("#")[0]).resolve().exists(), link
    metadata = read(folder / "metadata.json")
    metadata.update({"article_status": "complete", "article_file": "README.md", "title_zh": title,
                     "source_version": source["source_version"], "source_url": source.get("request_url") or source["source_url"],
                     "source_sha256": source["sha256"], "article_language": "zh-CN", "authored_at": "2026-10-08",
                     "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", body)), "figures_manifest": "figures-manifest.json"})
    dump(folder / "metadata.json", metadata)
    dump(manifest_path, manifest)
    dump(folder / "evidence.json", {"paper_id": entry_id, "source_url": metadata["source_url"], "source_version": metadata["source_version"],
                                   "source_sha256": source["sha256"], "verification_notes": checks,
                                   "citation_locations": "Sources and exact PDF pages/figure/table numbers appear next to the claims in README.md"})
    dump(folder / "review.json", {"status": "passed", "review_kind": "author_source_check", "checks": checks,
                                 "artifact_checks": {"seven_sections": True, "no_placeholders": True, "local_links_resolve": True,
                                                     "asset_hashes_match": True, "original_assets": len(manifest["assets"])},
                                 "article_sha256": hashlib.sha256(article.read_bytes()).hexdigest(),
                                 "limitations": ["Author-reported evidence; no model reproduction", "Descriptions fixed to the cited source snapshot"]})
    print(json.dumps({"id": entry_id, "status": "complete", "chinese_characters": metadata["chinese_characters"], "assets": len(manifest["assets"])}, ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--checks", required=True, help="Path to JSON list of actual source checks")
    args = parser.parse_args()
    finalize(args.id, args.title, read(args.checks))
