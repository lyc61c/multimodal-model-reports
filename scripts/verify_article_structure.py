"""Check the reading structure, without treating layout checks as factual review."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from build_reports import ROOT, read, write


def section(body: str, number: int) -> str:
    match = re.search(rf"^## {number}\. [^\n]+\n([\s\S]*?)(?=^## \d+\. |\Z)", body, re.M)
    return match[1] if match else ""


def plain_cell(cell: str) -> str:
    return cell.strip().replace("**", "").strip()


def structure_issues(body: str) -> tuple[list[str], dict]:
    architecture, principles = section(body, 3), section(body, 4)
    issues = []
    tables = re.findall(r"^\|[^\n]+\n\|[ :|\-]+\n(?:\|[^\n]+\n?)+", architecture, re.M)
    module_tables = [table for table in tables if [plain_cell(c) for c in table.splitlines()[0].strip("|").split("|")]
                     == ["模块", "输入与输出", "公开机制"]]
    if len(module_tables) != 1:
        issues.append("Section 3 must contain one module / input-output / public-mechanism table")
    module_rows = 0
    if module_tables:
        rows = module_tables[0].splitlines()[2:]
        module_rows = len(rows)
        if module_rows < 2:
            issues.append("Architecture table has fewer than two explained components")
        for row in rows:
            cells = re.split(r"(?<!\\)\|", row.strip().strip("|"))
            if len(cells) != 3 or any(not plain_cell(c) for c in cells):
                issues.append("Architecture table has an empty cell or inconsistent columns")
                break
    headings = list(re.finditer(r"^### (4\.(\d+))\s+([^\n]+)$", principles, re.M))
    if len(headings) < 2:
        issues.append("Section 4 needs numbered subsections for its distinct mechanisms")
    numbers = [int(h[2]) for h in headings]
    if numbers != list(range(1, len(numbers) + 1)):
        issues.append("Section 4 subsection numbers are not consecutive")
    other_headings = re.findall(r"^### (?!4\.\d+\s)[^\n]+", principles, re.M)
    if other_headings:
        issues.append("Section 4 contains unnumbered level-three headings")
    lengths = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(principles)
        content = principles[heading.end():end]
        lengths.append(len(re.findall(r"[\u4e00-\u9fff]", content)))
        if not content.strip():
            issues.append(f"Empty subsection {heading[1]}")
    return issues, {
        "module_rows": module_rows,
        "principle_subsections": [h[0].strip() for h in headings],
        "subsection_chinese_characters": lengths,
        "architecture_characters": len(architecture),
        "principles_characters": len(principles),
    }


def main() -> int:
    rows = []
    for entry in read("data/article-plan.json")["entries"]:
        folder = ROOT / entry["destination"]
        if read(str(folder.relative_to(ROOT) / "metadata.json")).get("article_status") != "complete":
            continue
        path = folder / "README.md"
        body = path.read_text(encoding="utf-8-sig")
        issues, metrics = structure_issues(body)
        rows.append({"id": entry["id"], "path": path.relative_to(ROOT).as_posix(),
                     "article_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     **metrics, "issues": issues})
    report = {"articles": len(rows), "articles_with_issues": sum(bool(r["issues"]) for r in rows),
              "module_rows": sum(r["module_rows"] for r in rows),
              "principle_subsections": sum(len(r["principle_subsections"]) for r in rows),
              "limits": "Checks structure and table syntax only. It does not establish technical accuracy, novelty or source support.",
              "entries": rows}
    write("data/article-structure-validation.json", json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, ensure_ascii=False))
    for row in rows:
        if row["issues"]:
            print(row["path"], "; ".join(row["issues"]))
    return 1 if report["articles_with_issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
