"""Validate protected GitHub math syntax and export exact TeX for engine/GFM checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter

from build_reports import ROOT, completed_articles, write
from format_articles import MATH_PATTERN, fenced_math, markdown_segments, math_expressions

DOCS = "https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions"
FORBIDDEN = re.compile(r"\\(?:operatorname|newcommand|renewcommand|providecommand|def|gdef|require|href|htmlClass|htmlId|htmlStyle)\b")
ORPHAN = re.compile(r"\\(?:mathcal|mathbb|mathrm|mathbf|alpha|beta|gamma|rho|theta|sqrt|frac|times|sum|prod)\b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", help="Repository-relative JSON path for a local rendering validator")
    args = parser.parse_args()
    issues, expressions, manifest = [], [], []
    articles = completed_articles()
    for article in articles:
        path = article["path"]
        relative = path.relative_to(ROOT).as_posix()
        body = path.read_text(encoding="utf-8-sig")
        manifest.append(relative + ":" + hashlib.sha256(path.read_bytes()).hexdigest())
        for expression in math_expressions(body):
            expressions.append({"article": relative, **expression})
            if expression["delimiter"] not in {"guarded", "math_fence"}:
                issues.append({"article": relative, "problem": "Unprotected or legacy math delimiter", "tex": expression["tex"][:120]})
            if FORBIDDEN.search(expression["tex"]):
                issues.append({"article": relative, "problem": "Restricted/custom math macro", "tex": expression["tex"][:120]})
            if "|" in expression["tex"]:
                issues.append({"article": relative, "problem": "Literal pipe can collide with a GFM table", "tex": expression["tex"][:120]})
            if re.search(r"(?<!\\)[<>]|&(?:lt|gt|amp);", expression["tex"]):
                issues.append({"article": relative, "problem": "Math can be truncated by HTML parsing; use TeX relation commands in inline AND display math", "tex": expression["tex"][:120]})
            if expression["display"] and re.search(r"(?<!\\)\\\\[ \t]*\n", expression["tex"]):
                issues.append({"article": relative, "problem": "Physical newline after a TeX row break can gain a control space on GitHub", "tex": expression["tex"][:120]})
        for segment, is_code in markdown_segments(body):
            if is_code:
                continue
            remainder = MATH_PATTERN.sub("", segment)
            if re.search(r"(?<!\\)\$", remainder):
                issues.append({"article": relative, "problem": "Unmatched math dollar outside code"})
            if re.search(r"(?<!\\)\\[()\[\]]", remainder):
                issues.append({"article": relative, "problem": "Broken legacy math boundary"})
            for match in ORPHAN.finditer(remainder):
                context = remainder[max(0, match.start()-40):match.end()+80]
                issues.append({"article": relative, "problem": "TeX command outside a math environment", "context": context})
    result = {
        "status": "failed" if issues else "passed",
        "articles": len(articles),
        "expressions": len(expressions),
        "delimiters": dict(Counter(value["delimiter"] for value in expressions)),
        "article_manifest_sha256": hashlib.sha256("\n".join(sorted(manifest)).encode()).hexdigest(),
        "issues": issues,
        "reference": DOCS,
        "limits": "Syntax/macro checks precede Markdown and engine validation. This is not a complete GitHub macro whitelist or a mathematical proof check.",
    }
    write("data/github-math-verification.json", json.dumps(result, ensure_ascii=False, indent=2))
    if args.export:
        target = (ROOT / args.export).resolve()
        if not target.is_relative_to(ROOT):
            parser.error("--export must stay inside the repository")
        write(target.relative_to(ROOT).as_posix(), json.dumps({"validation": result, "expressions": expressions}, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
