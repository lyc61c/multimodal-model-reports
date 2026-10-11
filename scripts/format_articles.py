"""Normalize reading navigation and GitHub math delimiters, without writing technical prose."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import string
import urllib.parse
import unicodedata
from datetime import date
from pathlib import Path

from build_reports import ROOT, completed_articles, read, write

SECTIONS = [
    ("paper-info", "论文定位"), ("overview", "快速概览"),
    ("architecture", "整体架构"), ("principles", "核心原理"),
    ("data", "数据构建"), ("training", "训练流程"),
    ("experiments", "实验结果"),
]


def markdown_segments(body):
    """Yield prose/code segments; never rewrite fenced, indented or inline code."""
    ranges = []
    fence = None
    start = 0
    offset = 0
    indented = None
    legacy_display = False
    for line in body.splitlines(keepends=True):
        if fence:
            marker, length = fence
            if re.fullmatch(rf" {{0,3}}{re.escape(marker)}{{{length},}}[ \t]*\n?", line):
                ranges.append((start, offset + len(line)))
                fence = None
        else:
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})([^\n]*)", line)
            if opening and (opening[1][0] != "`" or "`" not in opening[2]):
                if indented is not None:
                    ranges.append((indented, offset))
                    indented = None
                fence = (opening[1][0], len(opening[1]))
                start = offset
            elif re.match(r"^ {0,3}\\\[", line) or legacy_display:
                if indented is not None:
                    ranges.append((indented, offset))
                    indented = None
                legacy_display = r"\]" not in line
            elif re.match(r"^(?: {4}|\t)", line):
                if indented is None:
                    indented = offset
            elif indented is not None and line.strip():
                ranges.append((indented, offset))
                indented = None
        offset += len(line)
    if fence:
        ranges.append((start, len(body)))
    elif indented is not None:
        ranges.append((indented, len(body)))

    def inline_segments(prose):
        runs = list(re.finditer(r"`+", prose))
        cursor = 0
        index = 0
        while index < len(runs):
            opening = runs[index]
            slash_count = len(prose[:opening.start()]) - len(prose[:opening.start()].rstrip("\\"))
            if slash_count % 2:
                index += 1
                continue
            closing = next((j for j in range(index + 1, len(runs)) if len(runs[j][0]) == len(opening[0])), None)
            if closing is None:
                index += 1
                continue
            yield prose[cursor:opening.start()], False
            yield prose[opening.start():runs[closing].end()], True
            cursor = runs[closing].end()
            index = closing + 1
        yield prose[cursor:], False

    cursor = 0
    for start, end in ranges:
        yield from inline_segments(body[cursor:start])
        yield body[start:end], True
        cursor = end
    yield from inline_segments(body[cursor:])


def normalize_math(body):
    """Change delimiters outside code, preserving formula contents."""
    conversions = 0

    def prose(text):
        nonlocal conversions

        def inline(match):
            nonlocal conversions
            conversions += 1
            return "$" + match[1] + "$"

        def display(match):
            nonlocal conversions
            conversions += 1
            return "\n\n$$\n" + match[1].strip() + "\n$$\n\n"

        text = re.sub(r"(?<!\\)\\\((.*?)\\\)", inline, text, flags=re.S)
        text = re.sub(r"(?<!\\)\\\[(.*?)\\\]", display, text, flags=re.S)
        return re.sub(r"\n{4,}", "\n\n\n", text)

    return "".join(text if is_code else prose(text) for text, is_code in markdown_segments(body)), conversions


def normalize_emphasis(body):
    """Keep strong emphasis readable when Chinese punctuation touches prose."""
    def punctuation(character):
        return character in string.punctuation or unicodedata.category(character).startswith("P")

    def prose(text):
        def spacing(match):
            value = match[0]
            previous = text[match.start() - 1] if match.start() else " "
            following = text[match.end()] if match.end() < len(text) else " "
            if punctuation(match[1][0]) and not (previous.isspace() or punctuation(previous)):
                value = " " + value
            if punctuation(match[1][-1]) and not (following.isspace() or punctuation(following)):
                value += " "
            return value
        return re.sub(r"\*\*([^*\n]+)\*\*", spacing, text)

    return "".join(text if is_code else prose(text) for text, is_code in markdown_segments(body))


def relative_link(target, folder):
    return urllib.parse.quote(Path(os.path.relpath(target, folder)).as_posix(), safe="/.-_")


def short_title(article):
    return re.split(r"[：:]", article["title"], maxsplit=1)[0].replace("[", "\\[").replace("]", "\\]")


def format_article(article, previous, following, family):
    path = article["path"]
    body = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    for name in ("article-header", "article-toc", "article-footer"):
        body = re.sub(rf"<!-- {name}:start -->[\s\S]*?<!-- {name}:end -->\n*", "", body)
    # Replace the previous simple breadcrumb, when present.
    body = re.sub(r"\A\[返回[^\n]*\n+", "", body)
    for anchor, _ in SECTIONS:
        body = re.sub(rf'^<a id="{anchor}"></a>\n+', "", body, flags=re.M)
    body, conversions = normalize_math(body)
    body = normalize_emphasis(body)
    heading = re.search(r"^## 1[.、] ?", body, flags=re.M)
    if heading is None:
        raise ValueError(f"Missing first section: {path}")
    toc = ("<!-- article-toc:start -->\n<a id=\"article-toc\"></a>\n\n**本篇目录** · "
           + " · ".join(f"[{label}](#{anchor})" for anchor, label in SECTIONS)
           + "\n<!-- article-toc:end -->\n\n")
    body = body[:heading.start()] + toc + body[heading.start():]
    body = re.sub(r"^## ([1-7])[.、] ?", lambda m: f'<a id="{SECTIONS[int(m[1]) - 1][0]}"></a>\n\n' + m[0], body, flags=re.M)
    header = ("<!-- article-header:start -->\n"
              "[项目首页](../../../README.md) › [模型系列](../../../reports/model-family-index.md) › "
              f"[{family}](../README.md)\n<!-- article-header:end -->\n\n")
    links = []
    if previous:
        links.append(f'[← 上一篇：{short_title(previous)}]({relative_link(previous["path"], path.parent)})')
    links.append(f'[按发表时间浏览](../../../reports/paper-reading-guide.md#year-{article["date"][:4]})')
    if following:
        links.append(f'[下一篇：{short_title(following)} →]({relative_link(following["path"], path.parent)})')
    footer = ("\n\n<!-- article-footer:start -->\n---\n\n"
              + " · ".join(links) + "\n\n[返回本篇目录](#article-toc) · [返回模型系列](../README.md)"
              + "\n<!-- article-footer:end -->\n")
    return header + body.strip() + footer, conversions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Apply deterministic formatting; otherwise report drift")
    parser.add_argument("--revision-date", help="Explicit review date, YYYY-MM-DD, required when writing")
    args = parser.parse_args()
    if args.write:
        if not args.revision_date:
            parser.error("--write requires --revision-date YYYY-MM-DD")
        date.fromisoformat(args.revision_date)
    plan = read("data/article-plan.json")["entries"]
    families = {entry["family_slug"]: entry["family"] for entry in plan}
    title_changes = []
    for entry in plan:
        folder = ROOT / entry["destination"]
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("article_status") != "complete":
            continue
        article_path = folder / metadata.get("article_file", "README.md")
        titles = re.findall(r"^# (.+)$", article_path.read_text(encoding="utf-8-sig"), re.M)
        if len(titles) != 1:
            raise ValueError(f"Expected exactly one document title: {article_path}")
        if metadata.get("title_zh") != titles[0]:
            title_changes.append(entry["destination"])
            if args.write:
                metadata["title_zh"] = titles[0]
                write(str(metadata_path.relative_to(ROOT)), json.dumps(metadata, ensure_ascii=False, indent=2))
    articles = completed_articles()
    changed = []
    math_articles = 0
    math_expressions = 0
    for index, article in enumerate(articles):
        body, conversions = format_article(article, articles[index - 1] if index else None,
                                           articles[index + 1] if index + 1 < len(articles) else None,
                                           families[article["family_slug"]])
        path = article["path"]
        old = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        if old.rstrip() == body.rstrip():
            continue
        changed.append(path.relative_to(ROOT).as_posix())
        math_articles += bool(conversions)
        math_expressions += conversions
        if not args.write:
            continue
        write(str(path.relative_to(ROOT)), body)
        metadata_path = path.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        metadata["chinese_characters"] = len(re.findall(r"[\u4e00-\u9fff]", body))
        write(str(metadata_path.relative_to(ROOT)), json.dumps(metadata, ensure_ascii=False, indent=2))
        review_path = path.parent / "review.json"
        review = json.loads(review_path.read_text(encoding="utf-8-sig"))
        review["article_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        previous_revision = review.get("formatting_revision", {})
        previous_conversions = previous_revision.get("math_delimiters_converted", 0) if previous_revision.get("date") == args.revision_date else 0
        review["formatting_revision"] = {
            **(previous_revision if previous_revision.get("date") == args.revision_date else {}),
            "date": args.revision_date,
            "navigation": "Seven stable section anchors, breadcrumb and chronological previous/next links",
            "math_delimiters_converted": previous_conversions + conversions,
            "strong_emphasis": "Unicode punctuation spacing outside code",
            "scope": "Formatting and navigation only; scientific claims are not verified by this formatter",
        }
        write(str(review_path.relative_to(ROOT)), json.dumps(review, ensure_ascii=False, indent=2))
    print(json.dumps({"articles": len(articles), "changed": len(changed), "metadata_titles_synchronized": len(title_changes), "math_articles": math_articles,
                      "math_expressions": math_expressions, "mode": "write" if args.write else "check"}, ensure_ascii=False))
    if (changed or title_changes) and not args.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
