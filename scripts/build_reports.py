"""Render the screened catalogue, family pages and audit reports from JSON."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def write(name, text):
    target = ROOT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = target.read_bytes() if target.exists() else b""
    newline = "\r\n" if previous.count(b"\r\n") > previous.count(b"\n") / 2 else "\n"
    content = (text.rstrip() + "\n").replace("\r\n", "\n")
    encoded = content.replace("\n", newline).encode("utf-8")
    if encoded == previous:
        return
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".report-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
    try:
        for attempt in range(5):
            try:
                os.replace(temporary, target)
                break
            except OSError as error:
                if error.errno not in (13, 22) or attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def dump(name, value):
    write(name, json.dumps(value, ensure_ascii=False, indent=2))


def escape(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def organization_label(value):
    """Display repository owners as owners, without guessing affiliations."""
    if value.startswith("repo:"):
        owner = value.removeprefix("repo:")
        return f'[{escape(owner)}](https://github.com/{urllib.parse.quote(owner, safe="-_")})'
    return "论文作者" if value == "authors" else escape(value)


def family_for(entry, mapping):
    exact = mapping.get("exact_title_overrides", {})
    repos = mapping.get("repo_overrides", {})
    result = mapping.get("by_entry_id", {}).get(entry["id"]) or exact.get(entry["title"]) or repos.get((entry["repo"] or "").lower())
    if not result:
        for rule in mapping.get("title_rules", []):
            if re.search(rule.get("pattern", rule.get("regex", "(?!)")), entry["title"], re.I):
                result = rule
                break
    if isinstance(result, str):
        result = {"family": result}
    if result:
        family = result.get("name", result.get("family", result.get("model_family")))
        organization_id = result.get("organization", result.get("manufacturer", "未核实"))
        organization = mapping.get("organizations", {}).get(organization_id, {}).get("name", organization_id)
    else:
        family = (entry["repo"] or "").split("/")[-1] or entry["title"].split(":")[0]
        organization = "仓库主体：" + entry["repo"].split("/")[0] if entry["repo"] else "未核实"
    slug_source = result.get("family", family) if result else family
    slug = re.sub(r"[^a-z0-9]+", "-", slug_source.lower()).strip("-") or "family"
    return family, organization, slug


def paper_link(entry):
    title = escape(entry["title"])
    return f'[{title}]({entry["paper_url"]})' if entry["paper_url"] else title + "（原表未提供报告链接）"


def star_label(entry):
    observation = entry.get("star_observation")
    if not observation:
        return "N/A（原表无 GitHub）"
    if observation.get("stars") is not None:
        return f'[{observation["stars"]:,}]({observation["source_url"]})'
    return escape(observation.get("badge_value", "未确认"))


def completed_articles():
    """Discover finished introductions without touching their source files."""
    publication_dates = {
        (ROOT / entry["destination"]).resolve(): entry["date"]
        for entry in read("data/article-plan.json")["entries"]
    }
    articles = []
    for metadata_path in sorted((ROOT / "papers").glob("*/*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("article_status") != "complete":
            continue
        article_file = metadata.get("article_file")
        family_slug = metadata.get("family_slug")
        if not isinstance(article_file, str) or not article_file or not isinstance(family_slug, str):
            continue
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", family_slug):
            continue
        relative_path = Path(article_file)
        if relative_path.is_absolute():
            continue
        article_path = (metadata_path.parent / relative_path).resolve()
        if not article_path.is_relative_to(metadata_path.parent.resolve()) or not article_path.is_file():
            continue
        publication_date = metadata.get("source_first_submission") or publication_dates[metadata_path.parent.resolve()]
        date.fromisoformat(publication_date)
        articles.append({
            "date": publication_date,
            "family_slug": family_slug,
            "title": metadata.get("title_zh") or metadata.get("title") or metadata_path.parent.name,
            "source_version": metadata.get("source_version") or "未标注",
            "path": article_path,
        })
    return sorted(articles, key=lambda article: (article["date"], article["family_slug"], article["title"].casefold()))


def article_table(articles, directory, family_names=None):
    lines = ["| 发表 / 发布日期 | " + ("模型系列 | " if family_names is not None else "") + "逐篇介绍 |",
             "|---|" + ("---|" if family_names is not None else "") + "---|"]
    for article in articles:
        relative = Path(os.path.relpath(article["path"], directory)).as_posix()
        url = urllib.parse.quote(relative, safe="/.-_")
        prefix = f' {escape(family_names.get(article["family_slug"], article["family_slug"]))} |' if family_names is not None else ""
        lines.append(f'| {article["date"]} |{prefix} [{escape(article["title"])}]({url}) |')
    return "\n".join(lines)


def reading_guide(articles, family_names):
    years = defaultdict(list)
    for article in articles:
        years[article["date"][:4]].append(article)
    jumps = " · ".join(f'[{year}（{len(items)} 篇）](#year-{year})' for year, items in sorted(years.items()))
    lines = ["# 逐篇论文详解", "",
             f"**{len(articles)} 篇中文介绍** · 按发表 / 发布日期由早到晚排列。", "",
             jumps, "", "[按模型系列浏览](model-family-index.md) · [项目首页](../README.md)"]
    for year, items in sorted(years.items()):
        lines += ["", f'<a id="year-{year}"></a>', "", f"## {year} · {len(items)} 篇", "",
                  article_table(items, ROOT / "reports", family_names)]
    lines += ["", "[返回顶部](#逐篇论文详解) · [项目首页](../README.md)"]
    return "\n".join(lines)


def family_paper_table(items, introductions, directory):
    """List each retained source once, beside its local introduction."""
    lines = ["| 论文 / 资料 | 日期 | 中文介绍 | 原文 |", "|---|---|---|---|"]
    labels = {"source_unavailable": "原文待获取", "planned": "待撰写",
              "writing": "撰写中", "needs_review": "待复核"}
    for item in sorted(items, key=lambda item: (item["date"], item["title"].casefold())):
        introduction = introductions.get(item["id"], {})
        status = introduction.get("article_status", "planned")
        label = "阅读全文" if status == "complete" else labels.get(status, "待整理")
        if introduction.get("document_exists"):
            path = ROOT / introduction["article_path"]
            relative = Path(os.path.relpath(path, directory)).as_posix()
            reading_link = f'[{label}]({urllib.parse.quote(relative, safe="/.-_")})'
        else:
            reading_link = labels.get(status, "待整理")
        source_url = item.get("paper_url") or introduction.get("source_url") or item.get("github_code_url")
        source_link = f'[原文]({source_url})' if source_url else "—"
        lines.append(f'| {escape(item["title"])} | {escape(item["date"])} | {reading_link} | {source_link} |')
    return "\n".join(lines)


def article_progress(entries):
    """List every retained document; do not present unfinished notes as complete."""
    plan_path = ROOT / "data/article-plan.json"
    if not plan_path.exists():
        return None
    rows = []
    for row in read("data/article-plan.json")["entries"]:
        destination = ROOT / row["destination"]
        metadata = read(str((destination / "metadata.json").relative_to(ROOT)))
        document = destination / metadata.get("article_file", "README.md")
        rows.append({**row, "article_status": metadata.get("article_status", "planned"),
                     "document_exists": document.is_file(), "article_path": document.relative_to(ROOT).as_posix()})
    counts = dict(Counter(row["article_status"] for row in rows))
    dump("data/article-progress.json", {"total": len(rows), "status_counts": counts, "entries": rows})
    labels = {"complete": "已完成", "planned": "待撰写", "writing": "撰写中", "source_unavailable": "原文待获取", "needs_review": "待复核"}
    lines = ["# 全部保留资料的图文介绍", "", f"本轮范围共 **{len(rows)} 条**，沿用七部分模板。状态以逐篇元数据为准，待获取原文的说明页不计为完整技术介绍。", "",
             "| 家族 | 文献与阅读入口 | 文献类型 | 整理状态 |", "|---|---|---|---|"]
    for row in rows:
        title = escape(row["title"])
        if row["document_exists"]:
            title = f'[{title}](../{row["article_path"]})'
        elif row.get("source_url"):
            title = f'[{title}]({row["source_url"]})'
        lines.append(f'| {escape(row["family"])} | {title} | {escape(row["document_kind"])} | {labels.get(row["article_status"], row["article_status"])} |')
    lines += ["", "[返回项目](../README.md)"]
    write("reports/article-index.md", "\n".join(lines))
    return {"total": len(rows), "counts": counts, "entries": rows}


STATUS_LABELS = {"passes_star": "Star 达标", "excluded_low_star": "排除：Star < 100", "not_assessable": "候选：无 Star 指标", "unresolved_star": "待核验 Star"}
SCOPE_LABELS = {"include_candidate": "模型文献", "supplementary": "补充材料", "exclude_non_model": "非模型文献", "needs_scope_review": "范围待审"}


def table(entries, include_scope=True):
    header = "| 家族 | 文献 / 发布资料 | 日期 | Star | 仓库 | 筛选结果 |"
    separator = "|---|---|---|---:|---|---|"
    if include_scope:
        header += " 文献类型 / 范围 |"
        separator += "---|"
    lines = [header, separator]
    for item in sorted(entries, key=lambda x: (x["family"].casefold(), x["date"], x["title"])):
        repo = f'[{escape(item["repo"])}]({item["github_code_url"]})' if item["repo"] else "—"
        row = f'| {escape(item["family"])} | {paper_link(item)} | {escape(item["date"])} | {star_label(item)} | {repo} | {STATUS_LABELS[item["star_decision"]]} |'
        if include_scope:
            row += f' {escape(item["scope"]["kind"])} / {SCOPE_LABELS[item["scope"]["recommend_scope"]]} |'
        lines.append(row)
    return "\n".join(lines)


def main():
    entries = read("data/parsed-entries.json")
    stars = read("data/stars.json")
    snapshot = read("data/snapshot.json")
    families = read("data/family-mapping.json")
    scopes = read("data/scope-mapping.json")
    if "entries" in scopes:
        scopes = scopes["entries"]
    canonical_groups = defaultdict(set)
    for repo, observation in stars.items():
        final_url = observation.get("final_url", f"https://github.com/{repo}")
        canonical_repo = urllib.parse.urlparse(final_url).path.strip("/")
        observation["canonical_repo"] = canonical_repo
        canonical_groups[canonical_repo.casefold()].add(repo)
    missing = sorted(set(x["title"] for x in entries) - set(scopes))
    assert not missing, f"Missing scope mappings: {missing}"
    slugs = {}
    for item in entries:
        family, organization, slug = family_for(item, families)
        if slug in slugs and slugs[slug] != family:
            slug += "-" + hashlib.sha256(family.encode()).hexdigest()[:6]
        slugs[slug] = family
        item.update({"family": family, "organization": organization, "family_slug": slug, "scope": scopes[item["title"]]})
        observation = stars.get(item["repo"])
        item["star_observation"] = observation
        item["canonical_repo"] = observation.get("canonical_repo") if observation else None
        if observation:
            canonical_repo = observation["canonical_repo"]
            if canonical_repo.casefold() != item["repo"].casefold():
                item["warnings"].append(f"Code repository redirects to {canonical_repo}; stars are the current destination repository's stars")
            if len(canonical_groups[canonical_repo.casefold()]) > 1:
                item["warnings"].append(f"Several upstream Code repository identifiers share {canonical_repo}; no independent version-level star count")
        if not item["repo"]:
            decision = "not_assessable"
        elif observation and observation.get("stars") is not None:
            decision = "passes_star" if observation["stars"] >= snapshot["threshold"] else "excluded_low_star"
        elif observation and observation.get("status") == "above_threshold_rounded" and snapshot["threshold"] <= 900:
            decision = "passes_star"
        else:
            decision = "unresolved_star"
        item["star_decision"] = decision
        # The main project includes named-model research and model cards; other
        # work stays in the audit, even if its shared repository has many stars.
        item["catalogue_group"] = "model_catalogue" if scopes[item["title"]]["recommend_scope"] == "include_candidate" and decision in ("passes_star", "not_assessable") else "excluded_low_star" if decision == "excluded_low_star" else "scope_review" if scopes[item["title"]]["recommend_scope"] == "needs_scope_review" else "supporting_materials"
    assert len(entries) == snapshot["unique_entries"]
    assert sum(len(x["source_sections"]) for x in entries) == snapshot["raw_rows"]
    assert all(x["star_observation"]["stars"] < snapshot["threshold"] for x in entries if x["star_decision"] == "excluded_low_star")
    assert all(x["star_decision"] == "not_assessable" for x in entries if not x["repo"])
    groups = defaultdict(list)
    for item in entries:
        groups[item["family"]].append(item)
    main_entries = [x for x in entries if x["catalogue_group"] == "model_catalogue"]
    low = [x for x in entries if x["star_decision"] == "excluded_low_star"]
    no_stars = [x for x in entries if x["star_decision"] == "not_assessable"]
    supplements = [x for x in entries if x["scope"]["recommend_scope"] in ("supplementary", "exclude_non_model")]
    review = [x for x in entries if x["scope"]["recommend_scope"] == "needs_scope_review"]
    stats = {"snapshot_date": snapshot["snapshot_date"], "raw_rows": snapshot["raw_rows"], "unique_entries": len(entries),
             "unique_code_repository_identifiers": len(stars), "canonical_repositories": len(canonical_groups), "star_decisions": dict(Counter(x["star_decision"] for x in entries)),
             "scope_decisions": dict(Counter(x["scope"]["recommend_scope"] for x in entries)),
             "catalogue_groups": dict(Counter(x["catalogue_group"] for x in entries)),
             "main_model_entries": len(main_entries), "main_families": len({x["family"] for x in main_entries}),
             "total_families_in_audit": len(groups), "warnings": [{"title": x["title"], "warnings": x["warnings"]} for x in entries if x["warnings"]]}
    dump("data/catalogue.json", {"metadata": snapshot, "statistics": stats, "entries": entries})
    dump("data/statistics.json", stats)

    intro = f"数据快照：**{snapshot['snapshot_date']}**。范围仅包含 [原项目]({snapshot['upstream_repository']}) 的两个指定章节。\n\nGitHub Star 反映仓库关注度，用作本轮可复核的门槛，不直接衡量论文质量。数值来自 Code 栏对应仓库；共享仓库的 Star 不属于单篇论文。"
    write("reports/screening-all.md", "# 完整首轮筛选台账\n\n" + intro + "\n\n" + table(entries) + "\n\n## 记录说明\n\n每篇的原表章节、行号、官方链接、范围判定依据与 Star 证据保存在 [catalogue.json](../data/catalogue.json)。原表无 GitHub 只表示不能用此指标判定，不能视作 0 Star。")
    write("reports/excluded-low-star.md", "# 按 Star 门槛排除的条目\n\n" + intro + f"\n\n共 **{len(low)}** 篇，均有可确认的仓库整数 Star，且严格小于 {snapshot['threshold']}。排除只作用于本项目的筛选目录；原始记录保留。\n\n" + table(low))
    write("reports/no-github.md", "# 原表未提供 GitHub 仓库的条目\n\n" + intro + f"\n\n共 **{len(no_stars)}** 篇。原表只提供 Hugging Face、博客、论文或 Coming soon 的，也属于这一组；未断言这些论文在其他地方没有代码。模型论文和正式模型卡保留候选；其余按文献范围处理。\n\n" + table(no_stars))
    write("reports/supporting-materials.md", "# 补充材料与非模型文献\n\n" + intro + f"\n\n共 **{len(supplements)}** 篇，包含方法、数据、评测、综述、工具和发布资料。它们不进入模型技术报告主目录，但保留供项目后续参考；Star 筛选结果仍逐项显示。\n\n" + table(supplements))
    write("reports/scope-review.md", "# 纳入范围待审\n\n" + intro + f"\n\n共 **{len(review)}** 篇。包括是否属于多模态大模型、是否纳入具身与世界模型、材料是否足够构成模型报告等边界。具体依据见下表。\n\n" + table(review) + "\n\n" + "\n".join(f'- **{escape(x["title"])}**：{escape(x["scope"].get("note"))}' for x in review))

    # Create one concise reading list per surviving family, including singletons.
    articles = completed_articles()
    progress = article_progress(entries)
    introductions = {row["id"]: row for row in progress["entries"]} if progress else {}
    articles_by_family = defaultdict(list)
    for article in articles:
        articles_by_family[article["family_slug"]].append(article)
    retained_groups = defaultdict(list)
    for item in entries:
        if item["catalogue_group"] != "excluded_low_star":
            retained_groups[item["family"]].append(item)
    for name, items in retained_groups.items():
        exemplar = items[0]
        directory = ROOT / "papers" / exemplar["family_slug"]
        organizations = "；".join(organization_label(value) for value in dict.fromkeys(item["organization"] for item in items))
        body = (f"[项目首页](../../README.md) · [模型系列索引](../../reports/model-family-index.md)\n\n# {name}\n\n"
                f"{organizations}\n\n**{len(items)} 篇论文与资料** · 按发布日期由早到晚排列。\n\n## 论文列表\n\n"
                + family_paper_table(items, introductions, directory)
                + "\n\n[返回系列索引](../../reports/model-family-index.md) · [返回项目首页](../../README.md)")
        write(f"papers/{exemplar['family_slug']}/README.md", body)

    family_names = {items[0]["family_slug"]: name for name, items in retained_groups.items()}
    for family_slug, family_articles in articles_by_family.items():
        if family_slug not in family_names:
            body = f"# {family_slug}\n\n## 论文列表\n\n" + article_table(family_articles, ROOT / "papers" / family_slug)
            write(f"papers/{family_slug}/README.md", body + "\n\n[返回项目索引](../../README.md)")
    original_assets = 0
    for article in articles:
        manifest_path = article["path"].parent / "figures-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        assets = manifest.get("assets", [])
        original_assets += len(assets)

    # Keep complete directories one click away from a compact project homepage.
    letter_groups = defaultdict(list)
    for name in sorted(retained_groups, key=str.casefold):
        letter = name[0].upper()
        letter_groups[letter if "A" <= letter <= "Z" else "0–9" if letter.isdigit() else "其他"].append(name)
    anchors = {letter: "series-" + ("0-9" if letter == "0–9" else "other" if letter == "其他" else letter.lower()) for letter in letter_groups}
    family_index = ["# 模型系列索引", "",
                    f"**{len(retained_groups)} 个模型系列 · {sum(map(len, retained_groups.values()))} 篇论文与资料**", "",
                    " · ".join(f'[{letter}](#{anchors[letter]})' for letter in sorted(letter_groups)), "",
                    "[按发表时间阅读](paper-reading-guide.md) · [项目首页](../README.md)"]
    for letter in sorted(letter_groups):
        family_index += ["", f'<a id="{anchors[letter]}"></a>', "", f"## {letter}", "",
                         "| 模型系列 | 项目主体 / 机构 | 论文数量 |", "|---|---|---:|"]
        for name in letter_groups[letter]:
            items = retained_groups[name]
            exemplar = items[0]
            organizations = "；".join(organization_label(value) for value in dict.fromkeys(item["organization"] for item in items))
            family_index.append(f'| [{escape(name)}](../papers/{exemplar["family_slug"]}/README.md) | {organizations} | {len(items)} |')
    family_index += ["", "[返回顶部](#模型系列索引) · [项目首页](../README.md)"]
    write("reports/model-family-index.md", "\n".join(family_index))
    write("reports/paper-reading-guide.md", reading_guide(articles, family_names))

    by_slug = {items[0]["family_slug"]: name for name, items in retained_groups.items()}
    featured_slugs = ["qwen", "deepseek", "internvl", "minicpm", "llava", "kimi", "glm", "seed",
                      "gemini", "gpt", "claude", "llama", "blip", "emu", "mplug", "videollama",
                      "vita", "vila", "internlm-xcomposer", "mimo"]
    short_names = {"minicpm": "MiniCPM", "llava": "LLaVA", "glm": "GLM", "blip": "BLIP", "vila": "VILA", "internlm-xcomposer": "XComposer"}
    featured_links = [f'[{escape(short_names.get(slug, by_slug[slug]))}](papers/{slug}/README.md)'
                      for slug in featured_slugs if slug in by_slug]
    featured_columns = 3
    featured = ["| " + " | ".join(["模型系列"] * featured_columns) + " |",
                "|" + "---|" * featured_columns]
    for offset in range(0, len(featured_links), featured_columns):
        cells = featured_links[offset:offset + featured_columns]
        featured.append("| " + " | ".join(cells + ["—"] * (featured_columns - len(cells))) + " |")

    readme = f"""<div align="center">

# 多模态大模型技术报告

**Multimodal Model Reports**

多模态大模型的中文图文详解 · 架构 · 原理 · 数据 · 训练 · 实验

[![Introductions](https://img.shields.io/badge/Introductions-{len(articles)}-2563eb?style=flat-square)](reports/paper-reading-guide.md)
[![Series](https://img.shields.io/badge/Model_series-{len(retained_groups)}-7c3aed?style=flat-square)](reports/model-family-index.md)
[![Figures](https://img.shields.io/badge/Original_figures-{original_assets}-0f766e?style=flat-square)](reports/paper-reading-guide.md)

[开始阅读](reports/paper-reading-guide.md) · [模型系列](reports/model-family-index.md)

<picture>
  <source media="(max-width: 600px)" srcset="assets/multimodal-overview-mobile.svg">
  <img src="assets/multimodal-overview.svg" alt="多模态大模型概览：图像、视频、音频与文本，以及理解、推理与生成能力" width="1280">
</picture>

</div>

---

聚焦多模态大模型的整体架构、核心原理、数据与训练，结合原文图表解读实验结果。支持按论文阅读，也可按模型系列比较不同版本。

## 阅读导航

| 阅读方式 | 入口 |
|---|---|
| 按发表时间阅读中文详解（由早到晚） | [全部 {len(articles)} 篇逐篇论文详解 →](reports/paper-reading-guide.md) |
| 按模型系列比较不同版本 | [{len(retained_groups)} 个模型系列完整索引 →](reports/model-family-index.md) |

## 模型系列

点击下方系列查看论文列表，更多系列见[完整索引](reports/model-family-index.md)。

{"\n".join(featured)}

<a id="论文图文介绍"></a>

## 逐篇论文详解

[按发表时间阅读全部 {len(articles)} 篇 →](reports/paper-reading-guide.md)

每篇详解包含以下七部分：

| 内容 | 主要回答的问题 |
|---|---|
| **01 · 论文信息与模型定位** | 谁提出了模型？面向什么任务？与前代或相关工作是什么关系？ |
| **02 · 快速概览** | 主要贡献是什么？有哪些模型规格与能力边界？ |
| **03 · 整体架构** | 各模态如何编码、连接、融合与输出？结合架构图逐步说明。 |
| **04 · 核心创新与数学原理** | 关键机制如何工作？相关公式与设计动机是什么？ |
| **05 · 数据构建与处理** | 数据来自哪里？如何清洗、合成、配比与预处理？ |
| **06 · 训练流程与目标** | 分哪些阶段？训练哪些参数？优化目标与损失如何设计？ |
| **07 · 实验设置与主要结果** | 如何评测？主要结果、消融实验和适用范围是什么？ |
"""
    write("README.md", readme)
    print(json.dumps({k: v for k, v in stats.items() if k != "warnings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
