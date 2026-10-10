"""Render the screened catalogue, family pages and audit reports from JSON."""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from collections import Counter, defaultdict
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
    target.write_bytes(content.replace("\n", newline).encode("utf-8"))


def dump(name, value):
    write(name, json.dumps(value, ensure_ascii=False, indent=2))


def escape(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


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
        articles.append({
            "family_slug": family_slug,
            "title": metadata.get("title_zh") or metadata.get("title") or metadata_path.parent.name,
            "source_version": metadata.get("source_version") or "未标注",
            "path": article_path,
        })
    return sorted(articles, key=lambda article: (article["family_slug"], article["title"].casefold()))


def article_table(articles, directory, family_names=None):
    lines = ["| " + ("模型家族 | " if family_names is not None else "") + "逐篇介绍 | 原文版本 |",
             "|" + ("---|" if family_names is not None else "") + "---|---|"]
    for article in articles:
        relative = Path(os.path.relpath(article["path"], directory)).as_posix()
        url = urllib.parse.quote(relative, safe="/.-_")
        prefix = f' {escape(family_names.get(article["family_slug"], article["family_slug"]))} |' if family_names is not None else ""
        lines.append(f'|{prefix} [{escape(article["title"])}]({url}) | {escape(article["source_version"])} |')
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
    return {"total": len(rows), "counts": counts}


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

    # Create exactly one directory per surviving model family, including singleton families.
    main_groups = defaultdict(list)
    for item in main_entries:
        main_groups[item["family"]].append(item)
    articles = completed_articles()
    articles_by_family = defaultdict(list)
    for article in articles:
        articles_by_family[article["family_slug"]].append(article)
    surviving_groups = {name: values for name, values in groups.items() if any(x["catalogue_group"] != "excluded_low_star" for x in values)}
    for name, all_items in surviving_groups.items():
        items = main_groups.get(name, [])
        exemplar = all_items[0]
        organizations = "；".join(dict.fromkeys(x["organization"] for x in all_items))
        body = f"# {name}\n\n项目主体 / 机构：{escape(organizations)}。快照：{snapshot['snapshot_date']}。\n\n本页有 {len(items)} 篇模型文献；同家族其他资料与排除项一并列出便于审计。版本按日期列出。\n\n## 模型论文 / 技术报告 / 模型卡\n\n" + table(items) + "\n\n"
        if articles_by_family[exemplar["family_slug"]]:
            body += "## 已完成的逐篇介绍\n\n" + article_table(articles_by_family[exemplar["family_slug"]], ROOT / "papers" / exemplar["family_slug"]) + "\n\n"
        other = [x for x in all_items if x not in items]
        if other:
            body += "## 同家族补充与筛选记录\n\n" + table(other) + "\n\n"
        notes = [x for x in all_items if x["warnings"] or x["scope"].get("note")]
        if notes:
            body += "## 判定依据与数据提示\n\n" + "\n".join(f'- **{escape(x["title"])}**：{escape(x["scope"].get("note"))}' + ("；" + "；".join(x["warnings"]) if x["warnings"] else "") for x in notes)
        body += "\n\n[返回项目索引](../../README.md)"
        write(f"papers/{exemplar['family_slug']}/README.md", body)

    family_names = {items[0]["family_slug"]: name for name, items in surviving_groups.items()}
    for family_slug, family_articles in articles_by_family.items():
        if family_slug not in family_names:
            body = f"# {family_slug}\n\n## 已完成的逐篇介绍\n\n" + article_table(family_articles, ROOT / "papers" / family_slug)
            write(f"papers/{family_slug}/README.md", body + "\n\n[返回项目索引](../../README.md)")
    progress = article_progress(entries)
    original_assets = 0
    without_static_assets = []
    for article in articles:
        manifest_path = article["path"].parent / "figures-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        assets = manifest.get("assets", [])
        original_assets += len(assets)
        if not assets:
            relative = Path(os.path.relpath(article["path"], ROOT)).as_posix()
            without_static_assets.append(f'[{escape(article["title"])}]({urllib.parse.quote(relative, safe="/.-_")})')

    # Keep complete directories one click away from a compact project homepage.
    family_index = ["| 模型家族 | 项目主体 / 机构 | 主目录模型文献 | Star 达标 | 无 Star 候选 | 已完成介绍 |",
                    "|---|---|---:|---:|---:|---:|"]
    for name in sorted(main_groups, key=str.casefold):
        items = main_groups[name]
        exemplar = items[0]
        passed = sum(item["star_decision"] == "passes_star" for item in items)
        organizations = "；".join(dict.fromkeys(item["organization"] for item in items))
        family_index.append(f'| [{escape(name)}](../papers/{exemplar["family_slug"]}/README.md) | '
                            f'{escape(organizations)} | {len(items)} | {passed} | {len(items)-passed} | '
                            f'{len(articles_by_family[exemplar["family_slug"]])} |')
    write("reports/model-family-index.md", f"# 模型家族完整索引\n\n"
          f"主目录共 **{len(main_groups)} 个模型家族、{len(main_entries)} 篇模型文献**。按家族名称排序，点击家族查看版本与逐篇介绍。\n\n"
          "同一官方系列合并，单版本模型独立成类；机构与家族分开记录。已完成介绍还可能包含该家族的补充材料，"
          "因此不一定等于主目录模型文献数。主目录之外的保留资料见[全部中文详解](paper-reading-guide.md)"
          "和[筛选与整理状态](article-index.md)。\n\n"
          + "\n".join(family_index) + "\n\n[返回项目首页](../README.md)")
    write("reports/paper-reading-guide.md", f"# 逐篇论文详解\n\n"
          f"已完成 **{len(articles)} 篇**中文介绍，按模型家族排序。每篇采用统一的七部分结构，结合原文图表解释模型设计、数据、训练与实验。\n\n"
          "本页包含模型论文、技术报告、模型卡及保留的补充资料；文献的实际定位在各篇开头说明。"
          "[按模型家族浏览](model-family-index.md) · [查看全部保留资料与整理状态](article-index.md)\n\n"
          + article_table(articles, ROOT / "reports", family_names) + "\n\n[返回项目首页](../README.md)")

    by_slug = {items[0]["family_slug"]: name for name, items in main_groups.items()}
    featured_slugs = ["qwen", "deepseek", "internvl", "minicpm", "llava", "kimi", "glm", "seed",
                      "gemini", "gpt", "claude", "llama", "blip", "emu", "mplug", "videollama",
                      "vita", "vila", "internlm-xcomposer", "mimo"]
    featured_links = [f'[{escape(by_slug[slug])}](papers/{slug}/README.md)'
                      for slug in featured_slugs if slug in by_slug]
    featured = ["| 模型系列 | 模型系列 | 模型系列 | 模型系列 |", "|---|---|---|---|"]
    for offset in range(0, len(featured_links), 4):
        cells = featured_links[offset:offset + 4]
        featured.append("| " + " | ".join(cells + ["—"] * (4 - len(cells))) + " |")

    examples = [
        ("papers/qwen/qwen2.5-vl/README.md", "Qwen2.5-VL", "动态分辨率、窗口注意力与绝对时间编码"),
        ("papers/deepseek/arxiv-2412.10302/README.md", "DeepSeek-VL2", "动态切图与稀疏专家"),
        ("papers/internvl/arxiv-2504.10479/README.md", "InternVL3", "联合预训练与混合偏好优化"),
        ("papers/blip/arxiv-2301.12597/README.md", "BLIP-2", "Q-Former 与冻结视觉、语言模型之间的连接"),
    ]
    complete_paths = {article["path"].relative_to(ROOT).as_posix() for article in articles}
    example_rows = [f"| [{title}]({path}) | {topic} |" for path, title, topic in examples if path in complete_paths]
    total_retained = progress["total"] if progress else len(articles)
    unfinished = total_retained - len(articles)
    pending_note = f"另有 **{unfinished} 条**尚未完成，具体状态见[完整索引](reports/article-index.md)。" if unfinished else "全部保留资料均已完成介绍。"
    static_note = ("\n\n静态原图缺项：" + "、".join(without_static_assets)
                   + "。对应发布资料仅提供动态或 HTML 图表，已在文中注明。") if without_static_assets else ""

    readme = f"""<div align="center">

# 多模态大模型技术报告

**Multimodal Model Reports**

按模型家族整理的中文论文详解 · 原文图表 · 架构、原理、数据与训练

[![Introductions](https://img.shields.io/badge/Introductions-{len(articles)}-2563eb?style=flat-square)](reports/paper-reading-guide.md)
[![Figures](https://img.shields.io/badge/Original_figures-{original_assets}-0f766e?style=flat-square)](reports/figure-crop-audit.md)
[![Snapshot](https://img.shields.io/badge/Snapshot-{snapshot['snapshot_date'].replace('-', '.')}-64748b?style=flat-square)](reports/screening-all.md)

[开始阅读](reports/paper-reading-guide.md) · [模型家族](reports/model-family-index.md) · [整理状态](reports/article-index.md) · [来源与核对](#来源与核对)

</div>

---

本项目围绕多模态大模型的设计与训练，逐篇解读论文、技术报告与模型卡，并用原文中的架构图、流程图和实验表格辅助说明。同一官方模型系列集中整理，单版本模型独立成类，便于阅读与比较。

| 已完成中文详解 | 主目录模型家族 | 原文图表资产 |
|:---:|:---:|:---:|
| **{len(articles)} 篇** | **{len(main_groups)} 个** | **{original_assets:,} 个** |

本轮保留 **{total_retained} 条资料**；主目录包含 **{len(main_entries)} 篇模型文献**，完整阅读目录另收录保留的补充材料与范围待审资料。{pending_note}

## 阅读导航

| 想做什么 | 从这里进入 |
|---|---|
| 找一篇论文，直接阅读中文详解 | [全部 {len(articles)} 篇逐篇论文详解 →](reports/paper-reading-guide.md) |
| 按模型系列阅读，比较不同版本 | [{len(main_groups)} 个模型家族完整索引 →](reports/model-family-index.md) |
| 查文献类型、收录情况与整理进度 | [全部 {total_retained} 条保留资料与状态 →](reports/article-index.md) |

## 模型家族索引

以下为部分系列的快捷入口；[完整索引](reports/model-family-index.md)包含全部主目录家族及其机构、文献数量和筛选状态。

{"\n".join(featured)}

家族按官方系列与版本关系划分。Qwen、DeepSeek 各自合并为一类；同厂商的不同家族分别整理，第三方衍生模型独立记录。

<a id="论文图文介绍"></a>

## 逐篇论文详解

每篇介绍采用以下七部分结构，原文未披露的细节会在对应位置注明。

| 内容 | 主要回答的问题 |
|---|---|
| **01 · 论文信息与模型定位** | 谁提出了模型？面向什么任务？与前代或相关工作是什么关系？ |
| **02 · 快速概览** | 主要贡献是什么？有哪些模型规格与能力边界？ |
| **03 · 整体架构** | 各模态如何编码、连接、融合与输出？结合原文模型图说明。 |
| **04 · 核心创新与数学原理** | 关键机制如何工作？相关公式与设计动机是什么？ |
| **05 · 数据构建与处理** | 数据来自哪里？如何清洗、合成、配比与预处理？ |
| **06 · 训练流程与目标** | 分哪些阶段？训练哪些参数？优化目标与损失如何设计？ |
| **07 · 实验设置与主要结果** | 如何评测？主要结果、消融实验和适用范围是什么？ |

可以先从这些介绍查看内容形式：

| 阅读示例 | 主要内容 |
|---|---|
{"\n".join(example_rows)}

**[查看全部逐篇详解 →](reports/paper-reading-guide.md)**

配图只保留用到的原文图表区域，保留图例、表头与必要脚注；各篇记录图表来源与原文版本。{static_note}

## 收录范围与筛选

来源为 [Awesome Multimodal Large Language Models]({snapshot['upstream_repository']}) 的 **Multimodal Instruction Tuning (& Latest Works)** 与 **Foundation Models** 两个章节，数据快照为 **{snapshot['snapshot_date']}**。

按 Code 栏对应仓库的 Star 筛选：**小于 {snapshot['threshold']} 的条目排除**；未提供 GitHub 的条目记为 **N/A**，另作候选或范围待审。Star 是仓库关注度指标，不能直接代表论文质量。

<details>
<summary><strong>展开筛选统计与归类口径</strong></summary>

<a id="首轮筛选结果"></a>

### 首轮筛选结果

| 项目 | 数量 |
|---|---:|
| Multimodal Instruction Tuning (& Latest Works) 原始条目 | {snapshot['section_rows']['Multimodal Instruction Tuning (& Latest Works)']} |
| Foundation Models 原始条目 | {snapshot['section_rows']['Foundation Models']} |
| 原始总记录 / 去重后文献 | {snapshot['raw_rows']} / {len(entries)} |
| 原表 GitHub 仓库标识 / 重定向后实际仓库 | {len(stars)} / {len(canonical_groups)} |
| Star ≥ {snapshot['threshold']} 的文献 | {stats['star_decisions'].get('passes_star', 0)} |
| Star < {snapshot['threshold']} 的排除文献 | {len(low)} |
| 原表未提供 GitHub 的文献 | {len(no_stars)} |
| 主目录模型文献 / 家族 | {len(main_entries)} / {len(main_groups)} |

### 筛选口径

1. 只筛选上述两个章节，按 arXiv ID 或题名去重。GPT-5 发布资料和 Gemini 1 技术报告各跨章节出现一次。
2. 使用 Code 栏对应仓库的整数 Star；恰好 {snapshot['threshold']} 保留，数值以本次快照为准。
3. 原表没有 GitHub 的记录不视为 0 Star，也不宣称已经通过 Star 门槛。
4. 模型研究论文、技术报告与模型卡进入主目录；数据集、评测、综述、工具、通用方法和发布博客在补充清单中保留。逐篇介绍依据原文说明实际文献定位。
5. 机构与模型家族分开记录；未核实的机构明确标注。第三方基于 LLaVA、BLIP 等训练的独立模型不自动并入官方家族。

“Star 达标”包含全部文献类型，不等于技术报告数量。各审计清单可能交叉，不能相加作为总数。

原表 Step3 的徽章指向 GLM-4.1V-Thinking，本项目按实际 Code 仓库 stepfun-ai/Step3 计数。LAVIS、UniLM 等共享仓库的 Star 是母仓库指标；旧仓库重定向时按当前目标计数，相关条目均保留提示。

</details>

<a id="全量清单与审计"></a>

## 来源与核对

| 资料 | 阅读入口 |
|---|---|
| 筛选台账 | [全部 {len(entries)} 篇](reports/screening-all.md) · [低于门槛 {len(low)} 篇](reports/excluded-low-star.md) · [无 GitHub 指标 {len(no_stars)} 篇](reports/no-github.md) |
| 文献范围 | [补充与非模型文献 {len(supplements)} 篇](reports/supporting-materials.md) · [范围待审 {len(review)} 篇](reports/scope-review.md) |
| 图表来源 | [裁切与替换记录](reports/figure-crop-audit.md) · [逐图坐标与来源](data/figure-crop-audit.json) |
| 原文抽查 | [四篇介绍抽查](reports/final-source-spotcheck.md) · [Seed1.8 抽查](reports/seed18-source-spotcheck.md) |
| 数据与检查 | [文献数据](data/catalogue.json) · [统计](data/statistics.json) · [Star 证据](data/stars.json) · [数据与链接检查](data/validation.json) · [独立复核](data/quality-audit.json) · [文章与图片检查](data/article-verification.json) |

## 项目结构与更新

<details>
<summary><strong>查看目录结构与维护方式</strong></summary>

```text
papers/
└── <family>/
    ├── README.md                 家族介绍与版本入口
    └── <paper>/
        ├── README.md             七部分中文详解
        ├── figures/              原文图表
        ├── figures-manifest.json 图表编号、页码、坐标与来源
        ├── metadata.json         文献信息与整理状态
        ├── evidence.json         原文版本与资产记录
        └── review.json           来源核对记录
reports/                          阅读目录、筛选清单与核对报告
data/                             文献数据、分类规则与 Star 证据
source/awesome-upstream.md        上游清单快照
scripts/                          数据采集、目录生成与图表提取
```

更新公开仓库 Star 后重建目录（Python 3.12+）：

```bash
python scripts/catalog.py --refresh-stars --snapshot-date YYYY-MM-DD
python scripts/build_reports.py
python scripts/verify_catalogue.py
```

更新上游清单时加 `--refresh-source`；新题名须补充 `data/scope-mapping.json`。家族归类优先读取 `data/family-mapping.json` 的 `by_entry_id`，新记录再按题名与仓库规则匹配。调整后先审阅 JSON 再重建目录；取消整个家族时应移走对应旧目录，避免遗留页面。

PDF 配图须显式指定裁切坐标。`scripts/paper_sources.py assets --specs ...` 提取图表，`scripts/crop_paper_figures.py render/apply` 用于复核后批量替换已有图片。固定原文版本、页码、裁切范围与图片哈希记录在各篇 `figures-manifest.json`；整页渲染仅用于本地阅读。

修改文章后运行 `python scripts/verify_articles.py` 检查七节结构、图片引用与文件哈希，再运行 `python scripts/build_reports.py` 更新首页和阅读入口。`review.json` 与 `evidence.json` 是整理过程的来源记录，不代表外部同行评议；自动检查不证明数值与解释的语义准确性。

PDF/网页阅读缓存位于 `source/papers/`，由 `.gitignore` 排除；原图与出处清单保留在文章目录。

</details>

---

文献清单来源于 [BradyFU/Awesome-Multimodal-Large-Language-Models]({snapshot['upstream_repository']})。图表来自对应论文或官方发布资料，具体出处见各篇介绍。
"""
    write("README.md", readme)
    print(json.dumps({k: v for k, v in stats.items() if k != "warnings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
