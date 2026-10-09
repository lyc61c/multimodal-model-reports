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
    target.write_text(text.rstrip() + "\n", encoding="utf-8")


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
    article_index = article_table(articles, ROOT, family_names) if articles else "当前尚无已完成的逐篇介绍。"
    progress = article_progress(entries)
    progress_note = (f'本轮全部 **{progress["total"]} 条**保留资料，已完成 **{progress["counts"].get("complete", 0)} 篇**介绍。'
                     f'其余状态与阅读入口见[完整介绍索引](reports/article-index.md)。') if progress else ""
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
    progress_note += f' 已附 **{original_assets:,} 个**原始图表资产，各文记录来源、版本和图表位置。'
    if without_static_assets:
        progress_note += " 以下发布资料仅提供动态或 HTML 图表，静态原图缺项已在对应文档注明：" + "、".join(without_static_assets) + "。"

    priority = ["Qwen", "DeepSeek", "InternVL", "MiniCPM", "LLaVA", "VITA", "Kimi", "GLM", "Gemini", "GPT", "Claude", "Llama", "Emu", "BLIP", "VideoLLaMA", "mPLUG", "Phi", "MiMo"]
    def order(name):
        return (next((i for i, term in enumerate(priority) if name.casefold().startswith(term.casefold())), len(priority)), name.casefold())
    index = ["| 模型家族 | 项目主体 / 机构 | 模型文献 | Star 达标 | 无 Star 候选 |", "|---|---|---:|---:|---:|"]
    for name in sorted(main_groups, key=order):
        items = main_groups[name]
        x = items[0]
        passed = sum(item["star_decision"] == "passes_star" for item in items)
        organizations = "；".join(dict.fromkeys(item["organization"] for item in items))
        index.append(f'| [{escape(name)}](papers/{x["family_slug"]}/README.md) | {escape(organizations)} | {len(items)} | {passed} | {len(items)-passed} |')
    readme = f"""# 多模态大模型技术报告

[GitHub 仓库](https://github.com/lyc61c/multimodal-model-reports) · [全部介绍目录](reports/article-index.md)

按模型家族整理多模态大模型论文、技术报告与模型卡。Qwen 系列合并为一个家族，DeepSeek 系列合并为一个家族；单版本模型单独建立家族。机构与模型家族分开记录，同厂商的不同模型家族保持独立。

{intro}

{progress_note}

配图仅展示正文引用的原文图表区域，保留图例、表头与必要脚注，去掉周围正文及页眉页脚。

## 首轮筛选结果

| 项目 | 数量 |
|---|---:|
| Multimodal Instruction Tuning (& Latest Works) 原始条目 | {snapshot['section_rows']['Multimodal Instruction Tuning (& Latest Works)']} |
| Foundation Models 原始条目 | {snapshot['section_rows']['Foundation Models']} |
| 原始总记录 / 去重后文献 | {snapshot['raw_rows']} / {len(entries)} |
| 原表 GitHub 仓库标识（逐一核验） | {len(stars)} |
| 重定向后的实际 GitHub 仓库 | {len(canonical_groups)} |
| Star ≥ 100 的文献 | {stats['star_decisions'].get('passes_star', 0)} |
| Star < 100 的排除文献 | {len(low)} |
| 原表未提供 GitHub 的文献 | {len(no_stars)} |
| 主目录模型文献 / 家族 | {len(main_entries)} / {len(main_groups)} |

“Star 达标”统计包含全部文献类型，不等于技术报告数量。主目录收录首轮判为模型研究论文、技术报告与模型卡的候选；博客和发布页单列补充材料。目录保留首轮文献类型判定，逐篇介绍再根据原文说明实际定位；综述、评测、方法或发布页不会被写成作者新训练的模型。无 GitHub 指标的模型文献为候选，未宣称通过 Star 门槛。

## 筛选口径

1. 只筛选指定两个章节；按论文 arXiv ID 或题名去重。GPT-5 发布资料和 Gemini 1 技术报告各跨章节出现一次。
2. 使用 Code 栏对应 GitHub 仓库的精确整数 Star。小于 100 排除，恰好 100 保留。Star 随时间变化，应以本次快照为准。
3. 原表没有 GitHub 仓库的记录记为 N/A，保留候选或范围待审。没有把闭源报告当作 0 Star。
4. 数据集、评测、综述、工具、通用方法和发布博客不作为模型技术报告，独立列在附录。Star 达标不改变文献类型。
5. 家族依据发布主体与版本关系划分。第三方基于 LLaVA、BLIP 等训练的独立模型不自动并入原厂官方家族；仓库主体未核实的机构会明确标注。

## 模型家族索引

{"\n".join(index)}

## 论文图文介绍

当前已完成 **{len(articles)} 篇**逐篇介绍，按已审核的七部分模板组织，并引用原文图表。

{progress_note}

{article_index}

## 全量清单与审计

- [完整首轮筛选台账：全部 {len(entries)} 篇](reports/screening-all.md)
- [低于 100 Star：{len(low)} 篇](reports/excluded-low-star.md)
- [原表无 GitHub 指标：{len(no_stars)} 篇](reports/no-github.md)
- [补充材料与非模型文献：{len(supplements)} 篇](reports/supporting-materials.md)
- [纳入范围待审：{len(review)} 篇](reports/scope-review.md)
- [完整机器可读数据](data/catalogue.json) · [统计](data/statistics.json) · [Star 证据](data/stars.json)
- [数据完整性与链接验证](data/validation.json) · [独立复核](data/quality-audit.json)
- [文章结构与原图来源检查](data/article-verification.json) · [四篇介绍的独立原文抽查](reports/final-source-spotcheck.md)
- [Seed1.8 介绍的独立原文抽查](reports/seed18-source-spotcheck.md)
- [整页截图替换与图表裁切记录](reports/figure-crop-audit.md) · [逐图坐标与来源](data/figure-crop-audit.json)

各审计清单按不同维度切分，可能交叉，不应相加作为总数。原表 Step3 的徽章指向 GLM-4.1V-Thinking；本项目按实际 Code 仓库 stepfun-ai/Step3 计数。LAVIS、UniLM 等共享仓库的 Star 是母仓库指标，不能当作子模型单独的 Star。Qwen2-VL、Qwen2.5-VL 等旧地址重定向到新版本仓库，Star 按当前目标记录，逐篇标注重定向与共仓库关系。

## 项目结构与更新

```text
papers/<family>/README.md     每个模型家族的版本与论文链接
papers/<family>/<paper>/README.md        逐篇图文介绍
papers/<family>/<paper>/metadata.json    介绍状态、标题与原文版本
papers/<family>/<paper>/figures/         从原文提取的图表
papers/<family>/<paper>/figures-manifest.json  图表编号、页码与来源
reports/                     全量筛选、排除与待审清单
data/catalogue.json          合并后的模型、范围和 Star 数据
data/stars.json              仓库 Star、采集时间与证据
data/family-mapping.json     可编辑的家族归类规则
data/scope-mapping.json      可编辑的文献类型与纳入依据
source/awesome-upstream.md  原清单快照
scripts/                    数据采集、目录生成与可选 PDF 图表提取脚本
```

目录采集与生成脚本使用 Python 3.12+ 标准库，无第三方依赖；可选的 PDF 图表提取脚本需要 `pdfplumber`。更新公开仓库 Star 后重建目录：

```bash
python scripts/catalog.py --refresh-stars --snapshot-date YYYY-MM-DD
python scripts/build_reports.py
```

PDF 配图必须显式指定裁切坐标；`scripts/paper_sources.py assets --specs ...` 用于提取图表，`scripts/crop_paper_figures.py render/apply` 用于复核后批量替换已有图片。每张配图的固定原文版本、PDF 页码、裁切范围与图片哈希均保存在对应 `figures-manifest.json`。整页渲染只可用于本地阅读。

更新上游清单时加 `--refresh-source`；新题名须补充 scope-mapping，重建脚本会阻止未分类条目进入目录。现有家族分类优先读取 family-mapping 的 `by_entry_id`，调整归类时应同步该字段；新记录再按题名与仓库规则匹配。修改后先审阅 JSON 再重建报告，运行 `python scripts/verify_catalogue.py` 检查数据与内链。目录按当前输入生成；取消整个家族时应移走对应旧目录，避免遗留页面。

已完成文献筛选、项目索引与 **{len(articles)} 篇**介绍。每篇统一采用七部分结构；原文未披露的配置、表文冲突、评估范围与未独立复现的限制均在正文说明。未完成条目独立标识，不计入完成数。PDF/网页阅读缓存位于 `source/papers/`，由 `.gitignore` 排除；原图与出处清单保留在文章目录。

逐篇 `review.json` 记录作者来源核对，`evidence.json` 记录来源版本与资产；它们不代表外部同行评议。运行 `python scripts/verify_articles.py` 可检查七节结构、图片引用与文件哈希；该检查不自动证明数值和解释的语义准确性。更新文章后运行 `python scripts/build_reports.py` 重建阅读入口。
"""
    write("README.md", readme)
    print(json.dumps({k: v for k, v in stats.items() if k != "warnings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
