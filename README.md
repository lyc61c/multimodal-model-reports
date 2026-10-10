<div align="center">

# 多模态大模型技术报告

**Multimodal Model Reports**

按模型家族整理的中文论文详解 · 原文图表 · 架构、原理、数据与训练

[![Introductions](https://img.shields.io/badge/Introductions-271-2563eb?style=flat-square)](reports/paper-reading-guide.md)
[![Figures](https://img.shields.io/badge/Original_figures-1516-0f766e?style=flat-square)](reports/figure-crop-audit.md)
[![Snapshot](https://img.shields.io/badge/Snapshot-2026.10.08-64748b?style=flat-square)](reports/screening-all.md)

[开始阅读](reports/paper-reading-guide.md) · [模型家族](reports/model-family-index.md) · [整理状态](reports/article-index.md) · [来源与核对](#来源与核对)

</div>

---

本项目围绕多模态大模型的设计与训练，逐篇解读论文、技术报告与模型卡，并用原文中的架构图、流程图和实验表格辅助说明。同一官方模型系列集中整理，单版本模型独立成类，便于阅读与比较。

| 已完成中文详解 | 主目录模型家族 | 原文图表资产 |
|:---:|:---:|:---:|
| **271 篇** | **145 个** | **1,516 个** |

本轮保留 **272 条资料**；主目录包含 **199 篇模型文献**，完整阅读目录另收录保留的补充材料与范围待审资料。另有 **1 条**尚未完成，具体状态见[完整索引](reports/article-index.md)。

## 阅读导航

| 想做什么 | 从这里进入 |
|---|---|
| 找一篇论文，直接阅读中文详解 | [全部 271 篇逐篇论文详解 →](reports/paper-reading-guide.md) |
| 按模型系列阅读，比较不同版本 | [145 个模型家族完整索引 →](reports/model-family-index.md) |
| 查文献类型、收录情况与整理进度 | [全部 272 条保留资料与状态 →](reports/article-index.md) |

## 模型家族索引

以下为部分系列的快捷入口；[完整索引](reports/model-family-index.md)包含全部主目录家族及其机构、文献数量和筛选状态。

| 模型系列 | 模型系列 | 模型系列 | 模型系列 |
|---|---|---|---|
| [Qwen](papers/qwen/README.md) | [DeepSeek](papers/deepseek/README.md) | [InternVL](papers/internvl/README.md) | [MiniCPM-V / MiniCPM-o](papers/minicpm/README.md) |
| [LLaVA（官方研究谱系）](papers/llava/README.md) | [Kimi](papers/kimi/README.md) | [GLM / VisualGLM](papers/glm/README.md) | [Seed](papers/seed/README.md) |
| [Gemini](papers/gemini/README.md) | [GPT](papers/gpt/README.md) | [Claude](papers/claude/README.md) | [Llama](papers/llama/README.md) |
| [BLIP / xGen-MM](papers/blip/README.md) | [Emu](papers/emu/README.md) | [mPLUG](papers/mplug/README.md) | [VideoLLaMA](papers/videollama/README.md) |
| [VITA](papers/vita/README.md) | [VILA / NVILA](papers/vila/README.md) | [InternLM-XComposer](papers/internlm-xcomposer/README.md) | [MiMo](papers/mimo/README.md) |

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
| [Qwen2.5-VL](papers/qwen/qwen2.5-vl/README.md) | 动态分辨率、窗口注意力与绝对时间编码 |
| [DeepSeek-VL2](papers/deepseek/arxiv-2412.10302/README.md) | 动态切图与稀疏专家 |
| [InternVL3](papers/internvl/arxiv-2504.10479/README.md) | 联合预训练与混合偏好优化 |
| [BLIP-2](papers/blip/arxiv-2301.12597/README.md) | Q-Former 与冻结视觉、语言模型之间的连接 |

**[查看全部逐篇详解 →](reports/paper-reading-guide.md)**

配图只保留用到的原文图表区域，保留图例、表头与必要脚注；各篇记录图表来源与原文版本。

静态原图缺项：[MiMo-V2-Omni：统一感知与代理行动的全模态模型](papers/mimo/document-1ccf04ffaeaa/README.md)。对应发布资料仅提供动态或 HTML 图表，已在文中注明。

## 收录范围与筛选

来源为 [Awesome Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) 的 **Multimodal Instruction Tuning (& Latest Works)** 与 **Foundation Models** 两个章节，数据快照为 **2026-10-08**。

按 Code 栏对应仓库的 Star 筛选：**小于 100 的条目排除**；未提供 GitHub 的条目记为 **N/A**，另作候选或范围待审。Star 是仓库关注度指标，不能直接代表论文质量。

<details>
<summary><strong>展开筛选统计与归类口径</strong></summary>

<a id="首轮筛选结果"></a>

### 首轮筛选结果

| 项目 | 数量 |
|---|---:|
| Multimodal Instruction Tuning (& Latest Works) 原始条目 | 272 |
| Foundation Models 原始条目 | 31 |
| 原始总记录 / 去重后文献 | 303 / 301 |
| 原表 GitHub 仓库标识 / 重定向后实际仓库 | 197 / 192 |
| Star ≥ 100 的文献 | 199 |
| Star < 100 的排除文献 | 29 |
| 原表未提供 GitHub 的文献 | 73 |
| 主目录模型文献 / 家族 | 199 / 145 |

### 筛选口径

1. 只筛选上述两个章节，按 arXiv ID 或题名去重。GPT-5 发布资料和 Gemini 1 技术报告各跨章节出现一次。
2. 使用 Code 栏对应仓库的整数 Star；恰好 100 保留，数值以本次快照为准。
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
| 筛选台账 | [全部 301 篇](reports/screening-all.md) · [低于门槛 29 篇](reports/excluded-low-star.md) · [无 GitHub 指标 73 篇](reports/no-github.md) |
| 文献范围 | [补充与非模型文献 65 篇](reports/supporting-materials.md) · [范围待审 17 篇](reports/scope-review.md) |
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

文献清单来源于 [BradyFU/Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)。图表来自对应论文或官方发布资料，具体出处见各篇介绍。
