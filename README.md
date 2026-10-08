# 多模态大模型技术报告

[GitHub 仓库](https://github.com/lyc61c/multimodal-model-reports) · [全部介绍目录](reports/article-index.md)

按模型家族整理多模态大模型论文、技术报告与模型卡。Qwen 系列合并为一个家族，DeepSeek 系列合并为一个家族；单版本模型单独建立家族。机构与模型家族分开记录，同厂商的不同模型家族保持独立。

数据快照：**2026-10-08**。范围仅包含 [原项目](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) 的两个指定章节。

GitHub Star 反映仓库关注度，用作本轮可复核的门槛，不直接衡量论文质量。数值来自 Code 栏对应仓库；共享仓库的 Star 不属于单篇论文。

本轮全部 **272 条**保留资料，已完成 **271 篇**介绍。其余状态与阅读入口见[完整介绍索引](reports/article-index.md)。 已附 **1,092 个**原始图表资产，各文记录来源、版本和图表位置。 以下发布资料仅提供动态或 HTML 图表，静态原图缺项已在对应文档注明：[MiMo-V2-Omni：统一感知与代理行动的全模态模型](papers/mimo/document-1ccf04ffaeaa/README.md)。

## 首轮筛选结果

| 项目 | 数量 |
|---|---:|
| Multimodal Instruction Tuning (& Latest Works) 原始条目 | 272 |
| Foundation Models 原始条目 | 31 |
| 原始总记录 / 去重后文献 | 303 / 301 |
| 原表 GitHub 仓库标识（逐一核验） | 197 |
| 重定向后的实际 GitHub 仓库 | 192 |
| Star ≥ 100 的文献 | 199 |
| Star < 100 的排除文献 | 29 |
| 原表未提供 GitHub 的文献 | 73 |
| 主目录模型文献 / 家族 | 199 / 145 |

“Star 达标”统计包含全部文献类型，不等于技术报告数量。主目录收录首轮判为模型研究论文、技术报告与模型卡的候选；博客和发布页单列补充材料。目录保留首轮文献类型判定，逐篇介绍再根据原文说明实际定位；综述、评测、方法或发布页不会被写成作者新训练的模型。无 GitHub 指标的模型文献为候选，未宣称通过 Star 门槛。

## 筛选口径

1. 只筛选指定两个章节；按论文 arXiv ID 或题名去重。GPT-5 发布资料和 Gemini 1 技术报告各跨章节出现一次。
2. 使用 Code 栏对应 GitHub 仓库的精确整数 Star。小于 100 排除，恰好 100 保留。Star 随时间变化，应以本次快照为准。
3. 原表没有 GitHub 仓库的记录记为 N/A，保留候选或范围待审。没有把闭源报告当作 0 Star。
4. 数据集、评测、综述、工具、通用方法和发布博客不作为模型技术报告，独立列在附录。Star 达标不改变文献类型。
5. 家族依据发布主体与版本关系划分。第三方基于 LLaVA、BLIP 等训练的独立模型不自动并入原厂官方家族；仓库主体未核实的机构会明确标注。

## 模型家族索引

| 模型家族 | 项目主体 / 机构 | 模型文献 | Star 达标 | 无 Star 候选 |
|---|---|---:|---:|---:|
| [Qwen](papers/qwen/README.md) | 阿里云 Qwen 团队 | 6 | 6 | 0 |
| [DeepSeek](papers/deepseek/README.md) | DeepSeek（深度求索） | 4 | 4 | 0 |
| [InternVL](papers/internvl/README.md) | OpenGVLab 研究团队 | 6 | 6 | 0 |
| [MiniCPM-V / MiniCPM-o](papers/minicpm/README.md) | OpenBMB / ModelBest（面壁智能）/ 清华 THUNLP | 1 | 1 | 0 |
| [LLaVA-Med](papers/llava-med/README.md) | Microsoft Research | 1 | 1 | 0 |
| [LLaVA-MoD](papers/llava-mod/README.md) | repo:shufangxun | 1 | 1 | 0 |
| [LLaVAR](papers/llavar/README.md) | repo:SALT-NLP | 1 | 1 | 0 |
| [LLaVA（官方研究谱系）](papers/llava/README.md) | EvolvingLMMs Lab / LLaVA-OneVision 研究团队；LLaVA 官方研究团队 | 5 | 5 | 0 |
| [VITA](papers/vita/README.md) | VITA 联合研究团队 / 腾讯合作项目 | 4 | 4 | 0 |
| [Kimi](papers/kimi/README.md) | Moonshot AI（月之暗面） | 3 | 3 | 0 |
| [GLM / VisualGLM](papers/glm/README.md) | 智谱 / Z.ai / THUDM 的 GLM 团队 | 1 | 1 | 0 |
| [Gemini](papers/gemini/README.md) | Google / Google DeepMind | 3 | 0 | 3 |
| [GPT](papers/gpt/README.md) | OpenAI | 3 | 0 | 3 |
| [GPT4RoI](papers/gpt4roi/README.md) | repo:jshilong | 1 | 1 | 0 |
| [Claude](papers/claude/README.md) | Anthropic | 1 | 0 | 1 |
| [Llama](papers/llama/README.md) | Meta | 1 | 0 | 1 |
| [LLaMA-Adapter](papers/llama-adapter/README.md) | OpenGVLab 研究团队 | 1 | 1 | 0 |
| [LLaMA-VID](papers/llama-vid/README.md) | DVLab 研究团队 | 1 | 1 | 0 |
| [LlamaV-o1](papers/llamav-o1/README.md) | MBZUAI Oryx 研究团队 | 1 | 1 | 0 |
| [Emu](papers/emu/README.md) | 北京智源研究院（BAAI） | 3 | 3 | 0 |
| [BLIP / xGen-MM](papers/blip/README.md) | Salesforce AI Research | 3 | 3 | 0 |
| [BLIP3-o](papers/blip3-o/README.md) | repo:JiuhaiChen | 1 | 1 | 0 |
| [VideoLLaMA](papers/videollama/README.md) | 阿里巴巴 mPLUG / X-PLUG / DAMO 研究团队 | 3 | 3 | 0 |
| [mPLUG](papers/mplug/README.md) | 阿里巴巴 mPLUG / X-PLUG / DAMO 研究团队 | 5 | 5 | 0 |
| [Phi](papers/phi/README.md) | Microsoft Research | 1 | 0 | 1 |
| [MiMo](papers/mimo/README.md) | 小米 MiMo 团队 | 1 | 1 | 0 |
| [3D-LLM](papers/3d-llm/README.md) | repo:UMass-Foundation-Model | 1 | 1 | 0 |
| [All-Seeing](papers/all-seeing/README.md) | OpenGVLab 研究团队 | 2 | 2 | 0 |
| [ALLaVA](papers/allava/README.md) | repo:FreedomIntelligence | 1 | 1 | 0 |
| [An Empirical Study of Scaling Instruction-Tuned Large Multimodal Models](papers/an-empirical-study-of-scaling-instruction-tuned-large-m-805afa2bda72/README.md) | authors | 1 | 0 | 1 |
| [AnyGPT](papers/anygpt/README.md) | repo:OpenMOSS | 1 | 1 | 0 |
| [AnyMAL](papers/anymal/README.md) | Meta | 1 | 0 | 1 |
| [Apollo](papers/apollo/README.md) | Meta GenAI / Stanford 联合研究 | 1 | 0 | 1 |
| [AuroraCap](papers/auroracap/README.md) | repo:rese1f | 1 | 1 | 0 |
| [BAGEL](papers/bagel/README.md) | 字节跳动 Seed | 1 | 1 | 0 |
| [Baichuan](papers/baichuan/README.md) | 百川智能 | 1 | 1 | 0 |
| [BLIVA](papers/bliva/README.md) | repo:mlpc-ucsd | 1 | 1 | 0 |
| [BRAVE](papers/brave/README.md) | Google / EPFL 联合研究 | 1 | 0 | 1 |
| [BuboGPT](papers/bubogpt/README.md) | repo:magic-research | 1 | 1 | 0 |
| [Cambrian-1](papers/cambrian-1/README.md) | repo:cambrian-mllm | 1 | 1 | 0 |
| [Chameleon](papers/chameleon/README.md) | Meta | 1 | 0 | 1 |
| [ChartAssistant](papers/chartassistant/README.md) | OpenGVLab 研究团队 | 1 | 1 | 0 |
| [ChartLlama](papers/chartllama/README.md) | repo:tingxueronghua | 1 | 1 | 0 |
| [ChartMoE](papers/chartmoe/README.md) | repo:IDEA-FinAI | 1 | 1 | 0 |
| [Chat-UniVi](papers/chat-univi/README.md) | PKU-YuanGroup 研究团队 | 1 | 1 | 0 |
| [ChatRex](papers/chatrex/README.md) | repo:IDEA-Research | 1 | 1 | 0 |
| [ChatSpot](papers/chatspot/README.md) | authors | 1 | 0 | 1 |
| [Cheetah](papers/cheetah/README.md) | repo:DCDmllm | 1 | 1 | 0 |
| [CM3Leon](papers/cm3leon/README.md) | Meta | 1 | 0 | 1 |
| [CogCoM](papers/cogcom/README.md) | 清华 THUDM 研究团队 | 1 | 1 | 0 |
| [CogVLM / CogAgent](papers/cogvlm/README.md) | 智谱 / Z.ai / THUDM 的 GLM 团队 | 2 | 2 | 0 |
| [ConvLLaVA](papers/convllava/README.md) | repo:alibaba | 1 | 1 | 0 |
| [CoVLM](papers/covlm/README.md) | authors | 1 | 0 | 1 |
| [CuMo](papers/cumo/README.md) | repo:SHI-Labs | 1 | 1 | 0 |
| [DetGPT](papers/detgpt/README.md) | repo:OptimalScale | 1 | 1 | 0 |
| [DreamLLM](papers/dreamllm/README.md) | repo:RunpeiDong | 1 | 1 | 0 |
| [Eagle](papers/eagle/README.md) | NVIDIA | 2 | 2 | 0 |
| [Ego-R1](papers/ego-r1/README.md) | repo:egolife-ai | 1 | 1 | 0 |
| [EVE](papers/eve/README.md) | 北京智源研究院（BAAI） | 1 | 1 | 0 |
| [EVLM](papers/evlm/README.md) | authors | 1 | 0 | 1 |
| [Ferret](papers/ferret/README.md) | Apple | 2 | 1 | 1 |
| [GLaMM](papers/glamm/README.md) | MBZUAI Oryx 研究团队 | 1 | 1 | 0 |
| [GROUNDHOG](papers/groundhog/README.md) | authors | 1 | 0 | 1 |
| [Honeybee](papers/honeybee/README.md) | Kakao Brain | 1 | 1 | 0 |
| [ImageBind-LLM](papers/imagebind-llm/README.md) | OpenGVLab 研究团队 | 1 | 1 | 0 |
| [InternLM-XComposer](papers/internlm-xcomposer/README.md) | InternLM-XComposer 研究团队 | 5 | 5 | 0 |
| [Kosmos](papers/kosmos/README.md) | Microsoft Research | 2 | 2 | 0 |
| [LaVIT](papers/lavit/README.md) | repo:jy0205 | 1 | 1 | 0 |
| [Libra](papers/libra/README.md) | repo:YifanXu74 | 1 | 1 | 0 |
| [LION (JiuTian)](papers/lion/README.md) | repo:rshaojimmy | 1 | 1 | 0 |
| [LISA](papers/lisa/README.md) | DVLab 研究团队 | 1 | 1 | 0 |
| [LITA](papers/lita/README.md) | NVIDIA | 1 | 1 | 0 |
| [LL3DA](papers/ll3da/README.md) | repo:Open3DA | 1 | 1 | 0 |
| [LLMGA](papers/llmga/README.md) | DVLab 研究团队 | 1 | 1 | 0 |
| [LongLLaVA](papers/longllava/README.md) | repo:FreedomIntelligence | 1 | 1 | 0 |
| [LongVA](papers/longva/README.md) | EvolvingLMMs Lab / LLaVA-OneVision 研究团队 | 1 | 1 | 0 |
| [LongVU](papers/longvu/README.md) | Vision-CAIR 研究团队 | 1 | 1 | 0 |
| [LTU](papers/ltu/README.md) | repo:YuanGongND | 1 | 1 | 0 |
| [Lynx](papers/lynx/README.md) | 字节跳动研究团队 | 1 | 1 | 0 |
| [MA-LMM](papers/ma-lmm/README.md) | repo:boheumd | 1 | 1 | 0 |
| [Macaw-LLM](papers/macaw-llm/README.md) | repo:lyuchenyang | 1 | 1 | 0 |
| [MetaLM](papers/metalm/README.md) | Microsoft Research | 1 | 1 | 0 |
| [Meteor](papers/meteor/README.md) | repo:ByungKwanLee | 1 | 1 | 0 |
| [Mini-Gemini](papers/mini-gemini/README.md) | DVLab 研究团队 | 1 | 1 | 0 |
| [MiniGPT](papers/minigpt/README.md) | Vision-CAIR 研究团队 | 2 | 2 | 0 |
| [MM1](papers/mm1/README.md) | Apple | 1 | 0 | 1 |
| [MMaDA](papers/mmada/README.md) | repo:Gen-Verse | 1 | 1 | 0 |
| [MMSearch-R1](papers/mmsearch-r1/README.md) | EvolvingLMMs Lab / LLaVA-OneVision 研究团队 | 1 | 1 | 0 |
| [MoAI](papers/moai/README.md) | repo:ByungKwanLee | 1 | 1 | 0 |
| [MobileVLM](papers/mobilevlm/README.md) | 美团 AutoML 团队 | 2 | 2 | 0 |
| [MoE-LLaVA](papers/moe-llava/README.md) | PKU-YuanGroup 研究团队 | 1 | 1 | 0 |
| [Molmo](papers/molmo/README.md) | Allen Institute for AI（Ai2） | 1 | 0 | 1 |
| [Monkey / TextMonkey](papers/monkey/README.md) | repo:Yuliang-Liu | 2 | 2 | 0 |
| [MovieChat](papers/moviechat/README.md) | repo:rese1f | 1 | 1 | 0 |
| [MQT-LLaVA](papers/mqt-llava/README.md) | repo:gordonhu608 | 1 | 1 | 0 |
| [MultiModal-GPT](papers/multimodal-gpt/README.md) | repo:open-mmlab | 1 | 1 | 0 |
| [NExT-Chat](papers/next-chat/README.md) | repo:NExT-ChatV | 1 | 1 | 0 |
| [NExT-GPT](papers/next-gpt/README.md) | repo:NExT-GPT | 1 | 1 | 0 |
| [NExT-OMNI](papers/next-omni/README.md) | authors | 1 | 0 | 1 |
| [Nexus-O](papers/nexus-o/README.md) | authors | 1 | 0 | 1 |
| [OMG-LLaVA](papers/omg-llava/README.md) | repo:lxtGH | 1 | 1 | 0 |
| [Omni-Diffusion](papers/omni-diffusion/README.md) | VITA 联合研究团队 / 腾讯合作项目 | 1 | 1 | 0 |
| [OmniVinci](papers/omnivinci/README.md) | NVIDIA | 1 | 1 | 0 |
| [OneLLM](papers/onellm/README.md) | repo:csuhan | 1 | 1 | 0 |
| [OpenUni](papers/openuni/README.md) | repo:wusize | 1 | 1 | 0 |
| [Osprey](papers/osprey/README.md) | repo:CircleRadon | 1 | 1 | 0 |
| [Otter](papers/otter/README.md) | repo:Luodian | 1 | 1 | 0 |
| [Ovis](papers/ovis/README.md) | 阿里巴巴 Ovis 团队 | 1 | 1 | 0 |
| [PaLI](papers/pali/README.md) | Google / Google DeepMind | 1 | 0 | 1 |
| [PaLM-E](papers/palm-e/README.md) | Google / Google DeepMind | 1 | 0 | 1 |
| [PandaGPT](papers/pandagpt/README.md) | repo:yxuansu | 1 | 1 | 0 |
| [Pengi](papers/pengi/README.md) | Microsoft Research | 1 | 1 | 0 |
| [Pixel Aligned Language Models](papers/pixel-aligned-language-models/README.md) | authors | 1 | 0 | 1 |
| [PointLLM](papers/pointllm/README.md) | repo:OpenRobotLab | 1 | 1 | 0 |
| [Prismatic](papers/prismatic/README.md) | Toyota Research Institute / Prismatic 研究团队 | 1 | 1 | 0 |
| [Prismer](papers/prismer/README.md) | NVIDIA | 1 | 1 | 0 |
| [SALMONN](papers/salmonn/README.md) | 字节跳动研究团队 | 2 | 2 | 0 |
| [See, Say, and Segment: Teaching LMMs to Overcome False Premises](papers/see-say-and-segment-teaching-lmms-to-overcome-false-premises/README.md) | authors | 1 | 0 | 1 |
| [Seed](papers/seed/README.md) | 字节跳动 Seed | 3 | 0 | 3 |
| [ShareGPT4V](papers/sharegpt4v/README.md) | InternLM-XComposer 研究团队 | 1 | 1 | 0 |
| [Shikra](papers/shikra/README.md) | repo:shikras | 1 | 1 | 0 |
| [Show-o](papers/show-o/README.md) | repo:showlab | 1 | 1 | 0 |
| [Skywork](papers/skywork/README.md) | Skywork / 昆仑万维研究团队 | 1 | 1 | 0 |
| [SliME](papers/slime/README.md) | repo:yfzhang114 | 1 | 1 | 0 |
| [SpatialVLM](papers/spatialvlm/README.md) | Google / Google DeepMind | 1 | 0 | 1 |
| [SPHINX](papers/sphinx/README.md) | repo:Alpha-VLLM | 1 | 1 | 0 |
| [StreamChat](papers/streamchat/README.md) | authors | 1 | 0 | 1 |
| [Thyme](papers/thyme/README.md) | repo:yfzhang114 | 1 | 1 | 0 |
| [TimeChat](papers/timechat/README.md) | repo:RenShuhuai-Andy | 1 | 1 | 0 |
| [TimeMarker](papers/timemarker/README.md) | repo:TimeMarker-LLM | 1 | 1 | 0 |
| [UniGen](papers/unigen/README.md) | authors | 1 | 0 | 1 |
| [UnIVAL](papers/unival/README.md) | repo:mshukor | 1 | 1 | 0 |
| [VaQuitA](papers/vaquita/README.md) | authors | 1 | 0 | 1 |
| [Vary](papers/vary/README.md) | repo:Ucas-HaoranWei | 1 | 1 | 0 |
| [Video-ChatGPT](papers/video-chatgpt/README.md) | MBZUAI Oryx 研究团队 | 1 | 1 | 0 |
| [Video-LLaVA](papers/video-llava/README.md) | PKU-YuanGroup 研究团队 | 1 | 1 | 0 |
| [VideoChat](papers/videochat/README.md) | MCG-NJU / VideoChat3 研究团队；OpenGVLab 研究团队 | 2 | 2 | 0 |
| [VideoLLM-online](papers/videollm-online/README.md) | repo:showlab | 1 | 1 | 0 |
| [VILA / NVILA](papers/vila/README.md) | NVIDIA | 3 | 2 | 1 |
| [VIP-LLaVA](papers/vip-llava/README.md) | repo:mu-cai | 1 | 1 | 0 |
| [VisCPM](papers/viscpm/README.md) | OpenBMB / ModelBest（面壁智能）/ 清华 THUNLP | 1 | 1 | 0 |
| [VisionLLM](papers/visionllm/README.md) | OpenGVLab 研究团队 | 1 | 1 | 0 |
| [VITRON](papers/vitron/README.md) | Skywork / 昆仑万维研究团队 | 1 | 1 | 0 |
| [VTimeLLM](papers/vtimellm/README.md) | repo:huangb23 | 1 | 1 | 0 |
| [X-LLM](papers/x-llm/README.md) | repo:phellonchen | 1 | 1 | 0 |

## 论文图文介绍

当前已完成 **271 篇**逐篇介绍，按已审核的七部分模板组织，并引用原文图表。

本轮全部 **272 条**保留资料，已完成 **271 篇**介绍。其余状态与阅读入口见[完整介绍索引](reports/article-index.md)。 已附 **1,092 个**原始图表资产，各文记录来源、版本和图表位置。 以下发布资料仅提供动态或 HTML 图表，静态原图缺项已在对应文档注明：[MiMo-V2-Omni：统一感知与代理行动的全模态模型](papers/mimo/document-1ccf04ffaeaa/README.md)。

| 模型家族 | 逐篇介绍 | 原文版本 |
|---|---|---|
| 3D-LLM | [3D-LLM：把多视角视觉知识映射到三维场景，再交给语言模型](papers/3d-llm/arxiv-2307.12981/README.md) | 2307.12981v1 |
| All-Seeing | [All-Seeing：用区域数据引擎统一视觉识别与语言生成](papers/all-seeing/arxiv-2308.01907/README.md) | 2308.01907v1 |
| All-Seeing | [ASMv2：把对象、关系与位置统一为关系对话](papers/all-seeing/arxiv-2402.19474/README.md) | 2402.19474v1 |
| ALLaVA | [ALLaVA：用细粒度描述与复杂问答补偿小模型容量](papers/allava/arxiv-2402.11684/README.md) | 2402.11684v1 |
| An Empirical Study of Scaling Instruction-Tuned Large Multimodal Models | [大规模视觉指令微调：LLaVA 的 33B、65B 与参数高效训练研究](papers/an-empirical-study-of-scaling-instruction-tuned-large-m-805afa2bda72/arxiv-2309.09958/README.md) | 2309.09958v1 |
| AnyGPT | [AnyGPT：用离散序列统一图像、语音、音乐与文字](papers/anygpt/arxiv-2402.12226/README.md) | 2402.12226v1 |
| AnyMAL | [AnyMAL：让图像、音频、视频与运动传感器共享语言模型入口](papers/anymal/arxiv-2309.16058/README.md) | 2309.16058v1 |
| Apollo | [Apollo：用视频设计实验建立高效长视频理解模型](papers/apollo/arxiv-2412.10360/README.md) | 2412.10360v1 |
| AuroraCap | [AuroraCap：以 Token 合并提高视频详细描述效率](papers/auroracap/arxiv-2410.03051/README.md) | 2410.03051v1 |
| BAGEL | [BAGEL：以共享注意力连接多模态理解与连续视觉生成](papers/bagel/arxiv-2505.14683/README.md) | 2505.14683v1 |
| Baichuan | [Baichuan-Omni-1.5：语义声学双目标与全模态交互](papers/baichuan/document-165201d8355a/README.md) | official-source-snapshot-2026-10-08 |
| Beyond Language Modeling | [Beyond Language Modeling：从零统一预训练中的视觉表示、MoE与缩放不对称](papers/beyond-language-modeling/arxiv-2603.03276/README.md) | 2603.03276v1 |
| BLIP / xGen-MM | [BLIP-2：以 Q-Former 连接冻结视觉编码器和语言模型](papers/blip/arxiv-2301.12597/README.md) | 2301.12597v1 |
| BLIP / xGen-MM | [InstructBLIP：让视觉查询先理解指令，再读取图像](papers/blip/arxiv-2305.06500/README.md) | 2305.06500v1 |
| BLIP / xGen-MM | [xGen-MM（BLIP-3）：交错图文预训练与逐块视觉采样](papers/blip/arxiv-2408.08872/README.md) | 2408.08872v1 |
| BLIP3-o | [BLIP3-o：以连续 CLIP 特征与 Flow Matching 统一图像理解和生成](papers/blip3-o/arxiv-2505.09568/README.md) | 2505.09568v1 |
| BLIVA | [BLIVA：用查询特征与完整 patch 特征共同回答含文字图像的问题](papers/bliva/arxiv-2308.09936/README.md) | 2308.09936v1 |
| BRAVE | [BRAVE：以多编码器查询Transformer拓宽视觉表示](papers/brave/arxiv-2404.07204/README.md) | 2404.07204v1 |
| BuboGPT | [BuboGPT：以语言为桥梁，将图像、声音与视觉区域联系起来](papers/bubogpt/arxiv-2307.08581/README.md) | 2307.08581v1 |
| Cambrian-1 | [Cambrian-1：用空间视觉聚合器连接多种视觉表征与语言模型](papers/cambrian-1/arxiv-2406.16860/README.md) | 2406.16860v1 |
| Chameleon | [Chameleon：将图像和文字统一为离散token的早期融合模型](papers/chameleon/arxiv-2405.09818/README.md) | 2405.09818v1 |
| ChartAssistant | [ChartAssistant：先学图表到表格，再统一图表问答与推理](papers/chartassistant/arxiv-2401.02384/README.md) | 2401.02384v1 |
| ChartLlama | [ChartLlama：从表格与绘图代码合成监督，学习图表理解和编辑](papers/chartllama/arxiv-2311.16483/README.md) | 2311.16483v1 |
| ChartMoE | [ChartMoE：用表格、JSON与代码对齐初始化图表专家连接器](papers/chartmoe/arxiv-2409.03277/README.md) | 2409.03277v1 |
| Chat-UniVi | [Chat-UniVi：用多尺度动态视觉 Token 统一图像与视频理解](papers/chat-univi/arxiv-2311.08046/README.md) | 2311.08046v1 |
| ChatRex | [ChatRex：先产生候选框，再让语言模型选择对象并解释场景](papers/chatrex/arxiv-2411.18363/README.md) | 2411.18363v1 |
| ChatSpot | [ChatSpot：把点击和框选转换成精确区域指令](papers/chatspot/arxiv-2307.09474/README.md) | 2307.09474v1 |
| Cheetah | [Cheetah：让语言模型在中间层控制视觉信息重新提取](papers/cheetah/arxiv-2308.04152/README.md) | 2308.04152v1 |
| Claude | [Claude 3：视觉输入、行为对齐与长上下文能力的模型卡解读](papers/claude/document-d746d75f8685/README.md) | official-source-snapshot-2026-10-08 |
| CM3Leon | [CM3Leon：检索增强的自回归模型如何统一图像生成与视觉问答](papers/cm3leon/arxiv-2309.02591/README.md) | 2309.02591v1 |
| CogCoM | [CogCoM：通过定位、裁剪与放大建立可追踪的视觉操作链](papers/cogcom/arxiv-2402.04236/README.md) | 2402.04236v1 |
| CogVLM / CogAgent | [CogAgent：通过高分辨率视觉理解操作图形界面](papers/cogvlm/arxiv-2312.08914/README.md) | 2312.08914v1 |
| CogVLM / CogAgent | [CogVLM：在语言模型每一层加入视觉专家，实现深层图文融合](papers/cogvlm/arxiv-2311.03079/README.md) | 2311.03079v1 |
| Comparison Visual Instruction Tuning | [Comparison Visual Instruction Tuning：同时学习图像的共同点与差异](papers/comparison-visual-instruction-tuning/arxiv-2406.09240/README.md) | 2406.09240v1 |
| CompCap | [CompCap：用可控合成图像与复合描述增强多模态理解](papers/compcap/arxiv-2412.05243/README.md) | 2412.05243v1 |
| ConvLLaVA | [ConvLLaVA：用分层卷积压缩高分辨率视觉输入](papers/convllava/arxiv-2405.15738/README.md) | 2405.15738v1 |
| Cosmos | [Cosmos 3：以双路径 Transformer 统一理解、世界生成与动作预测](papers/cosmos/arxiv-2606.02800/README.md) | 2606.02800v1 |
| CoVLM | [CoVLM：通过通信解码组合视觉实体与关系](papers/covlm/arxiv-2311.03354/README.md) | 2311.03354v1 |
| CuMo | [CuMo：从稠密检查点扩展视觉编码器和连接器专家](papers/cumo/arxiv-2405.05949/README.md) | 2405.05949v1 |
| DeepSeek | [DeepSeek Harness 来源核验：Cordis 的时空可组合编程范式](papers/deepseek/document-ddf539694eca/README.md) | 2608.25512v1 |
| DeepSeek | [DeepSeek-OCR 2：以 Visual Causal Flow 改善文档阅读顺序](papers/deepseek/document-536ef2604f4a/README.md) | official-source-snapshot-2026-10-08 |
| DeepSeek | [DeepSeek-OCR：用少量视觉token读取文档与探索光学上下文压缩](papers/deepseek/arxiv-2510.18234/README.md) | 2510.18234v1 |
| DeepSeek | [DeepSeek-V4-Flash-Vision-Exp：视觉智能体发布说明](papers/deepseek/document-6648a6adbc11/README.md) | official-source-snapshot-2026-10-08 |
| DeepSeek | [DeepSeek-V4：压缩混合注意力与百万 token 推理](papers/deepseek/document-08e9e69bb6bb/README.md) | 2606.19348v1 |
| DeepSeek | [DeepSeek-VL2：结合动态切图与稀疏专家的视觉语言模型](papers/deepseek/arxiv-2412.10302/README.md) | 2412.10302v1 |
| DeepSeek | [DeepSeek-VL：双视觉编码器与保留语言能力的联合预训练](papers/deepseek/arxiv-2403.05525/README.md) | 2403.05525v1 |
| DenseWorld-1M | [DenseWorld-1M：从像素掩码到对象细节与场景关系的密集描述](papers/denseworld-1m/arxiv-2506.24102/README.md) | 2506.24102v1 |
| DetGPT | [DetGPT：从用户目标推理待检测对象，再交给开放词表检测器](papers/detgpt/arxiv-2305.14167/README.md) | 2305.14167v1 |
| dots | [dots.vlm1：从原生分辨率视觉预训练到DeepSeek-V3多模态适配](papers/dots/document-bdf40c9e7783/README.md) | official-source-snapshot-2026-10-08 |
| DreamLLM | [DreamLLM：联合学习图像理解、图像生成与交错图文创作](papers/dreamllm/arxiv-2309.11499/README.md) | 2309.11499v1 |
| Eagle | [Eagle 2.5：以信息保留和渐进训练增强长上下文视觉理解](papers/eagle/arxiv-2504.15271/README.md) | 2504.15271v1 |
| Eagle | [EAGLE：多视觉编码器如何组合，才能真正改善多模态理解](papers/eagle/arxiv-2408.15998/README.md) | 2408.15998v1 |
| Ego-R1 | [Ego-R1：通过工具链推理检索跨日第一视角视频](papers/ego-r1/arxiv-2506.13654/README.md) | 2506.13654v1 |
| Emu | [Emu3.5：用统一离散序列学习图文世界，并适配并行图像生成](papers/emu/arxiv-2510.26583/README.md) | 2510.26583v1 |
| Emu | [Emu3：以离散 token 的自回归预测统一文字、图像与视频](papers/emu/arxiv-2409.18869/README.md) | 2409.18869v1 |
| Emu | [Emu：交错多模态序列的生成预训练](papers/emu/arxiv-2307.05222/README.md) | 2307.05222v1 |
| Enhancing Multimodal Large Language Models with Vision Detection Models: An Empirical Study | [检测与 OCR 增强多模态模型：三种融合策略的实证比较](papers/enhancing-multimodal-large-language-models-with-vision-e94f5590857f/arxiv-2401.17981/README.md) | 2401.17981v1 |
| EVE | [EVE：用语言底座预对齐和视觉特征监督构建无深层视觉编码器模型](papers/eve/arxiv-2406.11832/README.md) | 2406.11832v1 |
| EVLM | [EVLM：用分层视觉特征和门控交叉注意力降低视觉理解成本](papers/evlm/arxiv-2407.14177/README.md) | 2407.14177v1 |
| Ferret | [Ferret-UI：面向移动界面的区域理解与定位](papers/ferret/arxiv-2404.05719/README.md) | 2404.05719v1 |
| Ferret | [Ferret：任意形状区域的指代理解与文本定位](papers/ferret/arxiv-2310.07704/README.md) | 2310.07704v1 |
| Fuyu | [Fuyu-8B：直接读取图像块的数字代理基础模型](papers/fuyu/document-87ff293e66f7/README.md) | official-model-card-2026-10-08 |
| Gemini | [Gemini 1.5：百万级混合上下文、稀疏专家模型与高效蒸馏](papers/gemini/document-1855460a9712/README.md) | official-source-snapshot-2026-10-08 |
| Gemini | [Gemini 2.5：原生多模态、思考预算与长上下文](papers/gemini/document-8c1dda86c921/README.md) | official-source-snapshot-2026-10-08 |
| Gemini | [Gemini 3.1 Pro：复杂推理与应用构建能力的官方发布说明](papers/gemini/document-7cca3ca5d4d3/README.md) | official-source-snapshot-2026-10-08 |
| Gemini | [Gemini 4 Argon：面向长程专业任务的推理模型](papers/gemini/document-a07676c13763/README.md) | official-source-snapshot-2026-10-08 |
| Gemini | [Gemini 初代：联合多模态预训练与分层部署的模型家族](papers/gemini/document-044099a491e0/README.md) | 2312.11805v1 |
| Gemini | [Gemini3.8Flash与FlashCyber：长程agent能力及安全专项评测](papers/gemini/document-390f20b2fe8b/README.md) | official-source-snapshot-2026-10-08 |
| Gemma | [Gemma 4：端侧多模态、统一解码器与稀疏专家模型](papers/gemma/document-92d16b8c59c9/README.md) | official-source-snapshot-2026-10-08 |
| GLaMM | [GLaMM：让回答中的短语对应图像里的像素掩码](papers/glamm/arxiv-2311.03356/README.md) | 2311.03356v1 |
| GLM / VisualGLM | [GLM-4.1V-Thinking：原生分辨率视觉基础与跨领域课程强化学习](papers/glm/arxiv-2507.01006/README.md) | 2507.01006v1 |
| GLM / VisualGLM | [GLM-5.3-Flash：稀疏与线性注意力结合的原生多模态模型](papers/glm/document-e213666dcb0d/README.md) | official-model-card-2026-10-08 |
| GLM / VisualGLM | [VisualGLM-6B：以Q-Former连接视觉表征与中英双语ChatGLM](papers/glm/document-4d81507404ce/README.md) | official-source-snapshot-2026-10-08 |
| GPT | [GPT-4.1：面向代码、指令遵循与长上下文的多模态模型家族](papers/gpt/document-32a268f7f1d8/README.md) | official-release-2026-10-08 |
| GPT | [GPT-4o 原生图像生成附录：生成能力与安全评测](papers/gpt/document-0b8790d1de58/README.md) | official-source-snapshot-2026-10-08 |
| GPT | [GPT-4o：端到端多模态交互与语音系统评估](papers/gpt/document-9ee9863086d5/README.md) | official-system-card-2024-08-08; release-2024-05-13 |
| GPT | [GPT-4V 系统卡：图像能力与部署安全评估](papers/gpt/document-5cd14e1ea9af/README.md) | official-source-snapshot-2026-10-08 |
| GPT | [GPT-4：图文输入、可预测扩展与反馈对齐](papers/gpt/arxiv-2303.08774/README.md) | 2303.08774v1 |
| GPT | [GPT-5.2：视觉理解、长上下文与工具工作流的升级](papers/gpt/document-e2a1f413a7a5/README.md) | official-release-2026-10-08 |
| GPT | [GPT-5：模型路由、强化学习推理与输出导向训练](papers/gpt/document-d0807662cb93/README.md) | official-system-card-2025-08-13; release-2025-08-07 |
| GPT | [GPT-6 Astra：面向视觉交互与长期任务的通用模型](papers/gpt/document-0d9109ba3678/README.md) | official-source-snapshot-2026-10-08 |
| GPT4RoI | [GPT4RoI：用框选区域的视觉特征执行空间指令](papers/gpt4roi/arxiv-2307.03601/README.md) | 2307.03601v1 |
| GPT4Tools | [GPT4Tools：通过自指令数据教开源语言模型调用视觉工具](papers/gpt4tools/arxiv-2305.18752/README.md) | 2305.18752v1 |
| Graphist | [Graphist：从无序素材生成分层平面设计](papers/graphist/arxiv-2404.14368/README.md) | 2404.14368v1 |
| GROUNDHOG | [GROUNDHOG：从候选实体掩码中检索并合并语言所指区域](papers/groundhog/arxiv-2402.16846/README.md) | 2402.16846v1 |
| Honeybee | [Honeybee：局部视觉投影器与指令数据配方](papers/honeybee/arxiv-2312.06742/README.md) | 2312.06742v1 |
| ImageBind-LLM | [ImageBind-LLM：借用共享嵌入空间的多模态指令模型](papers/imagebind-llm/arxiv-2309.03905/README.md) | 2309.03905v1 |
| InternLM-XComposer | [InternLM-XComposer-2.5：长上下文多图视频理解与内容创作](papers/internlm-xcomposer/arxiv-2407.03320/README.md) | 2407.03320v1 |
| InternLM-XComposer | [InternLM-XComposer2-4KHD：动态分块与二维结构保留的高分辨率视觉语言模型](papers/internlm-xcomposer/arxiv-2404.06512/README.md) | 2404.06512v1 |
| InternLM-XComposer | [InternLM-XComposer2.5-OmniLive：把流式感知、长记忆与推理解耦](papers/internlm-xcomposer/arxiv-2412.09596/README.md) | 2412.09596v1 |
| InternLM-XComposer | [InternLM-XComposer2：用 Partial LoRA 连接视觉理解与自由图文创作](papers/internlm-xcomposer/arxiv-2401.16420/README.md) | 2401.16420v1 |
| InternLM-XComposer | [InternLM-XComposer：双语视觉理解与检索式图文文章编排](papers/internlm-xcomposer/arxiv-2309.15112/README.md) | 2309.15112v1 |
| InternVL | [InternVL 1.5：大视觉骨干、动态高分辨率与双语数据](papers/internvl/arxiv-2404.16821/README.md) | 2404.16821v1 |
| InternVL | [InternVL 2.5：复用视觉主干、清理训练数据与扩展测试时推理](papers/internvl/arxiv-2412.05271/README.md) | 2412.05271v1 |
| InternVL | [InternVL 3.5：级联强化学习、自适应视觉压缩与分离部署](papers/internvl/arxiv-2508.18265/README.md) | 2508.18265v1 |
| InternVL | [InternVL-U：共享语义上下文与双流图像生成的统一模型](papers/internvl/arxiv-2603.09877/README.md) | 2603.09877v1 |
| InternVL | [InternVL3：联合预训练、可变视觉位置与混合偏好优化](papers/internvl/arxiv-2504.10479/README.md) | 2504.10479v1 |
| InternVL | [InternVL：扩展视觉骨干并逐步对齐语言模型](papers/internvl/arxiv-2312.14238/README.md) | 2312.14238v1 |
| Kimi | [Kimi K2.5：视觉文本联合训练与并行智能体编排](papers/kimi/arxiv-2602.02276/README.md) | 2602.02276v1 |
| Kimi | [Kimi K3：混合递归注意力、原生视觉与百万上下文 agent 训练](papers/kimi/document-eb106a710a29/README.md) | official-source-snapshot-2026-10-08 |
| Kimi | [Kimi-VL：原生分辨率、长上下文与稀疏多模态推理](papers/kimi/arxiv-2504.07491/README.md) | 2504.07491v1 |
| Kosmos | [Kosmos-1：把视觉感知接入通用语言接口](papers/kosmos/arxiv-2302.14045/README.md) | 2302.14045v1 |
| Kosmos | [KOSMOS-2：把文字指代表达连接到图像区域](papers/kosmos/arxiv-2306.14824/README.md) | 2306.14824v1 |
| LAMM | [LAMM：把图像、点云与传统视觉任务接入指令语言模型](papers/lamm/arxiv-2306.06687/README.md) | 2306.06687v1 |
| LanguageBind | [LanguageBind：以固定语言空间连接视频、音频、深度与红外](papers/languagebind/arxiv-2310.01852/README.md) | 2310.01852v1 |
| LaVIN | [LaVIN：按输入模态路由的低成本视觉语言适配](papers/lavin/arxiv-2305.15023/README.md) | 2305.15023v1 |
| LaVIT | [LaVIT：动态视觉 token 与统一图文下一 token 预测](papers/lavit/arxiv-2309.04669/README.md) | 2309.04669v1 |
| LEO | [LEO：以对象为中心连接三维理解、语言与具身动作](papers/leo/arxiv-2311.12871/README.md) | 2311.12871v1 |
| Libra | [Libra：在语言模型上构建解耦的视觉系统](papers/libra/arxiv-2405.10140/README.md) | 2405.10140v1 |
| LION (JiuTian) | [LION：用区域训练与图像标签补足双层视觉知识](papers/lion/arxiv-2311.11860/README.md) | 2311.11860v1 |
| LISA | [LISA：用分割token把语言推理连接到像素掩码](papers/lisa/arxiv-2308.00692/README.md) | 2308.00692v1 |
| LITA | [LITA：能回答何时发生的视频语言助手](papers/lita/arxiv-2403.19046/README.md) | 2403.19046v1 |
| LL3DA | [LL3DA：接受点击与框提示的三维语言助手](papers/ll3da/arxiv-2311.18651/README.md) | 2311.18651v1 |
| Llama | [Llama 3.2：冻结语言骨干的视觉扩展与轻量模型](papers/llama/document-0c6cca94a384/README.md) | official-source-snapshot-2026-10-08 |
| Llama | [Llama 3：大规模稠密语言模型与组合式视觉、语音扩展](papers/llama/arxiv-2407.21783/README.md) | 2407.21783v1 |
| Llama | [Llama 4：原生多模态 MoE、长上下文与在线强化学习](papers/llama/document-4003084812b3/README.md) | official-source-snapshot-2026-10-08 |
| LLaMA-Adapter | [LLaMA-Adapter V2：分离视觉对齐与语言指令的参数高效调优](papers/llama-adapter/arxiv-2304.15010/README.md) | 2304.15010v1 |
| LLaMA-Adapter | [LLaMA-Adapter：用零初始化注意力高效微调语言模型](papers/llama-adapter/arxiv-2303.16199/README.md) | 2303.16199v1 |
| LLaMA-VID | [LLaMA-VID：以问题相关 token 和内容 token 压缩视频帧](papers/llama-vid/arxiv-2311.17043/README.md) | 2311.17043v1 |
| LlamaV-o1 | [LlamaV-o1：以课程学习和搜索改进逐步视觉推理](papers/llamav-o1/arxiv-2501.06186/README.md) | 2501.06186v1 |
| LLaVA（官方研究谱系） | [LLaVA 初代：从文字教师生成视觉指令，到端到端语言适配](papers/llava/arxiv-2304.08485/README.md) | 2304.08485v1 |
| LLaVA（官方研究谱系） | [LLaVA-1.5：用简单连接器与可复核数据建立强基线](papers/llava/arxiv-2310.03744/README.md) | 2310.03744v1 |
| LLaVA（官方研究谱系） | [LLaVA-NeXT：用动态高分辨率与数据配方改善OCR和视觉推理](papers/llava/document-86e88fe4b1ec/README.md) | official-source-snapshot-2026-10-08 |
| LLaVA（官方研究谱系） | [LLaVA-OneVision-1.5：概念均衡数据与高效中期训练](papers/llava/arxiv-2509.23661/README.md) | 2509.23661v1 |
| LLaVA（官方研究谱系） | [LLaVA-OneVision：用统一视觉表示连接单图、多图与视频任务](papers/llava/arxiv-2408.03326/README.md) | 2408.03326v1 |
| LLaVA（官方研究谱系） | [LLaVA-Plus：学习调用视觉工具的多模态智能体](papers/llava/arxiv-2311.05437/README.md) | 2311.05437v1 |
| LLaVA-Med | [LLaVA-Med：用医学图文与合成指令适配视觉语言助手](papers/llava-med/arxiv-2306.00890/README.md) | 2306.00890v1 |
| LLaVA-MoD | [LLaVA-MoD：通过稀疏专家与渐进蒸馏训练小型视觉语言模型](papers/llava-mod/arxiv-2408.15881/README.md) | 2408.15881v1 |
| LLaVAR | [LLaVAR：用 OCR 数据与合成对话增强文字密集图像理解](papers/llavar/arxiv-2306.17107/README.md) | 2306.17107v1 |
| LLMGA | [LLMGA：用多模态语言模型组织生成提示与图像编辑](papers/llmga/arxiv-2311.16500/README.md) | 2311.16500v1 |
| LongLLaVA | [LongLLaVA：用Mamba、注意力与视觉压缩扩展多图上下文](papers/longllava/arxiv-2409.02889/README.md) | 2409.02889v1 |
| LongVA | [LongVA：长语言上下文向视觉的迁移](papers/longva/arxiv-2406.16852/README.md) | 2406.16852v1 |
| LongVU | [LongVU：在有限上下文中自适应压缩长视频](papers/longvu/arxiv-2410.17434/README.md) | 2410.17434v1 |
| LTU | [LTU：让语言模型从识别声音走向开放音频问答](papers/ltu/arxiv-2305.10790/README.md) | 2305.10790v1 |
| LVIS-INSTRUCT4V | [LVIS-Instruct4V：让数据生成模型真正看图，改善细粒度视觉指令监督](papers/lvis-instruct4v/arxiv-2311.07574/README.md) | 2311.07574v1 |
| Lynx | [Lynx：多模态输入语言模型的结构、数据与指令训练研究](papers/lynx/arxiv-2307.02469/README.md) | 2307.02469v1 |
| M3IT | [M3IT 与 Ying-VLM：多任务、多语言的视觉指令数据](papers/m3it/arxiv-2306.04387/README.md) | 2306.04387v1 |
| MA-LMM | [MA-LMM：双记忆库与相邻特征压缩的长视频理解](papers/ma-lmm/arxiv-2404.05726/README.md) | 2404.05726v1 |
| Macaw-LLM | [Macaw-LLM：通过词嵌入空间连接图像、音频与视频](papers/macaw-llm/arxiv-2306.09093/README.md) | 2306.09093v1 |
| MetaLM | [MetaLM：以半因果语言模型连接双向感知编码器](papers/metalm/arxiv-2206.06336/README.md) | 2206.06336v1 |
| Meteor | [Meteor：用Mamba与遍历标记压缩多方面推理依据](papers/meteor/arxiv-2405.15574/README.md) | 2405.15574v1 |
| MiMo | [MiMo-V2-Omni：统一感知与代理行动的全模态模型](papers/mimo/document-1ccf04ffaeaa/README.md) | official-source-snapshot-2026-10-08 |
| MiMo | [MiMo-VL：长推理预训练与混合在线强化学习](papers/mimo/arxiv-2506.03569/README.md) | 2506.03569v1 |
| MiMo | [Xiaomi MiMo-V2.5：视觉音频理解与长上下文 Agent 模型](papers/mimo/document-bc763cf700ef/README.md) | official-source-snapshot-2026-10-08 |
| MineDojo | [MineDojo：网络多模态知识与开放世界具身智能体](papers/minedojo/arxiv-2206.08853/README.md) | 2206.08853v1 |
| Mini-Gemini | [Mini-Gemini：双视觉编码器、局部信息挖掘与语言驱动生图](papers/mini-gemini/arxiv-2403.18814/README.md) | 2403.18814v1 |
| MiniCPM-V / MiniCPM-o | [MiniCPM-o 4.5：以时间片组织全双工多模态交互](papers/minicpm/document-9be303a31d11/README.md) | official-source-snapshot-2026-10-08 |
| MiniCPM-V / MiniCPM-o | [MiniCPM-V 4.5：视频压缩、文档恢复与长短推理联合学习](papers/minicpm/document-5679fdac7d01/README.md) | 2509.18154v1 |
| MiniCPM-V / MiniCPM-o | [MiniCPM-V：面向端侧的高分辨率视觉语言模型](papers/minicpm/arxiv-2408.01800/README.md) | 2408.01800v1 |
| MiniGPT | [MiniGPT-4：冻结视觉与语言组件，以线性接口和高质量描述实现视觉对话](papers/minigpt/arxiv-2304.10592/README.md) | 2304.10592v1 |
| MiniGPT | [MiniGPT-v2：任务标识与三阶段视觉多任务训练](papers/minigpt/arxiv-2310.09478/README.md) | 2310.09478v1 |
| Mistral / Pixtral | [Mistral 3：稀疏大模型与端侧多模态模型的联合发布](papers/mistral-pixtral/document-6276d6c37ba5/README.md) | official-source-snapshot-2026-10-08 |
| Mistral / Pixtral | [Pixtral 12B：可变分辨率视觉编码与交错图文指令遵循](papers/mistral-pixtral/document-3ebe839c6607/README.md) | official-source-snapshot-2026-10-08 |
| MM1 | [MM1：用系统消融建立多模态预训练配方](papers/mm1/arxiv-2403.09611/README.md) | 2403.09611v1 |
| MMaDA | [MMaDA：用统一离散扩散完成语言推理、视觉理解与图像生成](papers/mmada/arxiv-2505.15809/README.md) | 2505.15809v1 |
| MMSearch-R1 | [MMSearch-R1：让视觉语言模型学习何时搜索、搜索什么](papers/mmsearch-r1/arxiv-2506.20670/README.md) | 2506.20670v1 |
| MoAI | [MoAI：融合外部视觉专家的场景理解模型](papers/moai/arxiv-2403.07508/README.md) | 2403.07508v1 |
| MobileVLM | [MobileVLM V2：轻量投影、数据扩充与移动端联合训练](papers/mobilevlm/arxiv-2402.03766/README.md) | 2402.03766v1 |
| MobileVLM | [MobileVLM：面向移动设备的视觉语言助手](papers/mobilevlm/arxiv-2312.16886/README.md) | 2312.16886v1 |
| Modality-Integration-Rate | [Modality Integration Rate：从层间分布差异判断多模态对齐](papers/modality-integration-rate/arxiv-2410.07167/README.md) | 2410.07167v1 |
| MoE-LLaVA | [MoE-LLaVA：先学习图文理解，再扩展为稀疏专家模型](papers/moe-llava/arxiv-2401.15947/README.md) | 2401.15947v1 |
| Molmo | [Molmo与PixMo：以人工密集描述和二维指点建设开放多模态模型](papers/molmo/arxiv-2409.17146/README.md) | 2409.17146v1 |
| Monkey / TextMonkey | [Monkey：让高分辨率图像与细粒度描述共同发挥作用](papers/monkey/arxiv-2311.06607/README.md) | 2311.06607v1 |
| Monkey / TextMonkey | [TextMonkey：高分辨率文档理解、视觉token压缩与文字定位](papers/monkey/arxiv-2403.04473/README.md) | 2403.04473v1 |
| MotionGPT | [MotionGPT：动作词表与统一动作语言生成](papers/motiongpt/arxiv-2306.14795/README.md) | 2306.14795v1 |
| MovieChat | [MovieChat：把密集视频 token 转成短期与长期记忆](papers/moviechat/arxiv-2307.16449/README.md) | 2307.16449v1 |
| mPLUG | [mPLUG-DocOwl：以统一指令微调实现无需外部OCR的文档理解](papers/mplug/arxiv-2307.02499/README.md) | 2307.02499v1 |
| mPLUG | [mPLUG-Owl2：以模态专属 K/V 与共享模块促进协作](papers/mplug/arxiv-2311.04257/README.md) | 2311.04257v1 |
| mPLUG | [mPLUG-Owl3：用稀疏 Hyper Attention 读取长图像序列](papers/mplug/arxiv-2408.04840/README.md) | 2408.04840v1 |
| mPLUG | [mPLUG-Owl：以视觉模块学习和图文联合指令训练连接LLaMA](papers/mplug/arxiv-2304.14178/README.md) | 2304.14178v1 |
| mPLUG | [mPLUG-PaperOwl：用上下文和提纲控制科学图表分析](papers/mplug/arxiv-2311.18248/README.md) | 2311.18248v1 |
| MQT-LLaVA | [MQT-LLaVA：用嵌套查询实现可调视觉 token 预算](papers/mqt-llava/arxiv-2405.19315/README.md) | 2405.19315v1 |
| MultiInstruct | [MultiInstruct：以多任务指令微调改善多模态零样本泛化](papers/multiinstruct/arxiv-2212.10773/README.md) | 2212.10773v1 |
| Multimodal Foundation Models Survey | [Multimodal Foundation Models：从视觉专用模型到通用多模态助手](papers/multimodal-foundation-survey/arxiv-2309.10020/README.md) | 2309.10020v1 |
| MultiModal-GPT | [MultiModal-GPT：在 OpenFlamingo 上联合训练图文与纯文本指令](papers/multimodal-gpt/arxiv-2305.04790/README.md) | 2305.04790v1 |
| Large Multimodal Reasoning Survey | [感知、推理、思考与规划：大规模多模态推理模型研究路线综述](papers/multimodal-reasoning-survey/arxiv-2505.04921/README.md) | 2505.04921v1 |
| Muse | [Muse Spark：多模态推理、强化学习与并行 Agent 扩展](papers/muse/document-52928aca6048/README.md) | official-source-snapshot-2026-10-08 |
| NExT-Chat | [NExT-Chat：以位置嵌入统一对话、框定位与分割](papers/next-chat/arxiv-2311.04498/README.md) | 2311.04498v1 |
| NExT-GPT | [NExT-GPT：以语言模型连接四种模态的理解与生成](papers/next-gpt/arxiv-2309.05519/README.md) | 2309.05519v1 |
| NExT-OMNI | [NExT-OMNI：用离散流统一全模态理解、生成与检索](papers/next-omni/arxiv-2510.13721/README.md) | 2510.13721v1 |
| Nexus-O | [Nexus-O：在视觉语言基座上加入语音理解与生成](papers/nexus-o/arxiv-2503.01879/README.md) | 2503.01879v1 |
| OMG-LLaVA | [OMG-LLaVA：用统一分割模型连接图像、对象与像素推理](papers/omg-llava/arxiv-2406.19389/README.md) | 2406.19389v1 |
| Omni-Diffusion | [Omni-Diffusion：以掩码离散扩散统一图像、文字与语音](papers/omni-diffusion/arxiv-2603.06577/README.md) | 2603.06577v1 |
| OmniVinci | [OmniVinci：让视觉、声音与时间在共享空间中对齐](papers/omnivinci/arxiv-2510.15870/README.md) | 2510.15870v1 |
| OneLLM | [OneLLM：共享编码器与投影专家对齐八类输入](papers/onellm/arxiv-2312.03700/README.md) | 2312.03700v1 |
| OpenUni | [OpenUni：轻量查询连接理解与图像生成](papers/openuni/arxiv-2505.23661/README.md) | 2505.23661v1 |
| Osprey | [Osprey：用掩码区域与多尺度视觉特征实现像素级语义理解](papers/osprey/arxiv-2312.10032/README.md) | 2312.10032v1 |
| Otter | [OtterHD：从 Fuyu 出发研究高分辨率与动态图像输入](papers/otter/arxiv-2311.04219/README.md) | 2311.04219v1 |
| Otter / MIMIC-IT | [MIMIC-IT：多模态上下文指令调优与 Otter](papers/otter-398569/arxiv-2306.05425/README.md) | 2306.05425v1 |
| Ovis | [Ovis：用概率视觉词表对齐图像与语言的嵌入结构](papers/ovis/arxiv-2405.20797/README.md) | 2405.20797v1 |
| PaLI | [PaLI-3：以 SigLIP 视觉预训练增强 5B 视觉语言模型](papers/pali/arxiv-2310.09199/README.md) | 2310.09199v1 |
| PaLM-E | [PaLM-E：把连续感知与机器人状态嵌入语言模型](papers/palm-e/arxiv-2303.03378/README.md) | 2303.03378v1 |
| PandaGPT | [PandaGPT：通过共享嵌入空间迁移图文指令跟随能力](papers/pandagpt/arxiv-2305.16355/README.md) | 2305.16355v1 |
| Pengi | [Pengi：用条件文本生成统一音频任务](papers/pengi/arxiv-2305.11834/README.md) | 2305.11834v1 |
| Phi | [Phi-4-Mini 与 Phi-4-Multimodal：用模态 LoRA 保留语言骨干](papers/phi/arxiv-2503.01743/README.md) | 2503.01743v1 |
| π / Pi | [π0.7：用多模态上下文引导通用机器人策略](papers/pi/arxiv-2604.15483/README.md) | 2604.15483v1 |
| Pixel Aligned Language Models | [PixelLLM：逐词连续位置回归与区域语言任务](papers/pixel-aligned-language-models/arxiv-2312.09237/README.md) | 2312.09237v1 |
| PMC-VQA | [PMC-VQA：医学视觉问答数据构建与 MedVInT](papers/pmc-vqa/arxiv-2305.10415/README.md) | 2305.10415v1 |
| PointLLM | [PointLLM：三维点云与语言模型的对齐及评价](papers/pointllm/arxiv-2308.16911/README.md) | 2308.16911v1 |
| Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models | [世界动作模型：从视频预训练到机器人控制的设计空间](papers/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-wor-28acd80eab47/document-28acd80eab47/README.md) | official-source-snapshot-2026-10-08 |
| Prismatic | [Prismatic：视觉语言模型设计空间的控制实验与 PRISM 配方](papers/prismatic/arxiv-2402.07865/README.md) | 2402.07865v1 |
| Prismer | [Prismer：利用独立视觉专家的输出构建视觉语言模型](papers/prismer/arxiv-2303.02506/README.md) | 2303.02506v1 |
| Qwen | [QVQ-72B-Preview：以长程推理增强视觉解题](papers/qwen/document-b615730dc322/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen VLo：统一视觉理解、图像生成与指令编辑](papers/qwen/document-244376c2123a/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen-MM-Plugins：面向多模态代理的技能与工具插件](papers/qwen/document-0aebb3826b29/README.md) | official-readme-2026-10-08 |
| Qwen | [Qwen-RobotWorld](papers/qwen/arxiv-2606.17030/README.md) | 2606.17030v1 |
| Qwen | [Qwen-VL：位置感知视觉压缩与三阶段中英多模态训练](papers/qwen/arxiv-2308.12966/README.md) | 2308.12966v1 |
| Qwen | [Qwen2-VL：动态分辨率、M-RoPE 与统一图像视频理解](papers/qwen/arxiv-2409.12191/README.md) | 2409.12191v1 |
| Qwen | [Qwen2.5-Omni：Thinker-Talker、音视频时间对齐与流式语音生成](papers/qwen/document-1093f7b31e5b/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen2.5-VL：动态分辨率、窗口注意力与绝对时间编码](papers/qwen/qwen2.5-vl/README.md) | 2502.13923v1 |
| Qwen | [Qwen3-Omni：双MoE、时间对齐与逐帧流式语音](papers/qwen/arxiv-2509.17765/README.md) | 2509.17765v1 |
| Qwen | [Qwen3-VL：DeepStack、文字时间戳与 256K 多模态上下文](papers/qwen/arxiv-2511.21631/README.md) | 2511.21631v1 |
| Qwen | [Qwen3.5-Omni：Hybrid MoE、ARIA 与实时音视频交互](papers/qwen/document-54010b187750/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen3.5：原生多模态、线性注意力与稀疏专家](papers/qwen/document-7cdfed4ac9e5/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen3.6-27B：面向代码智能体的27B稠密多模态模型](papers/qwen/document-e122aca8cc58/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen3.6-Plus：面向真实工作流的多模态 Agent 升级](papers/qwen/document-67bea1fbb2c5/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen3.8-Max：超大规模模型、真实工作强化学习与视觉反馈 Agent](papers/qwen/document-ebaf67f69f80/README.md) | official-source-snapshot-2026-10-08 |
| Qwen | [Qwen3.8-Omni-Flash：从视听理解到工具执行与实时交互](papers/qwen/document-2ea7273632dc/README.md) | official-source-snapshot-2026-10-08 |
| RoboPoint | [RoboPoint：合成空间标签与二维行动点预测](papers/robopoint/arxiv-2406.10721/README.md) | 2406.10721v1 |
| SALMONN | [SALMONN：双音频编码器、窗口 Q-Former 与少样本激活微调](papers/salmonn/arxiv-2310.13289/README.md) | 2310.13289v1 |
| SALMONN | [video-SALMONN](papers/salmonn/arxiv-2406.15704/README.md) | 2406.15704v1 |
| See, Say, and Segment: Teaching LMMs to Overcome False Premises | [SESAME：错误前提感知的理解、解释与分割](papers/see-say-and-segment-teaching-lmms-to-overcome-false-premises/arxiv-2312.08366/README.md) | 2312.08366v1 |
| Seed | [Seed1.5-VL：原生分辨率视觉与混合强化学习](papers/seed/arxiv-2505.07062/README.md) | 2505.07062v1 |
| Seed | [Seed1.8：面向真实任务的视觉感知、推理与多步执行](papers/seed/document-6cb43357175b/README.md) | official-source-snapshot-2026-10-08 |
| Seed | [Seed2.0：真实复杂任务的多模态模型卡与工作流评测](papers/seed/arxiv-2607.00248/README.md) | 2607.00248v1 |
| ShareGPT4V | [ShareGPT4V：详细图像描述改善视觉语言对齐](papers/sharegpt4v/arxiv-2311.12793/README.md) | 2311.12793v1 |
| Shikra | [Shikra：普通数字表示的视觉指代对话](papers/shikra/arxiv-2306.15195/README.md) | 2306.15195v1 |
| Show-o | [Show-o2：连续视觉空间统一理解与生成](papers/show-o/arxiv-2506.15564/README.md) | 2506.15564v1 |
| Skywork | [Skywork R1V2：用混合强化学习迁移多模态推理能力](papers/skywork/arxiv-2504.16656/README.md) | 2504.16656v1 |
| SliME | [SliME：全局混合与局部压缩的高分辨率理解](papers/slime/arxiv-2406.08487/README.md) | 2406.08487v1 |
| SlowFast-LLaVA | [SlowFast-LLaVA：在固定 token 预算内兼顾空间细节和时间覆盖](papers/slowfast-llava/arxiv-2407.15841/README.md) | 2407.15841v1 |
| SpatialVLM | [SpatialVLM：合成三维空间监督的视觉语言模型](papers/spatialvlm/arxiv-2401.12168/README.md) | 2401.12168v1 |
| SPHINX | [SPHINX：权重、任务与视觉表示混合](papers/sphinx/arxiv-2311.07575/README.md) | 2311.07575v1 |
| Step | [Step3：通过模型与系统协同设计降低解码成本](papers/step/document-26d59ba3132d/README.md) | 2507.19427v1 |
| StreamChat | [StreamChat：回答生成过程中持续更新视频上下文](papers/streamchat/arxiv-2412.08646/README.md) | 2412.08646v1 |
| SVIT | [SVIT：密集标注驱动的视觉指令数据](papers/svit/arxiv-2307.04087/README.md) | 2307.04087v1 |
| Thyme | [Thyme：代码驱动图像操作与多模态推理](papers/thyme/arxiv-2508.11630/README.md) | 2508.11630v1 |
| TimeChat | [TimeChat：让视频内容与时间戳在视觉编码阶段结合](papers/timechat/arxiv-2312.02051/README.md) | 2312.02051v1 |
| TimeMarker | [TimeMarker：文字时间戳与自适应长短视频表示](papers/timemarker/arxiv-2411.18211/README.md) | 2411.18211v1 |
| UniGen | [UniGen：统一图像理解、生成与自验证](papers/unigen/arxiv-2505.14682/README.md) | 2505.14682v1 |
| UnIVAL | [UnIVAL：统一多模态序列模型](papers/unival/arxiv-2307.16184/README.md) | 2307.16184v1 |
| VaQuitA | [VaQuitA：视频与问题的数据及特征对齐](papers/vaquita/arxiv-2312.02310/README.md) | 2312.02310v1 |
| Vary | [Vary：扩展细粒度视觉表示的文档与图表模型](papers/vary/arxiv-2312.06109/README.md) | 2312.06109v1 |
| Video-ChatGPT | [Video-ChatGPT：时空池化、视频指令数据与对话评价](papers/video-chatgpt/arxiv-2306.05424/README.md) | 2306.05424v1 |
| Video-LLaVA | [Video-LLaVA：先对齐视觉表示，再联合学习图像与视频](papers/video-llava/arxiv-2311.10122/README.md) | 2311.10122v1 |
| Video-MME | [Video-MME-v2：一致性与连贯性的视频评测](papers/video-mme/arxiv-2604.05015/README.md) | 2604.05015v1 |
| VideoChat | [VideoChat3](papers/videochat/arxiv-2607.14935/README.md) | 2607.14935v1 |
| VideoChat | [VideoChat：从视频文本化到视频嵌入的早期多轮对话系统](papers/videochat/arxiv-2305.06355/README.md) | 2305.06355v1 |
| VideoLLaMA | [Video-LLaMA：用双分支适配器连接视频、音频与语言模型](papers/videollama/arxiv-2306.02858/README.md) | 2306.02858v1 |
| VideoLLaMA | [VideoLLaMA 2：用时空卷积连接器与联合训练改善视频和声音理解](papers/videollama/arxiv-2406.07476/README.md) | 2406.07476v1 |
| VideoLLaMA | [VideoLLaMA 3：以图像能力为基础的视频理解模型](papers/videollama/arxiv-2501.13106/README.md) | 2501.13106v1 |
| VideoLLM-online | [VideoLLM-online：流式视频学习与静默决策](papers/videollm-online/arxiv-2406.11816/README.md) | 2406.11816v1 |
| VILA / NVILA | [NVILA：兼顾效果与全流程效率的视觉语言模型](papers/vila/arxiv-2412.04468/README.md) | 2412.04468v1 |
| VILA / NVILA | [VILA²：用模型重写训练数据，再以专家知识突破自增强瓶颈](papers/vila/arxiv-2407.17453/README.md) | 2407.17453v1 |
| VILA / NVILA | [VILA：多模态预训练如何保留语言能力并获得上下文学习能力](papers/vila/arxiv-2312.07533/README.md) | 2312.07533v1 |
| VIMA | [VIMA：用图文交错提示统一机器人操作任务](papers/vima/arxiv-2210.03094/README.md) | 2210.03094v1 |
| VIP-LLaVA | [ViP-LLaVA：任意视觉提示理解](papers/vip-llava/arxiv-2312.00784/README.md) | 2312.00784v1 |
| VisCPM | [VisCPM：多语言骨干驱动的视觉能力迁移](papers/viscpm/arxiv-2308.12038/README.md) | 2308.12038v1 |
| VisionLLM | [VisionLLM：语言指令驱动的通用视觉任务解码](papers/visionllm/arxiv-2305.11175/README.md) | 2305.11175v1 |
| VITA | [Long-VITA：百万 token 的多模态上下文训练与分布式推理](papers/vita/arxiv-2502.05177/README.md) | 2502.05177v1 |
| VITA | [VITA-1.5：逐步引入视觉、语音理解与语音生成](papers/vita/arxiv-2501.01957/README.md) | 2501.01957v1 |
| VITA | [VITA-Audio：用跨模态多 token 预测加速流式语音](papers/vita/arxiv-2505.03739/README.md) | 2505.03739v1 |
| VITA | [VITA-E：并发、可打断的具身交互](papers/vita/arxiv-2510.21817/README.md) | 2510.21817v1 |
| VITA | [VITA-QinYu：在统一语音模型中学习角色扮演与歌唱表达](papers/vita/document-3a7f77dffb3b/README.md) | 2605.06765v1 |
| VITA | [VITA-VLA](papers/vita/arxiv-2510.09607/README.md) | 2510.09607v1 |
| VITA | [VITA：结合多模态理解与可打断语音交互](papers/vita/arxiv-2408.05211/README.md) | 2408.05211v1 |
| VITRON | [VITRON](papers/vitron/document-f588b0cc5a2b/README.md) | official-source-snapshot-2026-10-08 |
| VPGTrans | [VPGTrans：跨语言骨干迁移视觉接口](papers/vpgtrans/arxiv-2305.01278/README.md) | 2305.01278v1 |
| VTimeLLM | [VTimeLLM：以三阶段训练学习视频事件和时间边界](papers/vtimellm/arxiv-2311.18445/README.md) | 2311.18445v1 |
| X-LLM | [X-LLM：独立模态接口连接 ChatGLM](papers/x-llm/arxiv-2305.04160/README.md) | 2305.04160v1 |
| Yi-VL | [Yi-VL：以可训练视觉编码器和分辨率递进构建中英双语视觉助手](papers/yi/document-5991ef3fff72/README.md) | 2403.04652v1 |

## 全量清单与审计

- [完整首轮筛选台账：全部 301 篇](reports/screening-all.md)
- [低于 100 Star：29 篇](reports/excluded-low-star.md)
- [原表无 GitHub 指标：73 篇](reports/no-github.md)
- [补充材料与非模型文献：65 篇](reports/supporting-materials.md)
- [纳入范围待审：17 篇](reports/scope-review.md)
- [完整机器可读数据](data/catalogue.json) · [统计](data/statistics.json) · [Star 证据](data/stars.json)
- [数据完整性与链接验证](data/validation.json) · [独立复核](data/quality-audit.json)
- [文章结构与原图来源检查](data/article-verification.json) · [四篇介绍的独立原文抽查](reports/final-source-spotcheck.md)
- [Seed1.8 介绍的独立原文抽查](reports/seed18-source-spotcheck.md)

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

更新上游清单时加 `--refresh-source`；新题名须补充 scope-mapping，重建脚本会阻止未分类条目进入目录。现有家族分类优先读取 family-mapping 的 `by_entry_id`，调整归类时应同步该字段；新记录再按题名与仓库规则匹配。修改后先审阅 JSON 再重建报告，运行 `python scripts/verify_catalogue.py` 检查数据与内链。目录按当前输入生成；取消整个家族时应移走对应旧目录，避免遗留页面。

已完成文献筛选、项目索引与 **271 篇**介绍。每篇统一采用七部分结构；原文未披露的配置、表文冲突、评估范围与未独立复现的限制均在正文说明。未完成条目独立标识，不计入完成数。PDF/网页阅读缓存位于 `source/papers/`，由 `.gitignore` 排除；原图与出处清单保留在文章目录。

逐篇 `review.json` 记录作者来源核对，`evidence.json` 记录来源版本与资产；它们不代表外部同行评议。运行 `python scripts/verify_articles.py` 可检查七节结构、图片引用与文件哈希；该检查不自动证明数值和解释的语义准确性。更新文章后运行 `python scripts/build_reports.py` 重建阅读入口。
