# 原文图表裁切记录

修订日期：2026-10-09。整页论文截图已替换为介绍所需的独立图表区域。

| 项目 | 数量 |
|---|---:|
| 更新的论文介绍 | 189 |
| 移除的整页截图 | 742 |
| 其中：替换为局部图表的原页 | 732 |
| 其中：没有所需图表而直接移除的页面 | 10 |
| 新的独立图表裁图 | 1166 |
| 保持像素不变的原有局部图及网页图 | 350 |
| 更新后全仓库原文配图 | 1516 |
| 遗留整页截图 | 0 |

一张原页可能包含多个需要引用的图表，因此裁图数量大于替换页数。示意图、数据构建图、训练流程图、结果图及表格仅在介绍需要时保留；纯正文、公式段落或无关案例的页面不再作为配图，相关技术说明与原文链接保留。

裁切后逐项对照原页检查模型模块、子图、坐标轴、图例、表头和必要脚注。图注可随图保留；图注与正文混排时，使用介绍中的来源说明，避免重新引入整段正文。较长的表格仍可能使用竖版尺寸，但图片只包含表格内容。

来源继续使用已有的固定 PDF 版本，没有切换到新版本或重绘模型图。每张图片的原图编号、PDF 页码、裁切坐标、源文件哈希及图片哈希见对应文章的 `figures-manifest.json`；完整替换清单见[机器可读记录](../data/figure-crop-audit.json)。坐标为 PDF 点，格式为 `[x0, top, x1, bottom]`，原点位于页面左上角。

本记录核查配图范围和完整性，不代表重新完成全部文章的技术事实审阅，也没有运行模型实验。历史事实抽查保留当时版本的哈希与范围。

## 逐篇变更

| 论文介绍 | 移除原页截图 | 新图表裁图 | 直接移除 |
|---|---:|---:|---:|
| [3D-LLM: Injecting the 3D World into Large Language Models](../papers/3d-llm/arxiv-2307.12981/README.md) | 3 | 5 | 0 |
| [The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World](../papers/all-seeing/arxiv-2308.01907/README.md) | 4 | 8 | 0 |
| [The All-Seeing Project V2: Towards General Relation Comprehension of the Open World](../papers/all-seeing/arxiv-2402.19474/README.md) | 4 | 4 | 0 |
| [ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model](../papers/allava/arxiv-2402.11684/README.md) | 3 | 4 | 0 |
| [An Empirical Study of Scaling Instruction-Tuned Large Multimodal Models](../papers/an-empirical-study-of-scaling-instruction-tuned-large-m-805afa2bda72/arxiv-2309.09958/README.md) | 4 | 5 | 1 |
| [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](../papers/anygpt/arxiv-2402.12226/README.md) | 4 | 7 | 0 |
| [AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model](../papers/anymal/arxiv-2309.16058/README.md) | 3 | 5 | 0 |
| [Apollo: An Exploration of Video Understanding in Large Multimodal Models](../papers/apollo/arxiv-2412.10360/README.md) | 4 | 5 | 0 |
| [AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark](../papers/auroracap/arxiv-2410.03051/README.md) | 4 | 9 | 0 |
| [Emerging Properties in Unified Multimodal Pretraining](../papers/bagel/arxiv-2505.14683/README.md) | 4 | 5 | 0 |
| [Beyond Language Modeling: An Exploration of Multimodal Pretraining](../papers/beyond-language-modeling/arxiv-2603.03276/README.md) | 4 | 6 | 0 |
| [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](../papers/blip/arxiv-2301.12597/README.md) | 4 | 7 | 0 |
| [InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning](../papers/blip/arxiv-2305.06500/README.md) | 4 | 4 | 0 |
| [xGen-MM (BLIP-3): A Family of Open Large Multimodal Models](../papers/blip/arxiv-2408.08872/README.md) | 4 | 6 | 0 |
| [BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset](../papers/blip3-o/arxiv-2505.09568/README.md) | 4 | 6 | 0 |
| [BLIVA: A Simple Multimodal LLM for Better Handling of Text-rich Visual Questions](../papers/bliva/arxiv-2308.09936/README.md) | 3 | 7 | 0 |
| [BRAVE: Broadening the visual encoding of vision-language models](../papers/brave/arxiv-2404.07204/README.md) | 4 | 4 | 0 |
| [BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs](../papers/bubogpt/arxiv-2307.08581/README.md) | 4 | 5 | 0 |
| [Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs](../papers/cambrian-1/arxiv-2406.16860/README.md) | 4 | 7 | 0 |
| [Chameleon: Mixed-Modal Early-Fusion Foundation Models](../papers/chameleon/arxiv-2405.09818/README.md) | 4 | 4 | 0 |
| [ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning](../papers/chartassistant/arxiv-2401.02384/README.md) | 4 | 8 | 0 |
| [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](../papers/chartllama/arxiv-2311.16483/README.md) | 4 | 9 | 0 |
| [ChartMoE: Mixture of Expert Connector for Advanced Chart Understanding](../papers/chartmoe/arxiv-2409.03277/README.md) | 4 | 7 | 0 |
| [Chat-UniVi: Unified Visual Representation Empowers Large Language Models with Image and Video Understanding](../papers/chat-univi/arxiv-2311.08046/README.md) | 4 | 7 | 1 |
| [ChatRex: Taming Multimodal LLM for Joint Perception and Understanding](../papers/chatrex/arxiv-2411.18363/README.md) | 4 | 6 | 0 |
| [ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning](../papers/chatspot/arxiv-2307.09474/README.md) | 4 | 5 | 0 |
| [Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions](../papers/cheetah/arxiv-2308.04152/README.md) | 4 | 8 | 0 |
| [The Claude 3 Model Family: Opus, Sonnet, Haiku](../papers/claude/document-d746d75f8685/README.md) | 4 | 6 | 0 |
| [Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning](../papers/cm3leon/arxiv-2309.02591/README.md) | 4 | 8 | 0 |
| [CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations](../papers/cogcom/arxiv-2402.04236/README.md) | 4 | 6 | 0 |
| [CogVLM: Visual Expert For Large Language Models](../papers/cogvlm/arxiv-2311.03079/README.md) | 4 | 8 | 0 |
| [CogAgent: A Visual Language Model for GUI Agents](../papers/cogvlm/arxiv-2312.08914/README.md) | 3 | 6 | 0 |
| [Comparison Visual Instruction Tuning](../papers/comparison-visual-instruction-tuning/arxiv-2406.09240/README.md) | 4 | 5 | 0 |
| [CompCap: Improving Multimodal Large Language Models with Composite Captions](../papers/compcap/arxiv-2412.05243/README.md) | 4 | 6 | 0 |
| [ConvLLaVA: Hierarchical Backbones as Visual Encoder for Large Multimodal Models](../papers/convllava/arxiv-2405.15738/README.md) | 3 | 5 | 0 |
| [Cosmos 3: Omnimodal World Models for Physical AI](../papers/cosmos/arxiv-2606.02800/README.md) | 4 | 6 | 0 |
| [CoVLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding](../papers/covlm/arxiv-2311.03354/README.md) | 4 | 7 | 0 |
| [CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts](../papers/cumo/arxiv-2405.05949/README.md) | 3 | 5 | 0 |
| [DeepSeek-VL: Towards Real-World Vision-Language Understanding](../papers/deepseek/arxiv-2403.05525/README.md) | 4 | 5 | 0 |
| [DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](../papers/deepseek/arxiv-2412.10302/README.md) | 4 | 5 | 0 |
| [DeepSeek-OCR: Contexts Optical Compression](../papers/deepseek/arxiv-2510.18234/README.md) | 4 | 5 | 0 |
| [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](../papers/deepseek/document-08e9e69bb6bb/README.md) | 4 | 4 | 1 |
| [DeepSeek-OCR 2: Visual Causal Flow](../papers/deepseek/document-536ef2604f4a/README.md) | 4 | 5 | 0 |
| [DeepSeek Harness](../papers/deepseek/document-ddf539694eca/README.md) | 3 | 2 | 1 |
| [DenseWorld-1M: Towards Detailed Dense Grounded Caption in the Real World](../papers/denseworld-1m/arxiv-2506.24102/README.md) | 4 | 6 | 0 |
| [DetGPT: Detect What You Need via Reasoning](../papers/detgpt/arxiv-2305.14167/README.md) | 3 | 3 | 0 |
| [DreamLLM: Synergistic Multimodal Comprehension and Creation](../papers/dreamllm/arxiv-2309.11499/README.md) | 4 | 6 | 0 |
| [EAGLE: Exploring The Design Space for Multimodal LLMs with Mixture of Encoders](../papers/eagle/arxiv-2408.15998/README.md) | 4 | 7 | 0 |
| [Eagle 2.5: Boosting Long-Context Post-Training for Frontier Vision-Language Models](../papers/eagle/arxiv-2504.15271/README.md) | 4 | 7 | 0 |
| [Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning](../papers/ego-r1/arxiv-2506.13654/README.md) | 4 | 4 | 0 |
| [Generative Pretraining in Multimodality](../papers/emu/arxiv-2307.05222/README.md) | 4 | 4 | 0 |
| [Emu3: Next-Token Prediction is All You Need](../papers/emu/arxiv-2409.18869/README.md) | 4 | 8 | 0 |
| [Emu3.5: Native Multimodal Models are World Learners](../papers/emu/arxiv-2510.26583/README.md) | 4 | 7 | 0 |
| [Enhancing Multimodal Large Language Models with Vision Detection Models: An Empirical Study](../papers/enhancing-multimodal-large-language-models-with-vision-e94f5590857f/arxiv-2401.17981/README.md) | 4 | 10 | 0 |
| [Unveiling Encoder-Free Vision-Language Models](../papers/eve/arxiv-2406.11832/README.md) | 4 | 8 | 0 |
| [EVLM: An Efficient Vision-Language Model for Visual Understanding](../papers/evlm/arxiv-2407.14177/README.md) | 4 | 6 | 0 |
| [Ferret: Refer and Ground Anything Anywhere at Any Granularity](../papers/ferret/arxiv-2310.07704/README.md) | 4 | 7 | 0 |
| [Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs](../papers/ferret/arxiv-2404.05719/README.md) | 4 | 5 | 0 |
| [Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context](../papers/gemini/document-1855460a9712/README.md) | 4 | 4 | 0 |
| [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](../papers/gemini/document-8c1dda86c921/README.md) | 4 | 4 | 0 |
| [GLaMM: Pixel Grounding Large Multimodal Model](../papers/glamm/arxiv-2311.03356/README.md) | 3 | 5 | 0 |
| [GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning](../papers/glm/arxiv-2507.01006/README.md) | 4 | 4 | 0 |
| [GPT-4 Technical Report](../papers/gpt/arxiv-2303.08774/README.md) | 4 | 5 | 0 |
| [Addendum to GPT-4o System Card: Native image generation](../papers/gpt/document-0b8790d1de58/README.md) | 3 | 7 | 0 |
| [GPT-6 Astra: A new generation of intelligence](../papers/gpt/document-0d9109ba3678/README.md) | 3 | 3 | 0 |
| [GPT-4V(ision) System Card](../papers/gpt/document-5cd14e1ea9af/README.md) | 3 | 3 | 0 |
| [Hello GPT-4o](../papers/gpt/document-9ee9863086d5/README.md) | 4 | 7 | 0 |
| [Introducing GPT-5](../papers/gpt/document-d0807662cb93/README.md) | 4 | 6 | 0 |
| [GPT4RoI: Instruction Tuning Large Language Model on Region-of-Interest](../papers/gpt4roi/arxiv-2307.03601/README.md) | 3 | 4 | 0 |
| [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](../papers/gpt4tools/arxiv-2305.18752/README.md) | 4 | 4 | 0 |
| [Graphic Design with Large Multimodal Model](../papers/graphist/arxiv-2404.14368/README.md) | 4 | 6 | 0 |
| [GROUNDHOG: Grounding Large Language Models to Holistic Segmentation](../papers/groundhog/arxiv-2402.16846/README.md) | 4 | 12 | 0 |
| [Honeybee: Locality-enhanced Projector for Multimodal LLM](../papers/honeybee/arxiv-2312.06742/README.md) | 4 | 9 | 0 |
| [ImageBind-LLM: Multi-modality Instruction Tuning](../papers/imagebind-llm/arxiv-2309.03905/README.md) | 4 | 6 | 0 |
| [InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition](../papers/internlm-xcomposer/arxiv-2309.15112/README.md) | 4 | 6 | 0 |
| [InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model](../papers/internlm-xcomposer/arxiv-2401.16420/README.md) | 4 | 7 | 0 |
| [InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD](../papers/internlm-xcomposer/arxiv-2404.06512/README.md) | 4 | 9 | 0 |
| [InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output](../papers/internlm-xcomposer/arxiv-2407.03320/README.md) | 4 | 6 | 0 |
| [InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions](../papers/internlm-xcomposer/arxiv-2412.09596/README.md) | 4 | 7 | 0 |
| [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](../papers/internvl/arxiv-2312.14238/README.md) | 4 | 10 | 0 |
| [How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](../papers/internvl/arxiv-2404.16821/README.md) | 4 | 4 | 0 |
| [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](../papers/internvl/arxiv-2412.05271/README.md) | 4 | 5 | 0 |
| [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models](../papers/internvl/arxiv-2504.10479/README.md) | 4 | 5 | 0 |
| [InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency](../papers/internvl/arxiv-2508.18265/README.md) | 4 | 6 | 0 |
| [InternVL-U: Democratizing Unified Multimodal Models for Understanding, Reasoning, Generation and Editing](../papers/internvl/arxiv-2603.09877/README.md) | 4 | 7 | 0 |
| [Kimi-VL Technical Report](../papers/kimi/arxiv-2504.07491/README.md) | 4 | 5 | 0 |
| [Kimi K2.5: Visual Agentic Intelligence](../papers/kimi/arxiv-2602.02276/README.md) | 4 | 4 | 0 |
| [Kimi K3: Open Frontier Intelligence](../papers/kimi/document-eb106a710a29/README.md) | 4 | 4 | 0 |
| [Language Is Not All You Need: Aligning Perception with Language Models](../papers/kosmos/arxiv-2302.14045/README.md) | 4 | 9 | 0 |
| [Kosmos-2: Grounding Multimodal Large Language Models to the World](../papers/kosmos/arxiv-2306.14824/README.md) | 4 | 7 | 0 |
| [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](../papers/lamm/arxiv-2306.06687/README.md) | 4 | 7 | 0 |
| [LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment](../papers/languagebind/arxiv-2310.01852/README.md) | 4 | 5 | 0 |
| [Cheap and Quick: Efficient Vision-Language Instruction Tuning for Large Language Models](../papers/lavin/arxiv-2305.15023/README.md) | 4 | 8 | 0 |
| [Unified Language-Vision Pretraining in LLM with Dynamic Discrete Visual Tokenization](../papers/lavit/arxiv-2309.04669/README.md) | 4 | 5 | 1 |
| [An Embodied Generalist Agent in 3D World](../papers/leo/arxiv-2311.12871/README.md) | 4 | 6 | 0 |
| [Libra: Building Decoupled Vision System on Large Language Models](../papers/libra/arxiv-2405.10140/README.md) | 4 | 6 | 0 |
| [LION : Empowering Multimodal Large Language Model with Dual-Level Visual Knowledge](../papers/lion/arxiv-2311.11860/README.md) | 4 | 9 | 0 |
| [LISA: Reasoning Segmentation via Large Language Model](../papers/lisa/arxiv-2308.00692/README.md) | 4 | 7 | 0 |
| [LITA: Language Instructed Temporal-Localization Assistant](../papers/lita/arxiv-2403.19046/README.md) | 4 | 6 | 0 |
| [LL3DA: Visual Interactive Instruction Tuning for Omni-3D Understanding, Reasoning, and Planning](../papers/ll3da/arxiv-2311.18651/README.md) | 4 | 12 | 0 |
| [LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention](../papers/llama-adapter/arxiv-2303.16199/README.md) | 4 | 8 | 0 |
| [LLaMA-Adapter V2: Parameter-Efficient Visual Instruction Model](../papers/llama-adapter/arxiv-2304.15010/README.md) | 4 | 7 | 0 |
| [LLaMA-VID: An Image is Worth 2 Tokens in Large Language Models](../papers/llama-vid/arxiv-2311.17043/README.md) | 4 | 6 | 0 |
| [The Llama 3 Herd of Models](../papers/llama/arxiv-2407.21783/README.md) | 4 | 4 | 0 |
| [LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs](../papers/llamav-o1/arxiv-2501.06186/README.md) | 4 | 6 | 0 |
| [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](../papers/llava-med/arxiv-2306.00890/README.md) | 4 | 6 | 0 |
| [LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation](../papers/llava-mod/arxiv-2408.15881/README.md) | 4 | 5 | 0 |
| [Visual Instruction Tuning](../papers/llava/arxiv-2304.08485/README.md) | 4 | 4 | 0 |
| [Improved Baselines with Visual Instruction Tuning](../papers/llava/arxiv-2310.03744/README.md) | 4 | 7 | 0 |
| [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](../papers/llava/arxiv-2311.05437/README.md) | 4 | 6 | 0 |
| [LLaVA-OneVision: Easy Visual Task Transfer](../papers/llava/arxiv-2408.03326/README.md) | 4 | 7 | 0 |
| [LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training](../papers/llava/arxiv-2509.23661/README.md) | 4 | 4 | 0 |
| [LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding](../papers/llavar/arxiv-2306.17107/README.md) | 4 | 6 | 0 |
| [LLMGA: Multimodal Large Language Model based Generation Assistant](../papers/llmga/arxiv-2311.16500/README.md) | 4 | 8 | 0 |
| [LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via Hybrid Architecture](../papers/longllava/arxiv-2409.02889/README.md) | 4 | 6 | 0 |
| [Long Context Transfer from Language to Vision](../papers/longva/arxiv-2406.16852/README.md) | 4 | 7 | 0 |
| [LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding](../papers/longvu/arxiv-2410.17434/README.md) | 4 | 7 | 0 |
| [Listen, Think, and Understand](../papers/ltu/arxiv-2305.10790/README.md) | 4 | 5 | 0 |
| [To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning](../papers/lvis-instruct4v/arxiv-2311.07574/README.md) | 4 | 5 | 1 |
| [What Matters in Training a GPT4-Style Language Model with Multimodal Inputs?](../papers/lynx/arxiv-2307.02469/README.md) | 4 | 6 | 0 |
| [M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning](../papers/m3it/arxiv-2306.04387/README.md) | 4 | 9 | 0 |
| [MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding](../papers/ma-lmm/arxiv-2404.05726/README.md) | 4 | 13 | 0 |
| [Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration](../papers/macaw-llm/arxiv-2306.09093/README.md) | 4 | 5 | 0 |
| [Language Models are General-Purpose Interfaces](../papers/metalm/arxiv-2206.06336/README.md) | 4 | 6 | 0 |
| [Meteor: Mamba-based Traversal of Rationale for Large Language and Vision Models](../papers/meteor/arxiv-2405.15574/README.md) | 4 | 3 | 1 |
| [MiMo-VL Technical Report](../papers/mimo/arxiv-2506.03569/README.md) | 4 | 5 | 0 |
| [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](../papers/minedojo/arxiv-2206.08853/README.md) | 4 | 4 | 0 |
| [Mini-Gemini: Mining the Potential of Multi-modality Vision Language Models](../papers/mini-gemini/arxiv-2403.18814/README.md) | 4 | 5 | 0 |
| [MiniCPM-V: A GPT-4V Level MLLM on Your Phone](../papers/minicpm/arxiv-2408.01800/README.md) | 4 | 5 | 0 |
| [MiniCPM-V 4.5: A GPT-4o Level MLLM for Single Image, Multi Image and Video Understanding on Your Phone](../papers/minicpm/document-5679fdac7d01/README.md) | 4 | 6 | 0 |
| [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](../papers/minigpt/arxiv-2304.10592/README.md) | 4 | 5 | 0 |
| [MiniGPT-v2: large language model as a unified interface for vision-language multi-task learning](../papers/minigpt/arxiv-2310.09478/README.md) | 4 | 6 | 0 |
| [MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training](../papers/mm1/arxiv-2403.09611/README.md) | 4 | 4 | 0 |
| [MMaDA: Multimodal Large Diffusion Language Models](../papers/mmada/arxiv-2505.15809/README.md) | 4 | 6 | 0 |
| [MMSearch-R1: Incentivizing LMMs to Search](../papers/mmsearch-r1/arxiv-2506.20670/README.md) | 4 | 4 | 0 |
| [MoAI: Mixture of All Intelligence for Large Language and Vision Models](../papers/moai/arxiv-2403.07508/README.md) | 4 | 5 | 0 |
| [MobileVLM : A Fast, Reproducible and Strong Vision Language Assistant for Mobile Devices](../papers/mobilevlm/arxiv-2312.16886/README.md) | 4 | 7 | 0 |
| [MobileVLM V2: Faster and Stronger Baseline for Vision Language Model](../papers/mobilevlm/arxiv-2402.03766/README.md) | 4 | 8 | 0 |
| [Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate](../papers/modality-integration-rate/arxiv-2410.07167/README.md) | 4 | 5 | 1 |
| [MoE-LLaVA: Mixture of Experts for Large Vision-Language Models](../papers/moe-llava/arxiv-2401.15947/README.md) | 4 | 7 | 0 |
| [Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models](../papers/molmo/arxiv-2409.17146/README.md) | 4 | 3 | 1 |
| [Monkey: Image Resolution and Text Label Are Important Things for Large Multi-modal Models](../papers/monkey/arxiv-2311.06607/README.md) | 4 | 7 | 0 |
| [TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document](../papers/monkey/arxiv-2403.04473/README.md) | 4 | 10 | 0 |
| [MotionGPT: Human Motion as a Foreign Language](../papers/motiongpt/arxiv-2306.14795/README.md) | 4 | 6 | 0 |
| [MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](../papers/moviechat/arxiv-2307.16449/README.md) | 4 | 5 | 0 |
| [mPLUG-Owl: Modularization Empowers Large Language Models with Multimodality](../papers/mplug/arxiv-2304.14178/README.md) | 4 | 7 | 0 |
| [mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding](../papers/mplug/arxiv-2307.02499/README.md) | 4 | 6 | 0 |
| [mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration](../papers/mplug/arxiv-2311.04257/README.md) | 4 | 9 | 0 |
| [mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model](../papers/mplug/arxiv-2311.18248/README.md) | 4 | 7 | 0 |
| [mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models](../papers/mplug/arxiv-2408.04840/README.md) | 4 | 7 | 0 |
| [Matryoshka Query Transformer for Large Vision-Language Models](../papers/mqt-llava/arxiv-2405.19315/README.md) | 4 | 5 | 0 |
| [MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning](../papers/multiinstruct/arxiv-2212.10773/README.md) | 4 | 7 | 1 |
| [Multimodal Foundation Models: From Specialists to General-Purpose Assistants](../papers/multimodal-foundation-survey/arxiv-2309.10020/README.md) | 4 | 6 | 0 |
| [MultiModal-GPT: A Vision and Language Model for Dialogue with Humans](../papers/multimodal-gpt/arxiv-2305.04790/README.md) | 3 | 4 | 0 |
| [NExT-Chat: An LMM for Chat, Detection and Segmentation](../papers/next-chat/arxiv-2311.04498/README.md) | 4 | 7 | 0 |
| [NExT-GPT: Any-to-Any Multimodal LLM](../papers/next-gpt/arxiv-2309.05519/README.md) | 4 | 12 | 0 |
| [NExT-OMNI: Towards Any-to-Any Omnimodal Foundation Models with Discrete Flow Matching](../papers/next-omni/arxiv-2510.13721/README.md) | 4 | 5 | 0 |
| [Nexus-O: An Omni-Perceptive And -Interactive Model for Language, Audio, And Vision](../papers/nexus-o/arxiv-2503.01879/README.md) | 4 | 6 | 0 |
| [OMG-LLaVA: Bridging Image-level, Object-level, Pixel-level Reasoning and Understanding](../papers/omg-llava/arxiv-2406.19389/README.md) | 4 | 9 | 0 |
| [Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion](../papers/omni-diffusion/arxiv-2603.06577/README.md) | 4 | 8 | 0 |
| [OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM](../papers/omnivinci/arxiv-2510.15870/README.md) | 4 | 6 | 0 |
| [OneLLM: One Framework to Align All Modalities with Language](../papers/onellm/arxiv-2312.03700/README.md) | 4 | 11 | 0 |
| [OpenUni: A Simple Baseline for Unified Multimodal Understanding and Generation](../papers/openuni/arxiv-2505.23661/README.md) | 4 | 6 | 0 |
| [Osprey: Pixel Understanding with Visual Instruction Tuning](../papers/osprey/arxiv-2312.10032/README.md) | 4 | 8 | 0 |
| [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](../papers/otter-398569/arxiv-2306.05425/README.md) | 4 | 5 | 0 |
| [OtterHD: A High-Resolution Multi-modality Model](../papers/otter/arxiv-2311.04219/README.md) | 4 | 8 | 0 |
| [Ovis: Structural Embedding Alignment for Multimodal Large Language Model](../papers/ovis/arxiv-2405.20797/README.md) | 4 | 5 | 0 |
| [PaLI-3 Vision Language Models: Smaller, Faster, Stronger](../papers/pali/arxiv-2310.09199/README.md) | 4 | 6 | 0 |
| [PaLM-E: An Embodied Multimodal Language Model](../papers/palm-e/arxiv-2303.03378/README.md) | 4 | 10 | 0 |
| [PandaGPT: One Model To Instruction-Follow Them All](../papers/pandagpt/arxiv-2305.16355/README.md) | 4 | 4 | 0 |
| [Pengi: An Audio Language Model for Audio Tasks](../papers/pengi/arxiv-2305.11834/README.md) | 4 | 8 | 0 |
| [Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs](../papers/phi/arxiv-2503.01743/README.md) | 4 | 4 | 0 |
| [π0.7: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities](../papers/pi/arxiv-2604.15483/README.md) | 4 | 7 | 0 |
| [Pixel Aligned Language Models](../papers/pixel-aligned-language-models/arxiv-2312.09237/README.md) | 4 | 8 | 0 |
| [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](../papers/pmc-vqa/arxiv-2305.10415/README.md) | 4 | 5 | 0 |
| [PointLLM: Empowering Large Language Models to Understand Point Clouds](../papers/pointllm/arxiv-2308.16911/README.md) | 4 | 6 | 0 |
| [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](../papers/prismatic/arxiv-2402.07865/README.md) | 4 | 4 | 0 |
| [Prismer: A Vision-Language Model with An Ensemble of Experts](../papers/prismer/arxiv-2303.02506/README.md) | 4 | 4 | 0 |
| [Qwen-VL: A Frontier Large Vision-Language Model with Versatile Abilities](../papers/qwen/arxiv-2308.12966/README.md) | 4 | 7 | 0 |
| [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](../papers/qwen/arxiv-2409.12191/README.md) | 4 | 7 | 0 |
| [Qwen3-Omni Technical Report](../papers/qwen/arxiv-2509.17765/README.md) | 4 | 7 | 0 |
| [Qwen3-VL Technical Report](../papers/qwen/arxiv-2511.21631/README.md) | 4 | 5 | 0 |
| [Qwen2.5-Omni Technical Report](../papers/qwen/document-1093f7b31e5b/README.md) | 5 | 7 | 0 |
| [RoboPoint: A Vision-Language Model for Spatial Affordance Prediction for Robotics](../papers/robopoint/arxiv-2406.10721/README.md) | 4 | 9 | 0 |
| [SALMONN: Towards Generic Hearing Abilities for Large Language Models](../papers/salmonn/arxiv-2310.13289/README.md) | 4 | 7 | 0 |
| [Seed1.5-VL Technical Report](../papers/seed/arxiv-2505.07062/README.md) | 4 | 5 | 0 |
| [Seed1.8 Model Card: Towards Generalized Real-World Agency](../papers/seed/document-6cb43357175b/README.md) | 4 | 7 | 0 |
| [VITA-QinYu: Expressive Spoken Language Model for Role-Playing and Singing](../papers/vita/document-3a7f77dffb3b/README.md) | 5 | 6 | 0 |
| [Yi-VL](../papers/yi/document-5991ef3fff72/README.md) | 3 | 4 | 0 |
