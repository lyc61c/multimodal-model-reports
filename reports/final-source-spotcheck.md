# 四篇介绍的独立来源事实抽查

检查日期：2026-10-08。检查者：独立子代理 `/root/qwen_experiments`。

**结论：本次抽查的四篇介绍未发现需要修改的实质事实错误。** 未修改 README、metadata、证据、图片或索引；本次唯一写入文件为本记录。

检查范围是指定四篇的公式与解释、实际更新模块、数据数量与口径、关键结果与反例、原图编号/页码/内容对应。依据固定缓存的原文 PDF、`pages.json` 和 `source.json`；必要时检查已保存的原文架构图与训练表。原图裁切完整性此前已由撰写者完成最终视觉 QA，本次没有把重新检查全部配图作为独立工作。

这不是复现实验，没有运行模型；也不是全仓库事实认证或全仓库外部同行评审。通过表示所抽查关键陈述与指定版本原文相符，不表示已经独立验证作者的训练数据、运行结果或全部结论。

## 检查对象与版本

| 条目 | 介绍 | 固定来源 |
|---|---|---|
| VideoChat3，6527e8551a9d | [README](../papers/videochat/arxiv-2607.14935/README.md) | [2607.14935v1 原文 PDF](https://arxiv.org/pdf/2607.14935v1)，32 页 |
| VITRON，f588b0cc5a2b | [README](../papers/vitron/document-f588b0cc5a2b/README.md) | [作者公开会议稿 PDF](https://haofei.vip/downloads/papers/Skywork_Vitron_2024.pdf)，33 页 |
| Qwen-RobotWorld，6f94cf51e082 | [README](../papers/qwen/arxiv-2606.17030/README.md) | [2606.17030v1 原文 PDF](https://arxiv.org/pdf/2606.17030v1)，25 页 |
| video-SALMONN，ca69d196c301 | [README](../papers/salmonn/arxiv-2406.15704/README.md) | [2406.15704v1 原文 PDF](https://arxiv.org/pdf/2406.15704v1)，20 页 |

页码均为 PDF 页序，以下结果只适用于上述版本。检查实际使用的本地副本位于 `source/papers/<paper-id>/source.pdf`，该阅读缓存不纳入 Git；表中保留原文链接供仓库读者访问。

## VideoChat3：通过

- **数学与控制器**：PDF 第 4–6 页、Figure 2–3 规定块内时空交互后 T 倍时间池化与 2×2 空间合并，默认 T=4；介绍的 `FP/(4T)` 明确作为忽略边界的近似式，正确区分 16 倍原始 patch 压缩与相对 Qwen3-VL 的实测一半视觉 token。第 6 页控制器确实根据当前 Standby 将下一窗口设为 448²，其余两状态为 224²；保持原宽高比的说明也有来源。
- **状态损失**：PDF 第 14–15 页保留全部状态转换，并从延续位置均匀抽取等量监督；介绍的掩码交叉熵对应原式。2:2:1 被保留为典型循环的有效监督比例；训练采用上一个目标状态，推理采用预测状态，没有误写为完全相同的无误差闭环。
- **更新模块**：PDF 第 11 页 Table 1、第 11–13 页课程说明与介绍一致：Stage 0/1 投影器热身后全参数，Stage 2 全参数，Stage 3 仅投影器和 LLM；临时 Qwen3-4B 解码器在 Stage 0 后丢弃。已有训练表图另作必要视觉核验，模块行、学习率、batch、样本量均对应。
- **数据口径**：第 7 页 Figure 4 的 Academic2M 为约 2.27M；第 9–10 页长视频约 116.2K JSONL 行；第 10 页 Figure 8 为 617,183 原始实例。第 13 页 Stage 2 为 10.33M 样本/23.62M QA，第 15 页 Stage 3 的 OL 为 191K 视频记录/4.69M 可训练轮次。介绍没有将不同计数直接替换或相加成完整训练规模。
- **结果与反例**：第 16–19 页 Tables 2–7 中的 MotionBench 61.7、Video-MME 无字幕70.1、开放 VideoEval-Pro37.8、TimeLens Charades56.1、VUE-TR v2 40.2 均对应正确列。保留了相对 Qwen 开放回答退化、相对 Molmo2 的 TVBench不足、ProactiveVQA EGO28.1低于Qwen34.8，以及学术重标数据的LVBench/MMVU退化。
- **成本和帧预算**：Table 4 的256帧更慢、2048帧20.412s对44.449s及显存值准确；Table 6 的30.2%是像素预算。附录第29–30页Tables10–12支持Video-MME 2FPS/1024帧/80K token、OVO-Timing 4FPS/128帧、Standby后3帧高预算。图注编号与对应页一致。

## VITRON：通过

- **系统结构**：PDF 第 5–6 页支持CLIP逐帧编码与时间平均池化、Vicuna-7B-v1.5、区域mask特征以及混合文本/连续信号调用。第19页Table13的GLIGEN、SEEM、ZeroScope、I2VGen-XL、StableVideo映射正确；介绍未把7B当整个专家系统参数量。
- **数学解释**：第20–22页B.1/B.3支持共享/私有特征拼接、解码侧L2对齐及任务判别器。介绍以标准交叉熵说明判别器最小化、特征学习最大化判别损失，方向正确，且明确属于解释性整理。原文Eq.(4)确实采用预测量乘真实标签对数的异常写法；介绍透明指出这一问题，没有假装给出原作者可执行损失。4层768维判别器见第22–23页。
- **数据与训练**：第20页Table14各调用集合相加为55,028；不是全部训练量。第21–23页支持定位→时空定位→定位QA，以及最终协同损失加各任务损失；第23页明确后端专家还进行针对性预训练。40/35/10h、128/90/128 batch、3e-4/1e-5/1e-5均有来源。正文10张A100和附录10–16张的不同披露被保留，没有强行推算GPU-hours。
- **结果与反例**：第7–9页Tables2–12支持RefCOCO TestA79.5、VidSTG39.5、DAVIS J&F84.2、GQA62.1、ActivityNet-QA51.0、MagicBrush0.274/0.047；FID7.57、视频FID10.11、FVD175.46也正确。IS56.89低于AnimateAny63.87被保留。原文RefCOCO正文与表值相反，以及Table9协同CLIPSIM下降0.5672的疑点，介绍没有抄成确定优势。
- **配图说明**：Figure2/3、Table13/14、Tables8–10、Figure4的页码和内容对应。Figure4双轴的任务分数与执行率未混用；视频编辑5人评分的8.9/8.2有来源，未补造盲评和样本规模。

## Qwen-RobotWorld：通过

- **模型与更新范围**：PDF 第10–11页、Figure3支持60层双流MMDiT，24头×128维=3072、patch2×2、48,360视频token，7B语义编码器/20B主干/127M VAE分别计量。Figure3原图可见Qwen2.5-VL和VAE编码器冻结标记；正文冻结语义编码器并训练connector的说明与介绍一致。
- **数学与条件段**：第11页3D RoPE为16/56/56；第11–12页Figure4与§3.3规定场景F帧、模拟机器人F帧、生成F帧，前两段t=0并排除损失。介绍的注意力与线性flow matching公式明确是解释性形式，第13页只披露目标、标准正态噪声和随长度偏移的时间采样；未误称完整实现公式，也未把排除条件损失等同于阻断所有注意力梯度。
- **数据与采样**：第2页明确70%具身/30%通用及具身部分约90/5/5采样权重。第10页最终8.6M对、200M+帧、4.3M单视角、1.6M多视角、约200K导航/驾驶；第7页的1,744,405驾驶原始clip、10FPS导航和第8页14种机器人、约80K配对episode在介绍中按不同口径保留。第9页caption双粒度等概率采样对应50/50说明。
- **结果与反例**：第14页Tables2–3支持EWMBench21样本/7任务、Overall4.60、HSD0.5660对0.4248、DreamGen总分4.952；GR1行为IF0.832低于LVP0.889被保留。第15页Tables4–5支持PBench平均0.804、WorldModelBench8.99、穿透0.94及物理4.94；介绍正确避免五项满分和全部领先的说法。
- **证据边界与图注**：RoboTwin-IF在第18–19页只有定性案例的限制透明，未捏造量化表或真实机器人控制成功率。Figure1/2/3/4/6与Tables2/4/5对应页和图注一致。

## video-SALMONN：通过

- **结构与训练模块**：PDF 第3–5页支持冻结感知编码器、2Hz视频、32特征/帧、50Hz音频、两层768维MRC Q-Former，以及默认0.5s/5s窗口与3/30查询。第5页支持Vicuna-v1.5 13B默认、LoRA rank32、Q/K/V和前馈权重、约0.4%参数；介绍没有把LLM主体或感知前端写成全参数更新。
- **数学与边界**：第4页Eq.(3)规定各级W×N=C；Eq.(4)确实同时出现左乘投影与D×E形状的方向不一致。介绍用行向量方向整理为H^(r)P_r，明确解释性整理，维度自洽。Eq.(6)从r=2开始、同窗口i≠j余弦相似度双向求和与介绍一致；Eq.(7)是CE加λ多样性损失。第8页支持过大λ损害生成，不是“越多样越好”。
- **数据与zero-shot**：第5–6页总1M且视频<300K，LibriSpeech100h+360h、Ego4D约600h、How2 300h，以及COCO5K/VideoChat11K；附录第12页TextCaps80K、MSVD-QA补充清单和第12–13页两个各1000样本QA集合均相符。第5页脚注明确NExT-QA训练数据存在，zero-shot指未见测试视频与推理指令格式；介绍正确保留这一区别。
- **主结果和消融**：第6–8页Tables2–5支持13B WER2.6、VideoQA49.6、Ego4D-QA49.8、Presentation-QA70.5、AVSSD47.6、AVM79.7；7B WER4.1和AVSSD50.5的反例正确。AVQA60.2→去混合54.0/去多样性53.5/去MRC45.3均正确，去高分辨率的VideoQA49.9/AVM80.6也被保留。Table4重训与Table5置零分支没有混为同一实验。
- **附录与图注**：第15页LRS2音画输入WER4.9对Whisper5.3不是无音频唇读；第16页Table13 spotlight OCR56.1/VideoQA49.1被标成另一个训练变体。第13页Figure6是Presentation-QA示例，第3–4页Figures1–3及第6–7页Tables2–4的编号正确；第14页VQA字词匹配、VideoQA选项协议、GPT-3.5-turbo声源评估与介绍一致。

## 已明确的来源疑点如何处理

本次没有把以下透明披露判成介绍错误：VITRON原文公式、RefCOCO叙述、CLIPSIM消融值和坐标示例的内部疑点；video-SALMONN的投影维度方向和160/150查询数量边界；Qwen-RobotWorld的原始clip与最终样本差别；未公开的超参、评测预算和来源未给出的量化结果。介绍均已区分原文证据、解释性公式和限制，没有把未知项补成事实。

实质发现列表：**空**。建议在现有抽查范围内通过。

## 检查时的 README SHA-256

- VideoChat3：`b02df61b0293d436a2d655902fd9a6c9962e6a86ff9a095ced57eeeb7d826af2`
- VITRON：`58de3a67206e8a184e55e1035cbed016bc209a7eb94a4308a66b89f3e49372d6`
- Qwen-RobotWorld：`7bc00bbd3c61aa350f152755fa81377f9572291dcf1146d148ff3dadc1d0c81b`
- video-SALMONN：`31ad6ec5e92758d715182f3eddbccd64d040986c12939aeef6eec1e2492b753e`
