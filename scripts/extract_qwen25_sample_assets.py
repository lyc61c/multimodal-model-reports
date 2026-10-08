"""Extract faithful Figure 1 regions and original tables for the sample article."""
import hashlib
import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "source/papers/qwen2.5-vl-2502.13923v1.pdf"
DEST = ROOT / "papers/qwen/qwen2.5-vl"

ASSETS = [
    ("fig-01-architecture.png", 3, (69, 68, 523, 372), "Figure 1", "architecture", "Qwen2.5-VL 的整体架构与视觉输入处理", [3]),
    ("fig-01-window-attention-detail.png", 3, (430, 151, 524, 310), "Figure 1（右侧注意力模块局部）", "architecture_detail", "视觉编码器中的窗口与全局注意力", [4]),
    ("fig-01-time-encoding-detail.png", 3, (284, 201, 430, 248), "Figure 1（中央时间编码局部）", "architecture_detail", "动态帧率与绝对时间对齐", [4]),
    ("table-02-training.png", 8, (92, 147, 503, 267), "Table 2", "training_table", "三阶段预训练的数据量、序列长度与更新模块", [6]),
    ("table-03-main-results.png", 11, (74, 213, 520, 413), "Table 3", "result_table", "综合多模态基准的模型比较", [7]),
    ("table-05-document-results.png", 13, (74, 247, 521, 439), "Table 5", "result_table", "OCR、图表和文档理解结果", [7]),
    ("table-08-video-results.png", 14, (75, 509, 519, 715), "Table 8", "result_table", "视频理解与时间定位结果", [7]),
]


def main():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    pdf_sha = hashlib.sha256(PDF.read_bytes()).hexdigest()
    records = []
    with pdfplumber.open(PDF) as pdf:
        for filename, page_no, bbox, label, kind, caption, sections in ASSETS:
            resolution = 400 if kind == "architecture_detail" else 300
            target = DEST / "figures" / filename
            pdf.pages[page_no - 1].crop(bbox).to_image(resolution=resolution, antialias=True).save(target, quantize=False)
            records.append({"asset_id": filename.rsplit(".", 1)[0], "original_label": label, "asset_type": kind,
                            "pdf_page": page_no, "printed_page": str(page_no), "asset_path": "figures/" + filename,
                            "asset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "caption_zh": caption,
                            "source_url": f"https://arxiv.org/pdf/2502.13923v1#page={page_no}", "source_version": "2502.13923v1",
                            "source_sha256": pdf_sha, "processing": {"method": "pdf_region_render", "crop_bbox": bbox,
                            "coordinate_unit": "pt", "coordinate_origin": "top-left", "dpi": resolution,
                            "changes": ["Crop Figure/Table region; partial Figure 1 views explicitly labelled"]}, "used_in_sections": sections})
            print(filename, target.stat().st_size)
    (DEST / "figures-manifest.json").write_text(json.dumps({"paper_id": "qwen2.5-vl", "source_version": "2502.13923v1",
                              "source_pdf_sha256": pdf_sha, "assets": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
