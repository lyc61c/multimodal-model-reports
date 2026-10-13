"""Render source-grounded architecture plans as clearly labelled original SVG diagrams."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import unicodedata
from pathlib import Path

from build_reports import ROOT, write

PALETTE = {
    "input": ("#eff6ff", "#93c5fd"), "encoder": ("#ecfdf5", "#6ee7b7"),
    "bridge": ("#f5f3ff", "#c4b5fd"), "backbone": ("#eef2ff", "#a5b4fc"),
    "output": ("#fff7ed", "#fdba74"), "unknown": ("#f8fafc", "#94a3b8"),
}
KINDS = {"architecture": "模型架构示意", "public_interface": "公开接口示意", "method_workflow": "方法流程示意"}


def units(text):
    return sum(2 if unicodedata.east_asian_width(character) in "WF" else 1 for character in text)


def wrap(text, maximum):
    lines, current = [], ""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/+\-]*|\n|[^\S\n]+|.",str(text))
    for token in tokens:
        if token == "\n":
            lines.append(current.rstrip())
            current = ""
            continue
        if current and units(current + token) > maximum:
            lines.append(current.rstrip())
            current = ""
        token = token if current else token.lstrip()
        for character in token:
            if current and units(current + character) > maximum:
                lines.append(current.rstrip())
                current = ""
            current += character
    if current:
        lines.append(current)
    return lines or [""]


def validate(plan):
    assert plan.get("version") == 1, "Unsupported diagram plan version"
    assert plan["kind"] in KINDS, "Diagram type is not explicit"
    assert plan["columns"] in (2, 3), "Use two or three columns"
    assert plan.get("sources") and all(source.get("url", "").startswith("https://") and source.get("locator") for source in plan["sources"]), "Every diagram requires located primary sources"
    assert plan.get("notes"), "State the diagram's scope"
    identifiers, cells = set(), set()
    for node in plan["nodes"]:
        assert node["id"] not in identifiers, "Duplicate node"
        identifiers.add(node["id"])
        assert node["kind"] in PALETTE and node["title"], "Invalid component"
        row, col, span = node["row"], node["col"], node.get("span", 1)
        assert isinstance(row, int) and 0 <= row <= 9
        assert isinstance(col, int) and 0 <= col < plan["columns"]
        assert isinstance(span, int) and 1 <= span <= plan["columns"] - col
        for position in range(col, col + span):
            assert (row, position) not in cells, "Overlapping components"
            cells.add((row, position))
    for edge in plan["edges"]:
        assert edge["from"] in identifiers and edge["to"] in identifiers
        assert edge["from"] != edge["to"], "Use an explicit state component rather than a hidden self-loop"
        assert edge.get("style", "solid") in ("solid", "dashed")
        assert edge.get("entry_side") in (None,"left","right"), "Unsupported reviewed edge port"
    return plan


def render(plan):
    validate(plan)
    padding, gap, row_gap = 36, 28, 66
    by_id = {node['id']:node for node in plan['nodes']}
    for edge in plan['edges']:
        if edge.get('label') and by_id[edge['from']]['row']==by_id[edge['to']]['row']:
            gap = max(gap, math.ceil(units(edge['label'])*8.1+28))
    column_width = 310 if plan["columns"] == 2 else 288
    width = padding * 2 + plan["columns"] * column_width + (plan["columns"] - 1) * gap
    headings = wrap(plan["title"], int((width - 70) / 16))
    top = 89 + len(headings) * 35
    nodes, row_heights = {}, {}
    for node in plan["nodes"]:
        span = node.get("span", 1)
        box_width = span * column_width + (span - 1) * gap
        titles = wrap(node["title"], max(10, int((box_width - 30) / 14)))
        details = [line for value in node.get("lines", []) for line in wrap(value, max(12, int((box_width - 30) / 11)))]
        height = max(106, 26 + len(titles)*30 + len(details)*25)
        row_heights[node["row"]] = max(row_heights.get(node["row"], 0), height)
        nodes[node["id"]] = {**node, "x": padding + node["col"] * (column_width + gap),
                             "w": box_width, "h": height, "titles": titles, "details": details}
    y = top
    for row in sorted(row_heights):
        for node in nodes.values():
            if node["row"] == row:
                node["y"] = y + (row_heights[row] - node["h"])/2
        y += row_heights[row] + row_gap
    notes = [line for note in plan["notes"] for line in wrap(note, int((width-76)/10))]
    height = int(y + 60 + len(notes)*25)
    esc = lambda value: html.escape(str(value), quote=True)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           f'<title id="title">{esc(plan["title"])}（自绘，非原文图）</title>',
           f'<desc id="desc">{esc("；".join(plan["notes"]))}</desc>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#64748b"/></marker></defs>',
           f'<rect x="1" y="1" width="{width-2}" height="{height-2}" rx="20" fill="#fff" stroke="#dbe3ef"/>',
           '<g font-family="Microsoft YaHei, Noto Sans CJK SC, Segoe UI, sans-serif">',
           f'<text x="{padding}" y="35" font-size="17" fill="#64748b">自绘 · 非原文图 · {esc(KINDS[plan["kind"]])}</text>']
    for index, line in enumerate(headings):
        svg.append(f'<text x="{width/2}" y="{75+index*35}" text-anchor="middle" font-size="29" font-weight="700" fill="#0f172a">{esc(line)}</text>')

    for edge in plan["edges"]:
        source, target = nodes[edge["from"]], nodes[edge["to"]]
        sx, tx = source["x"]+source["w"]/2, target["x"]+target["w"]/2
        if edge.get('entry_side') and target['row'] > source['row']:
            right=edge['entry_side']=='right'
            sx=source['x']+source['w'] if right else source['x']
            sy=source['y']+source['h']/2
            tx=target['x']+target['w'] if right else target['x']
            ty=target['y']+target['h']/2
            channel=width-padding/2 if right else padding/2
            route=f'M{sx:g},{sy:g} H{channel:g} V{ty:g} H{tx:g}'
            middle,label_x=ty,(channel+tx)/2
        elif target["row"] > source["row"]:
            sy, ty = source["y"]+source["h"], target["y"]
            middle = (sy + ty)/2
            if target["row"] - source["row"] > 1:
                right = sx > width/2
                channel = width-padding/2 if right else padding/2
                sx = source["x"]+source["w"] if right else source["x"]
                sy = source["y"]+source["h"]/2
                middle = ty-row_gap/2
                route = f'M{sx:g},{sy:g} H{channel:g} V{middle:g} H{tx:g} V{ty:g}'
                label_x = (channel+tx)/2
            else:
                route = f'M{sx:g},{sy:g} V{middle:g} H{tx:g} V{ty:g}'
                label_x = (sx+tx)/2
        elif source["row"] == target["row"]:
            forward = target["x"] > source["x"]
            sx = source["x"]+source["w"] if forward else source["x"]
            tx = target["x"] if forward else target["x"]+target["w"]
            sy, ty = source["y"]+source["h"]/2, target["y"]+target["h"]/2
            route = f'M{sx:g},{sy:g} H{(sx+tx)/2:g} V{ty:g} H{tx:g}'
            middle, label_x = min(sy,ty)-3, (sx+tx)/2
        else:
            raise ValueError("Upward edges need an explicit, reviewed layout")
        dashed = ' stroke-dasharray="7 5"' if edge.get("style")=="dashed" else ""
        svg.append(f'<path d="{route}" fill="none" stroke="#64748b" stroke-width="2.2"{dashed} marker-end="url(#arrow)"/>')
        if edge.get("label"):
            label = edge["label"]
            label_width = units(label)*8.1+12
            svg.append(f'<rect x="{label_x-label_width/2:g}" y="{middle-23:g}" width="{label_width:g}" height="23" rx="4" fill="#fff"/>')
            svg.append(f'<text x="{label_x:g}" y="{middle-6:g}" text-anchor="middle" font-size="16" fill="#475569">{esc(label)}</text>')

    for node in nodes.values():
        fill, stroke = PALETTE[node["kind"]]
        dashed = ' stroke-dasharray="7 5"' if node["kind"]=="unknown" else ""
        svg.append(f'<rect x="{node["x"]:g}" y="{node["y"]:g}" width="{node["w"]:g}" height="{node["h"]:g}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.6"{dashed}/>')
        center, baseline = node["x"]+node["w"]/2, node["y"]+35
        for title in node["titles"]:
            svg.append(f'<text x="{center:g}" y="{baseline:g}" text-anchor="middle" font-size="25" font-weight="650" fill="#0f172a">{esc(title)}</text>')
            baseline += 30
        baseline += 1
        for detail in node["details"]:
            svg.append(f'<text x="{center:g}" y="{baseline:g}" text-anchor="middle" font-size="20" fill="#475569">{esc(detail)}</text>')
            baseline += 25
    svg.append(f'<line x1="{padding}" y1="{y-18}" x2="{width-padding}" y2="{y-18}" stroke="#e2e8f0"/>')
    svg.append(f'<text x="{padding}" y="{y+9}" font-size="17" fill="#475569">实线：公开信息流　虚线：可选路径或未披露边界</text>')
    for index, note in enumerate(notes):
        svg.append(f'<text x="{padding}" y="{y+40+index*25}" font-size="17" fill="#64748b">{esc(note)}</text>')
    svg.append('</g></svg>\n')
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--revision-date")
    parser.add_argument("--ids", help="Only render the comma-separated paper IDs; omit for all plans")
    args = parser.parse_args()
    if args.write and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.revision_date or ""):
        parser.error("--write requires --revision-date YYYY-MM-DD")
    changed = []
    plans = sorted(ROOT.glob("papers/*/*/architecture-diagram.json"))
    if args.ids:
        selected = set(args.ids.split(","))
        plans = [path for path in plans if json.loads(path.read_text(encoding="utf-8-sig")).get("paper_id") in selected]
        found = {json.loads(path.read_text(encoding="utf-8-sig"))["paper_id"] for path in plans}
        if selected - found:
            parser.error("No architecture plan for IDs: " + ",".join(sorted(selected - found)))
    for path in plans:
        folder = path.parent
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
        svg = render(plan)
        asset_path = folder / "figures/architecture-overview.svg"
        body_path = folder / "README.md"
        body = body_path.read_text(encoding="utf-8-sig")
        block = ('<!-- derived-architecture:start -->\n'
                 f'![自绘{KINDS[plan["kind"]]}，非原文图](figures/architecture-overview.svg)\n\n'
                 f'*自绘{KINDS[plan["kind"]]}（非原文图）。依据下述公开资料归纳；方框表示功能组件，不代表逐层实现。*\n\n'
                 + '依据：' + '；'.join(f'[{source["locator"]}]({source["url"]})' for source in plan["sources"]) + '。\n'
                 '<!-- derived-architecture:end -->\n\n')
        body = re.sub(r'<!-- derived-architecture:start -->[\s\S]*?<!-- derived-architecture:end -->\n*','',body)
        heading = re.search(r'^## 3[.、] ?整体架构\n+',body,re.M)
        if not heading:
            raise ValueError(f'Missing architecture section: {body_path}')
        body = body[:heading.end()] + block + body[heading.end():]
        manifest_path = folder/'derived-figures-manifest.json'
        existing_manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else {}
        asset_matches = any(asset.get('asset_path')=='figures/architecture-overview.svg'
                            and asset.get('origin')=='self_drawn' and asset.get('sources')==plan['sources']
                            and asset.get('scope')=='；'.join(plan['notes'])
                            and asset.get('asset_sha256')==hashlib.sha256(asset_path.read_bytes()).hexdigest()
                            for asset in existing_manifest.get('assets',[])) if asset_path.exists() else False
        if asset_path.exists() and asset_path.read_text(encoding='utf-8-sig') == svg and body_path.read_text(encoding='utf-8-sig') == body and asset_matches:
            continue
        changed.append(path.relative_to(ROOT).as_posix())
        if not args.write:
            continue
        write(asset_path.relative_to(ROOT).as_posix(),svg)
        write(body_path.relative_to(ROOT).as_posix(),body)
        manifest = {
            'version':1,'paper_id':plan['paper_id'],
            'assets':[{'asset_path':'figures/architecture-overview.svg',
                       'asset_sha256':hashlib.sha256(asset_path.read_bytes()).hexdigest(),
                       'origin':'self_drawn','kind':plan['kind'],'sources':plan['sources'],
                       'scope':'；'.join(plan['notes']),
                       'review':{'date':args.revision_date,'status':'structurally_checked',
                                 'scope':'Source-grounded plan, labelled origin, XML bounds and local links; visual QA is recorded separately'}}],
        }
        write((folder/'derived-figures-manifest.json').relative_to(ROOT).as_posix(),json.dumps(manifest,ensure_ascii=False,indent=2))
        metadata_path = folder/'metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        metadata['chinese_characters'] = len(re.findall(r'[\u4e00-\u9fff]',body))
        write(metadata_path.relative_to(ROOT).as_posix(),json.dumps(metadata,ensure_ascii=False,indent=2))
        review_path = folder/'review.json'
        review = json.loads(review_path.read_text(encoding='utf-8-sig'))
        review['article_sha256'] = hashlib.sha256(body_path.read_bytes()).hexdigest()
        review['derived_architecture_revision'] = {'date':args.revision_date,'kind':plan['kind'],
                                                 'origin':'self_drawn','manifest':'derived-figures-manifest.json',
                                                 'scope':'Diagram added from the explicitly cited architecture plan; original assets retained'}
        write(review_path.relative_to(ROOT).as_posix(),json.dumps(review,ensure_ascii=False,indent=2))
    print(json.dumps({'plans':len(plans),'changed':len(changed),'mode':'write' if args.write else 'check'}))
    if changed and not args.write:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
