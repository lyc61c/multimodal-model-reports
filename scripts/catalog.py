"""Build an auditable catalogue from the two requested upstream sections.

Python standard library only. GitHub HTML is preferred over rounded badges.
No credentials are read and no upstream files or remote repositories are changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/main/README.md"
SECTIONS = ("Multimodal Instruction Tuning (& Latest Works)", "Foundation Models")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "MLLM-Technical-Report-Catalog/1.0", "Accept": "text/html,application/json"})
    with urllib.request.urlopen(request, timeout=35) as response:
        return response.read().decode("utf-8"), response.url


def parse_source(text):
    entries = []
    section = None
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            heading = line[3:].strip()
            section = heading if heading in SECTIONS else None
        if not section or not line.startswith("|") or "**" not in line:
            continue
        # Escaped vertical bars do not separate Markdown table cells.
        cells = re.split(r"(?<!\\)\|", line)[1:]
        if len(cells) < 5:
            raise ValueError(f"Unparseable upstream row at line {number}")
        title_match = re.search(r"\*\*(.*?)\*\*", cells[0])
        title = html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1)))
        title_link = re.search(r"\[\*\*.*?\*\*\]\(([^)]+)\)", cells[0])
        code_links = re.findall(r"\[([^\]]+)\]\(([^)]*)\)", cells[3])
        code_links = [{"label": a, "url": b} for a, b in code_links if b.startswith("http")]
        github = next((x["url"] for x in code_links if urllib.parse.urlparse(x["url"]).hostname == "github.com"), None)
        repo = None
        if github:
            parts = urllib.parse.urlparse(github).path.strip("/").split("/")
            if len(parts) >= 2:
                repo = "/".join(parts[:2]).removesuffix(".git")
        badge = re.search(r"img\.shields\.io/github/stars/([^/?]+/[^/?.]+(?:\.[^/?]+)*?)(?:\.svg|\?|\))", cells[0])
        # Simpler path extraction also handles repository names containing dots.
        badge_path = None
        badge_url = re.search(r"https://img\.shields\.io/github/stars/([^?\s)]+)", cells[0])
        if badge_url:
            badge_path = badge_url.group(1).removesuffix(".svg").strip("/")
        warnings = []
        if badge_path and repo and badge_path.lower() != repo.lower():
            warnings.append(f"Upstream star badge points to {badge_path}; Code column points to {repo}")
        if github and len(urllib.parse.urlparse(github).path.strip("/").split("/")) > 2:
            warnings.append("Stars belong to the parent repository, not this model's branch/subdirectory")
        url = title_link.group(1) if title_link else None
        arxiv = re.search(r"(?:arxiv\.org/(?:pdf|abs)/)(\d{4}\.\d{4,5})", url or "")
        identity = "arxiv:" + arxiv.group(1) if arxiv else "title:" + re.sub(r"\s+", " ", title).casefold()
        entries.append({"id": hashlib.sha256(identity.encode()).hexdigest()[:12], "identity": identity,
                        "title": title, "paper_url": url, "venue": cells[1].strip(), "date": cells[2].strip(),
                        "code_links": code_links, "github_code_url": github, "repo": repo,
                        "badge_repo": badge_path, "source_sections": [section], "source_lines": [number], "warnings": warnings})
    unique = {}
    for entry in entries:
        key = entry["identity"]
        if key not in unique:
            unique[key] = entry
        else:
            previous = unique[key]
            previous["source_sections"] += entry["source_sections"]
            previous["source_lines"] += entry["source_lines"]
            previous["warnings"] = list(dict.fromkeys(previous["warnings"] + entry["warnings"]))
            if previous["repo"] != entry["repo"]:
                previous["warnings"].append("Duplicate rows have different Code repositories")
    return entries, list(unique.values())


def observe_stars(repo):
    observation = {"requested_repo": repo, "source_url": f"https://github.com/{repo}", "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    errors = []
    for attempt in range(2):
        try:
            content, final_url = fetch(observation["source_url"])
            matches = re.findall(r'"stargazerCount"\s*:\s*(\d+)', content)
            if not matches:
                # GitHub's server-rendered sidebar exposes exact small numbers.
                match = re.search(r'href="/[^"]+/stargazers"[^>]*>.*?<strong>([\d,]+)</strong>', content, re.S)
                if match:
                    matches = [match.group(1).replace(",", "")]
            if not matches:
                raise ValueError("Exact stargazerCount not found in GitHub HTML")
            stars = int(matches[0])
            observation.update({"stars": stars, "status": "verified", "method": "github_html_exact", "final_url": final_url,
                                "evidence": f'"stargazerCount":{stars}'})
            return observation
        except Exception as error:
            errors.append(str(error))
            if attempt == 0:
                time.sleep(0.5)
    # Shields JSON is explicitly labelled as rounded if it uses a k/M suffix.
    try:
        value, final_url = fetch(f"https://img.shields.io/github/stars/{repo}.json")
        badge = json.loads(value)
        displayed = str(badge.get("value", badge.get("message", "")))
        if re.fullmatch(r"[\d,]+", displayed):
            observation.update({"stars": int(displayed.replace(",", "")), "status": "verified", "method": "shields_exact_integer",
                                "source_url": final_url, "badge_value": displayed, "github_errors": errors})
        elif re.fullmatch(r"\d+(?:\.\d+)?[kKmM]", displayed):
            observation.update({"stars": None, "status": "above_threshold_rounded", "method": "shields_rounded",
                                "source_url": final_url, "badge_value": displayed, "github_errors": errors})
        else:
            raise ValueError(f"Unusable badge value: {displayed}")
    except Exception as error:
        observation.update({"stars": None, "status": "unresolved", "errors": errors + [str(error)]})
    return observation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-source", action="store_true")
    parser.add_argument("--refresh-stars", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--snapshot-date", default="2026-10-08")
    parser.add_argument("--threshold", type=int, default=100)
    args = parser.parse_args()
    source_path = ROOT / "source/awesome-upstream.md"
    if args.refresh_source or not source_path.exists():
        source, _ = fetch(SOURCE_URL)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(source, encoding="utf-8")
    source = source_path.read_text(encoding="utf-8-sig")
    raw, entries = parse_source(source)
    write_json(ROOT / "data/parsed-entries.json", entries)
    repos = sorted({x["repo"] for x in entries if x["repo"]}, key=str.casefold)
    stars_path = ROOT / "data/stars.json"
    stars = load_json(stars_path, {})
    pending = [repo for repo in repos if args.refresh_stars or repo not in stars or stars[repo].get("status") == "unresolved"]
    print(json.dumps({"raw_rows": len(raw), "unique_entries": len(entries), "unique_repositories": len(repos), "queries": len(pending),
                      "sections": dict(Counter(x["source_sections"][0] for x in raw))}, ensure_ascii=False), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(observe_stars, repo): repo for repo in pending}
        for index, future in enumerate(concurrent.futures.as_completed(jobs), 1):
            repo = jobs[future]
            stars[repo] = future.result()
            write_json(stars_path, stars)
            if index % 20 == 0 or stars[repo].get("stars", 100) is not None and stars[repo].get("stars", 100) < args.threshold:
                print(json.dumps({"completed": index, "total": len(pending), "repo": repo, "stars": stars[repo].get("stars"), "status": stars[repo]["status"]}), flush=True)
    write_json(ROOT / "data/snapshot.json", {"source_url": SOURCE_URL, "upstream_repository": "https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models",
               "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot_date": args.snapshot_date, "threshold": args.threshold,
               "raw_rows": len(raw), "unique_entries": len(entries), "unique_repositories": len(repos),
               "section_rows": dict(Counter(x["source_sections"][0] for x in raw)),
               "duplicates": [{"title": x["title"], "source_lines": x["source_lines"]} for x in entries if len(x["source_lines"]) > 1]})
    print("Saved parsed-entries.json, stars.json and snapshot.json", flush=True)


if __name__ == "__main__":
    main()
