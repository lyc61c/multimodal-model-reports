"""Check article integrity, reading links, dates and optional local source caches."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import html.parser
import json
import re
import subprocess
import sys
import unicodedata
import urllib.parse
from collections import Counter
from pathlib import Path

from build_reports import ROOT, completed_articles, read, write
from format_articles import SECTIONS, markdown_segments


def anchors(body):
    result = set(re.findall(r'<a\s+id="([^"]+)"', body))
    counts = Counter()
    for heading in re.findall(r"^#{1,6} (.+)$", body, re.M):
        heading = re.sub(r"<[^>]+>", "", heading).replace("`", "").replace("*", "")
        slug = "".join(c for c in heading.lower() if c in "_- " or unicodedata.category(c)[0] in "LN")
        slug = slug.replace(" ", "-")
        result.add(slug + (f"-{counts[slug]}" if counts[slug] else ""))
        counts[slug] += 1
    return result


def without_code(body):
    return "".join(text for text, is_code in markdown_segments(body) if not is_code)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity(url):
    """Compare known raw/download aliases without dropping meaningful URL queries."""
    parsed = urllib.parse.urlsplit(url or "")
    host, path = parsed.netloc.lower(), parsed.path.rstrip("/")
    query = urllib.parse.urlencode(sorted((key, value) for key, value in
                                         urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                                         if (key, value) != ("output", "1")))
    if host == "raw.githubusercontent.com":
        host = "github.com"
        parts = path.split("/")
        if len(parts) >= 5:
            path = "/".join(parts[:3] + ["blob"] + parts[3:])
    if host == "huggingface.co":
        path = path.replace("/raw/", "/blob/", 1)
    return host, path, query


def matches_source(declaration, record):
    identity = source_identity(declaration.get("source_url"))
    return bool(identity[0]) and any(identity == source_identity(record.get(key))
                                   for key in ("source_url", "request_url", "final_url", "canonical_url"))


def verified_bytes(path, expected_hash, expected_bytes=None, allow_windows_text=False):
    """Return exact original bytes; tolerate only a proved, one-pass Windows transform."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""):
        raise ValueError(f"Missing or invalid original SHA256: {path.name}")
    if not path.is_file():
        raise ValueError(f"Missing original source object: {path.name}")
    cached = path.read_bytes()
    candidates = [(cached, "original_bytes")]
    if allow_windows_text and b"\r\n" in cached:
        candidates.append((cached.replace(b"\r\n", b"\n"), "reversed_windows_newlines"))
    for original, representation in candidates:
        if (hashlib.sha256(original).hexdigest() == expected_hash and
                (expected_bytes is None or len(original) == expected_bytes)):
            return original, representation
    raise ValueError(f"Original source SHA256 or byte length differs: {path.name}")


def verify_source_object(cache, record, statistics, basename="source"):
    if record.get("status", "fetched") != "fetched":
        raise ValueError(f"Source cache status is not fetched: {basename}.json")
    kind = record.get("kind", "html" if basename == "release" else None)
    if kind not in ("pdf", "html"):
        raise ValueError(f"Unsupported source cache kind: {kind}")
    if record.get("content_sha256"):
        if kind != "html" or basename != "source":
            raise ValueError("API HTML derivative has an incompatible source kind")
        original, representation = verified_bytes(cache / "source-response.json", record.get("sha256"),
                                                   record.get("bytes"))
        try:
            content = json.loads(original)["data"]["content"]
            if not isinstance(content, str):
                raise ValueError("API data.content is not text")
            content = content.encode("utf-8")
        except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("API original JSON has no valid data.content") from error
        if hashlib.sha256(content).hexdigest() != record["content_sha256"]:
            raise ValueError("API extracted content SHA256 differs")
        derived, derived_representation = verified_bytes(cache / "source.html", record["content_sha256"],
                                                          len(content), allow_windows_text=True)
        if derived != content:
            raise ValueError("API cached HTML differs from original JSON data.content")
        statistics["api_json_with_verified_html_derivative"] += 1
        statistics["derived_html_" + derived_representation] += 1
    else:
        original, representation = verified_bytes(cache / f"{basename}.{kind}", record.get("sha256"),
                                                   record.get("bytes"), allow_windows_text=kind == "html")
        statistics[kind + "_" + representation] += 1
    statistics["original_source_objects_verified"] += 1
    return {"record": record, "bytes": original, "kind": kind}


class VisibleSourceText(html.parser.HTMLParser):
    """Read publication dates from verified HTML, rather than unverified extracted text."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def verify_article_sources(cache, metadata, evidence, statistics):
    record = json.loads((cache / "source.json").read_text(encoding="utf-8-sig"))
    objects = [verify_source_object(cache, record, statistics)]
    release_record = cache / "release-source.json"
    if release_record.exists():
        release = json.loads(release_record.read_text(encoding="utf-8-sig"))
        objects.append(verify_source_object(cache, release, statistics, basename="release"))

    versions = [part.strip() for part in metadata["source_version"].split(";")]
    primary = next((obj for obj in objects if matches_source(metadata, obj["record"])), None)
    if primary is None:
        raise ValueError("Declared primary source URL does not identify a verified cache object")
    primary_record = primary["record"]
    declared_hashes = [metadata[key] for key in ("source_sha256", "source_pdf_sha256") if metadata.get(key)]
    expected_hash = declared_hashes[0] if declared_hashes else None
    if any(value != primary_record["sha256"] for value in declared_hashes):
        raise ValueError("Declared primary source SHA256 differs from original cache record")
    if "source_version" in primary_record:
        if primary_record["source_version"] not in versions:
            raise ValueError("Declared primary source version differs from original cache record")
        versions.remove(primary_record["source_version"])
    else:
        # Historical release records store URL+raw SHA but no independent snapshot label.
        # Require the explicit metadata/evidence binding; do not substitute the PDF version.
        if not (expected_hash and len(versions) == 1 and
                re.fullmatch(r"official-source-snapshot-\d{4}-\d{2}-\d{2}", versions[0]) and
                evidence.get("source_sha256") == expected_hash and
                evidence.get("source_version") == metadata["source_version"] and
                source_identity(evidence.get("source_url")) == source_identity(metadata["source_url"])):
            raise ValueError("Primary release snapshot lacks an explicit matching version/URL/SHA binding")
        versions.clear()
        statistics["snapshot_labels_bound_by_metadata_and_evidence_sha"] += 1
    statistics["primary_sources_verified"] += 1
    used = {id(primary)}

    for declaration in metadata.get("supplemental_sources", []):
        obj = next((obj for obj in objects if matches_source(declaration, obj["record"])), None)
        if obj is None:
            raise ValueError("Declared supplemental source URL has no verified cache object")
        if (declaration.get("source_version") != obj["record"].get("source_version") or
                declaration.get("source_sha256") != obj["record"].get("sha256")):
            raise ValueError("Declared supplemental source version or SHA256 differs")
        used.add(id(obj))
        statistics["explicit_supplemental_sources_verified"] += 1

    for version in versions:
        match = re.fullmatch(r"release-(\d{4})-(\d{2})-(\d{2})", version)
        releases = [obj for obj in objects if id(obj) not in used and obj["kind"] == "html"]
        if not match or len(releases) != 1:
            raise ValueError(f"Composite source version has no distinct verified release object: {version}")
        release = releases[0]
        # A composite release must also be named by article evidence or the article's official links.
        declared_release_urls = evidence.get("_article_source_links", [])
        if not any(source_identity(url) == source_identity(release["record"]["source_url"])
                   for url in declared_release_urls):
            raise ValueError("Composite release cache URL is not declared by the article")
        year, month, day = map(int, match.groups())
        if not 1 <= month <= 12 or not 1 <= day <= 31:
            raise ValueError(f"Invalid composite release date: {version}")
        parser = VisibleSourceText()
        parser.feed(release["bytes"].decode("utf-8"))
        opening = " ".join(" ".join(parser.parts).split())[:3000]
        if f"{calendar.month_name[month]} {day}, {year}" not in opening:
            raise ValueError(f"Composite release date is absent from verified HTML opening: {version}")
        used.add(id(release))
        statistics["composite_release_sources_verified"] += 1
    if len(used) != len(objects):
        raise ValueError("A verified current cache object has no declared primary or supplemental role")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", action="store_true", help="Also verify ignored original source caches on this machine")
    args = parser.parse_args()
    for script in ("verify_catalogue.py", "verify_articles.py", "verify_github_math.py"):
        completed = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script)],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode:
            print(completed.stdout + completed.stderr)
            raise SystemExit(completed.returncode)

    issues = []
    documents = [ROOT / "README.md", *sorted(ROOT.glob("reports/*.md")), *sorted(ROOT.glob("papers/**/README.md"))]
    bodies = {path.resolve(): path.read_text(encoding="utf-8-sig") for path in documents}
    anchor_cache = {path: anchors(body) for path, body in bodies.items()}
    links_checked = 0
    for document, body in bodies.items():
        targets = re.findall(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", body)
        html = without_code(body)
        targets += re.findall(r'\b(?:src|href)="([^"]+)"', html)
        for srcset in re.findall(r'\bsrcset="([^"]+)"', html):
            targets += [item.strip().split()[0] for item in srcset.split(",") if item.strip()]
        for link in targets:
            if link.startswith("<") and link.endswith(">"):
                link = link[1:-1]
            parsed = urllib.parse.urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (document.parent / urllib.parse.unquote(parsed.path)).resolve() if parsed.path else document
            links_checked += 1
            if not target.is_relative_to(ROOT):
                issues.append({"file": document.relative_to(ROOT).as_posix(), "problem": "Link leaves repository", "link": link})
            elif not target.exists():
                issues.append({"file": document.relative_to(ROOT).as_posix(), "problem": "Missing local target", "link": link})
            elif parsed.fragment and target.suffix.lower() == ".md":
                fragment = urllib.parse.unquote(parsed.fragment)
                if fragment not in anchor_cache.get(target, set()):
                    issues.append({"file": document.relative_to(ROOT).as_posix(), "problem": "Missing section anchor", "link": link})

    articles = completed_articles()
    plan = read("data/article-plan.json")["entries"]
    by_folder = {(ROOT / entry["destination"]).resolve(): entry for entry in plan}
    expected_paths = {article["path"].resolve(): article["date"] for article in articles}
    guide_path = ROOT / "reports/paper-reading-guide.md"
    guide = bodies[guide_path.resolve()]
    listed = re.findall(r"^\| (\d{4}-\d{2}-\d{2}) \|.*?\]\(([^)]+)\)", guide, re.M)
    listed_dates = [value for value, _ in listed]
    listed_paths = {(guide_path.parent / urllib.parse.unquote(link)).resolve(): value for value, link in listed}
    chronological_navigation = listed_paths == expected_paths and len(listed) == len(articles) and listed_dates == sorted(listed_dates)
    if not chronological_navigation:
        issues.append({"file": "reports/paper-reading-guide.md", "problem": "Chronological guide coverage or order differs"})

    source_caches = 0
    source_statistics = Counter()
    for index, article in enumerate(articles):
        path = article["path"].resolve()
        relative = path.relative_to(ROOT).as_posix()
        body = bodies[path]
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8-sig"))
        h1 = re.findall(r"^# (.+)$", body, re.M)
        if len(h1) != 1 or metadata.get("title_zh") != h1[0]:
            issues.append({"file": relative, "problem": "Document title differs from metadata"})
        if any(body.count(f'<a id="{anchor}"></a>') != 1 for anchor in ["article-toc", *(key for key, _ in SECTIONS)]):
            issues.append({"file": relative, "problem": "Missing or repeated stable section anchor"})
        if any(body.count(f"<!-- {name}:start -->") != 1 for name in ("article-header", "article-toc", "article-footer")):
            issues.append({"file": relative, "problem": "Reading navigation is incomplete"})
        for adjacent in (articles[index - 1] if index else None, articles[index + 1] if index + 1 < len(articles) else None):
            if adjacent:
                footer = body.split("<!-- article-footer:start -->")[-1]
                adjacent_targets = {(path.parent / urllib.parse.unquote(link)).resolve()
                                    for link in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", footer)}
                if adjacent["path"].resolve() not in adjacent_targets:
                    issues.append({"file": relative, "problem": "Previous/next article differs from date order"})
        prose = without_code(body)
        if re.search(r"(?<!\\)\\[([]", prose):
            issues.append({"file": relative, "problem": "Legacy math delimiters remain"})
        if len(re.findall(r"(?<!\\)\$", prose)) % 2:
            issues.append({"file": relative, "problem": "Unbalanced dollar math delimiters"})
        if "![](" in body:
            issues.append({"file": relative, "problem": "Image lacks alternative text"})
        if metadata.get("chinese_characters") != len(re.findall(r"[\u4e00-\u9fff]", body)):
            issues.append({"file": relative, "problem": "Character count differs from document"})
        if args.sources:
            entry = by_folder[path.parent]
            cache = ROOT / "source/papers" / entry["id"]
            try:
                if entry.get("arxiv_id") == "2502.13923":
                    source_file = ROOT / "source/papers/qwen2.5-vl-2502.13923v1.pdf"
                    verified_bytes(source_file, metadata["source_pdf_sha256"])
                    source_statistics.update({"pdf_original_bytes": 1, "primary_sources_verified": 1,
                                              "original_source_objects_verified": 1})
                else:
                    evidence = json.loads((path.parent / "evidence.json").read_text(encoding="utf-8-sig"))
                    evidence["_article_source_links"] = re.findall(r"\]\((https?://[^)\s]+)\)", body)
                    verify_article_sources(cache, metadata, evidence, source_statistics)
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append({"file": relative, "problem": "Source provenance verification failed", "detail": str(error)})
            source_caches += 1

    integrity = read("data/article-verification.json")
    result = {"status": "failed" if issues else "passed", "documents_checked": len(documents),
              "complete_introductions": len(articles), "model_series": len(list(ROOT.glob("papers/*/README.md"))),
              "original_assets": integrity["original_assets"], "derived_assets": integrity["derived_assets"],
              "local_links_and_anchors_checked": links_checked,
              "chronological_navigation": chronological_navigation, "source_caches_checked": source_caches,
              "source_provenance_statistics": dict(source_statistics),
              "issues": issues, "limits": "Structural, link, delimiter and provenance checks; scientific claims need source review. External URL availability is not inferred from local links."
              + (" Source checks prove cached byte identity and declared source roles. A release snapshot label without its own dated cache record is bound by matching metadata/evidence SHA, not independently attested as a capture date." if args.sources else "")}
    write("data/repository-quality.json", json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(issues))


if __name__ == "__main__":
    main()
