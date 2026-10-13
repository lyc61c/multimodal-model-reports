"""Load official additions without altering the original upstream screening snapshot."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "data/supplemental-catalogue.json"


def https_url(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an HTTPS URL")
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{label} must be an HTTPS URL")
    return value


def iso_date(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{label} must be YYYY-MM-DD")
    return date.fromisoformat(value)


def document_identity(url):
    """Identify arXiv aliases as one document; otherwise retain meaningful URL queries."""
    parsed = urllib.parse.urlsplit(url or "")
    arxiv = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d+)", url or "")
    if arxiv:
        return "arxiv:" + arxiv[1]
    return urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(),
                                   parsed.path.rstrip("/"), parsed.query, ""))


def load_supplemental_entries(upstream_entries=(), root=ROOT):
    path = Path(root) / REGISTRY
    if not path.exists():
        return []
    registry = json.loads(path.read_text(encoding="utf-8-sig"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("entries"), list):
        raise ValueError("Supplemental catalogue needs schema_version=1 and entries[]")
    cutoff = iso_date(registry.get("as_of_date"), "Supplemental as_of_date")
    upstream = list(upstream_entries)
    ids = {entry["id"] for entry in upstream}
    identities = {entry.get("identity") for entry in upstream}
    documents = {document_identity(entry.get("paper_url")) for entry in upstream if entry.get("paper_url")}
    # A family slug must keep its established display name across both collections.
    family_names = {entry["family_slug"]: entry["family"] for entry in upstream if entry.get("family_slug")}
    result = []
    for supplied in registry["entries"]:
        if not isinstance(supplied, dict):
            raise ValueError("Supplemental entries must be objects")
        entry = dict(supplied)
        for key in ("id", "identity", "title", "family", "family_slug", "organization", "source_version"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"Supplemental entry lacks {key}")
        expected_id = hashlib.sha256(entry["identity"].encode("utf-8")).hexdigest()[:12]
        if entry["id"] != expected_id:
            raise ValueError(f"Supplemental id must be SHA256(identity)[:12]: {entry['title']}")
        https_url(entry.get("paper_url"), "paper_url")
        https_url(entry.get("source_url"), "source_url")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", entry["family_slug"]):
            raise ValueError("Invalid supplemental family_slug")
        if entry["family_slug"] in family_names and family_names[entry["family_slug"]] != entry["family"]:
            raise ValueError(f"Family name differs for existing slug: {entry['family_slug']}")
        published = iso_date(entry.get("date"), "Supplemental date")
        if published > cutoff:
            raise ValueError(f"Publication is later than supplemental cutoff: {entry['title']}")
        provenance = entry.get("provenance", {})
        if provenance.get("origin") != "official_supplement":
            raise ValueError("Supplemental provenance.origin must be official_supplement")
        discovered = iso_date(provenance.get("discovered_at"), "provenance.discovered_at")
        if discovered > cutoff:
            raise ValueError("Supplemental discovery date is later than cutoff")
        https_url(provenance.get("official_source_url"), "provenance.official_source_url")
        evidence = provenance.get("date_evidence", {})
        https_url(evidence.get("url"), "provenance.date_evidence.url")
        if not isinstance(evidence.get("locator"), str) or not evidence["locator"].strip():
            raise ValueError("Supplemental publication date requires a source locator")
        if entry.get("source_sections") or entry.get("source_lines"):
            raise ValueError("Official additions must not claim rows in the original upstream snapshot")
        scope = entry.get("scope", {})
        if scope.get("kind") not in {"model_paper", "technical_report", "model_card", "system_card", "release_blog"}:
            raise ValueError("Supplemental document kind must identify an official model document")
        recommendation = scope.get("recommend_scope")
        expected_group = {"include_candidate": "model_catalogue", "supplementary": "supporting_materials"}.get(recommendation)
        if not expected_group or entry.get("catalogue_group") != expected_group:
            raise ValueError("Supplemental catalogue_group differs from scope decision")
        if not isinstance(scope.get("note"), str) or not scope["note"].strip():
            raise ValueError("Supplemental scope decision needs a note")
        repo = entry.get("repo")
        if repo:
            if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo):
                raise ValueError("Supplemental repo must be owner/name")
            github = urllib.parse.urlsplit(https_url(entry.get("github_code_url"), "github_code_url"))
            if github.netloc.lower() != "github.com" or "/".join(github.path.strip("/").split("/")[:2]).removesuffix(".git").casefold() != repo.casefold():
                raise ValueError("Supplemental GitHub URL does not identify repo")
            observation = entry.get("star_observation") or {}
            stars = observation.get("stars")
            if isinstance(stars, bool) or not isinstance(stars, int) or stars < 100 or entry.get("star_decision") != "passes_star":
                raise ValueError("A supplemental repository needs a verified integer Star count >=100")
            https_url(observation.get("source_url"), "star_observation.source_url")
            try:
                observed = datetime.fromisoformat(observation["observed_at_utc"].replace("Z", "+00:00"))
                if observed.tzinfo is None:
                    raise ValueError("Timezone required")
            except (KeyError, AttributeError, ValueError) as error:
                raise ValueError("Star observation needs an ISO timestamp with timezone") from error
            entry.setdefault("canonical_repo", repo)
            if not isinstance(entry["canonical_repo"], str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", entry["canonical_repo"]):
                raise ValueError("Invalid supplemental canonical_repo")
            if entry["canonical_repo"].casefold() != repo.casefold() and not any("redirects" in warning for warning in entry.get("warnings", [])):
                raise ValueError("Supplemental repository redirects must be labelled")
        elif (entry.get("star_decision") != "not_assessable" or entry.get("star_observation") is not None
              or entry.get("github_code_url") is not None or entry.get("canonical_repo") is not None):
            raise ValueError("Missing repo must remain not_assessable, with null Star and GitHub fields")
        identity = document_identity(entry["paper_url"])
        if entry["id"] in ids or entry["identity"] in identities or identity in documents:
            raise ValueError(f"Supplemental document duplicates an existing entry: {entry['title']}")
        if entry.get("destination"):
            destination = Path(entry["destination"])
            if destination.is_absolute() or len(destination.parts) != 3 or destination.parts[:2] != ("papers", entry["family_slug"]) or ".." in destination.parts:
                raise ValueError("Supplemental destination must be papers/<family_slug>/<document_slug>")
        entry.setdefault("repo", None)
        entry.setdefault("github_code_url", None)
        entry.setdefault("star_observation", None)
        entry.setdefault("canonical_repo", None)
        entry.setdefault("venue", "Blog" if scope["kind"] == "release_blog" else "Technical Report")
        entry.setdefault("code_links", [])
        entry.setdefault("badge_repo", None)
        entry.setdefault("source_sections", [])
        entry.setdefault("source_lines", [])
        entry.setdefault("warnings", [])
        result.append(entry)
        ids.add(entry["id"])
        identities.add(entry["identity"])
        documents.add(identity)
        family_names[entry["family_slug"]] = entry["family"]
    return result
