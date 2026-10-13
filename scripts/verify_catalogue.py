"""Check source coverage, star decisions, lineage and local report links."""
import json
import re
from collections import Counter
from pathlib import Path
from supplemental_catalogue import load_supplemental_entries

ROOT = Path(__file__).resolve().parents[1]
catalogue = json.loads((ROOT / "data/catalogue.json").read_text(encoding="utf-8"))
entries = catalogue["entries"]
snapshot = catalogue["metadata"]
stats = catalogue["statistics"]
checks = {}

assert len(entries) == snapshot["unique_entries"]
assert sum(len(e["source_sections"]) for e in entries) == snapshot["raw_rows"]
checks["complete_source_coverage"] = True
assert len({e["id"] for e in entries}) == len(entries)
checks["unique_document_ids"] = True
supplemental = load_supplemental_entries(entries, ROOT)
checks["official_supplemental_entries_validated"] = len(supplemental)

threshold = snapshot["threshold"]
for entry in entries:
    observation = entry["star_observation"]
    if not entry["repo"]:
        assert entry["star_decision"] == "not_assessable"
    elif observation["stars"] < threshold:
        assert entry["star_decision"] == "excluded_low_star"
        assert entry["catalogue_group"] != "model_catalogue"
    else:
        assert entry["star_decision"] == "passes_star"
checks["exact_star_decisions_and_no_zero_for_missing"] = True
assert Counter(e["star_decision"] for e in entries) == stats["star_decisions"]
checks["statistics_match_entries"] = True

for entry in entries:
    if entry["title"].startswith(("GPT4RoI", "GPT4Tools")):
        assert entry["family"] != "GPT" and entry["organization"] != "OpenAI"
checks["third_party_gpt_names_separate"] = True
for entry in entries:
    if entry["repo"] and entry["canonical_repo"].casefold() != entry["repo"].casefold():
        assert any("redirects" in warning for warning in entry["warnings"])
checks["repository_redirects_labelled"] = True

broken = []
for document in [ROOT / "README.md", *ROOT.glob("papers/*/README.md"), *ROOT.glob("reports/*.md")]:
    for destination in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
        if "://" in destination or destination.startswith("#"):
            continue
        target = (document.parent / destination.split("#")[0]).resolve()
        if not target.exists():
            broken.append({"file": str(document.relative_to(ROOT)), "target": destination})
assert not broken, broken
checks["all_local_markdown_links_resolve"] = True

audit_path = ROOT / "data/foundation-star-audit.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
observations = audit.get("repositories", audit) if isinstance(audit, dict) else audit
if isinstance(observations, dict):
    verified_count = 0
    stars = json.loads((ROOT / "data/stars.json").read_text(encoding="utf-8"))
    for repo, observation in observations.items():
        if repo in stars and isinstance(observation, dict):
            independent = observation.get("exact_stars", observation.get("stars", observation.get("stargazers_count")))
            if independent is not None:
                assert stars[repo]["stars"] == independent
                verified_count += 1
    checks["independent_foundation_repositories_matched"] = verified_count

result = {"status": "passed", "snapshot_date": snapshot["snapshot_date"], "checks": checks,
          "entries": len(entries), "main_model_entries": stats["main_model_entries"], "model_families": stats["main_families"],
          "supplemental_entries": len(supplemental), "collection_entries": len(entries) + len(supplemental),
          "collection_retained_entries": sum(entry["catalogue_group"] != "excluded_low_star" for entry in entries) + len(supplemental)}
(ROOT / "data/validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
