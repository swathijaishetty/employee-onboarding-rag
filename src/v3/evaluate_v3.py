"""Measure retrieval hit rate and reciprocal rank on a small regression set."""

import json
from pathlib import Path

from .retrieval_v3 import retrieve


def evaluate(path: str | Path | None = None) -> dict:
    cases_path = Path(path or Path(__file__).with_name("evaluation_cases.json"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    hits = 0
    reciprocal_ranks: list[float] = []
    for case in cases:
        results = retrieve(case["question"])
        sources = [item.source for item in results]
        expected = case["expected_sources"]
        source_match = all(source in sources for source in expected) if expected else not results
        section_match = all(
            any(
                item.source == requirement["source"]
                and requirement["contains"].lower() in item.section.lower()
                for item in results
            )
            for requirement in case.get("expected_sections", [])
        )
        found = source_match and section_match
        hits += int(found)
        ranks = [sources.index(source) + 1 for source in expected if source in sources]
        reciprocal_ranks.append(1 / min(ranks) if ranks else (1.0 if found else 0.0))
        print(("PASS" if found else "FAIL") + f" | {case['question']}\n  retrieved: {sources}")
    report = {
        "cases": len(cases),
        "all_expected_source_hit_rate": hits / len(cases),
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(cases),
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    evaluate()
