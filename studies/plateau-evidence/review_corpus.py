"""Account for every changed report; preserve unknown inputs and other detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            state.update(chunk)
    return state.hexdigest()


def compact(finding: dict) -> dict:
    omitted = ("fixes", "detector", "metric", "severity")
    return {k: v for k, v in finding.items() if k not in omitted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    counts = {"compared": 0, "readable": 0, "unreadable": 0, "unavailable": 0,
              "exactly_equal": 0, "old_plateaus": 0, "new_plateaus": 0}
    changed = []
    aggregate = hashlib.sha256()
    with args.before.open() as before, args.after.open() as after:
        for old_line, new_line in zip(before, after, strict=True):
            old, new = json.loads(old_line), json.loads(new_line)
            assert old["identity"] == new["identity"]
            assert old.get("sha256") == new.get("sha256")
            aggregate.update(f"{old['identity']}\0{old.get('sha256')}\n".encode())
            counts["compared"] += 1
            for key in ("unreadable", "unavailable"):
                counts[key] += key in old
            if "report" not in old:
                assert old == new
                counts["exactly_equal"] += 1
                continue
            counts["readable"] += 1
            a, b = old["report"], new["report"]
            assert {k: v for k, v in a.items() if k not in ("findings", "summary")} == {
                k: v for k, v in b.items() if k not in ("findings", "summary")
            }
            assert [f for f in a["findings"] if f["detector"] != "plateau"] == [
                f for f in b["findings"] if f["detector"] != "plateau"
            ], old["identity"]
            old_plateau = [f for f in a["findings"] if f["detector"] == "plateau"]
            new_plateau = [f for f in b["findings"] if f["detector"] == "plateau"]
            counts["old_plateaus"] += len(old_plateau)
            counts["new_plateaus"] += len(new_plateau)
            for finding in new_plateau:
                assert finding["severity"] == "info"
                e = finding["evidence"]
                low, high = e["level_min"], e["level_max"]
                bound = max(e["rel_change_threshold"] * max(abs(low), abs(high), 1e-12),
                            e["noise_allowance"])
                assert high - low < bound, (old["identity"], finding)
            if old == new:
                counts["exactly_equal"] += 1
            else:
                changed.append({"identity": old["identity"], "input_sha256": old["sha256"],
                                "before": [compact(f) for f in old_plateau],
                                "after": [compact(f) for f in new_plateau]})
    assert counts["compared"] == 4259
    result = {
        **counts, "changed_reports": len(changed),
        "before_jsonl_sha256": digest(args.before), "after_jsonl_sha256": digest(args.after),
        "input_identity_sha256_aggregate": aggregate.hexdigest(),
        "other_findings_exactly_equal": True, "input_and_timing_evidence_exactly_equal": True,
        "warning_and_error_gates_unchanged": True, "reported_level_bounds_satisfied": True,
        "scope": "Historical cached corpus compatibility, not a ground-truth accuracy estimate.",
        "changed": changed,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "changed"}, indent=2))


if __name__ == "__main__":
    main()
