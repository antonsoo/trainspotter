"""Classify every change in verify_loss_corpus.py's retained JSONL output."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    changed = []
    total = readable = unchanged = unreadable = 0
    input_hash = hashlib.sha256()
    with (
        (args.artifacts / "before.jsonl").open() as before,
        (args.artifacts / "after.jsonl").open() as after,
    ):
        for old_line, new_line in zip(before, after, strict=True):
            old, new = json.loads(old_line), json.loads(new_line)
            assert old["identity"] == new["identity"]
            assert old.get("sha256") == new.get("sha256")
            input_hash.update(f"{old['identity']}\0{old.get('sha256')}\n".encode())
            total += 1
            readable += "report" in old
            unreadable += "unreadable" in old
            if old == new:
                unchanged += 1
                continue
            normalized = copy.deepcopy(new)
            rows = []
            old_findings = old["report"]["findings"]
            new_findings = normalized["report"]["findings"]
            for a, b in zip(old_findings, new_findings, strict=True):
                if a == b:
                    continue
                assert a["detector"] == b["detector"] == "divergence"
                assert a["title"] in (
                    "Diverged (non-finite values)", "Non-finite evaluation values",
                )
                assert b["title"] == (
                    "Non-finite training loss" if a["title"] == "Diverged (non-finite values)"
                    else "Non-finite evaluation values"
                )
                assert a["evidence"]["count"] > 0
                rows.append({
                    "metric": a["metric"], "count": a["evidence"]["count"],
                    "before_title": a["title"], "after_title": b["title"],
                    "suggestions_changed": a["fixes"] != b["fixes"],
                })
                b["title"], b["fixes"] = a["title"], a["fixes"]
            # All report values, trajectories, summaries and input states must
            # match exactly after accounting for these two wording fields.
            assert normalized == old, old["identity"]
            changed.append({"identity": old["identity"], "input_sha256": old["sha256"],
                            "changes": rows})
    result = {
        "compared": total, "readable": readable, "unreadable": unreadable,
        "exactly_equal": unchanged, "wording_only_changes": len(changed),
        "trajectory_changes": 0, "input_state_changes": 0, "severity_changes": 0,
        "finding_count_changes": 0, "input_identity_sha256_aggregate": input_hash.hexdigest(),
        "changed": changed,
        "scope": "Cached historical corpus, not an accuracy label or new live Hub census.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "changed"}, indent=2))


if __name__ == "__main__":
    main()
