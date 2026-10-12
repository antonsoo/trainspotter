"""Compare retained CLI controls and exact cached public-log reports, without tolerances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--corpus-before", type=Path, required=True)
    parser.add_argument("--corpus-after", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())
    cases = []
    for a, b in zip(before["records"], after["records"], strict=True):
        assert a["name"] == b["name"]
        assert a["inputSha256"] == b["inputSha256"]
        assert a["expectedUnreadableSteps"] == b["expectedUnreadableSteps"]
        assert a["exitCode"] == b["exitCode"]
        equal = a["report"] == b["report"]
        if a["name"] in ("observed-nan-control", "finite-control"):
            assert equal, a["name"]
        else:
            assert not equal, a["name"]
        cases.append({
            "name": a["name"], "inputSha256": a["inputSha256"],
            "fullReportExactlyEqual": equal, "exitCodeUnchanged": a["exitCode"],
            "before": [(f["title"], f["step_start"], f["step_end"])
                       for f in a["report"]["findings"]],
            "after": [(f["title"], f["step_start"], f["step_end"])
                      for f in b["report"]["findings"]],
        })
    compared = readable = unreadable = unavailable = exact = 0
    changed = []
    identities = hashlib.sha256()
    with args.corpus_before.open() as old, args.corpus_after.open() as new:
        for old_line, new_line in zip(old, new, strict=True):
            a, b = json.loads(old_line), json.loads(new_line)
            assert a["identity"] == b["identity"]
            assert a.get("sha256") == b.get("sha256")
            identities.update(f"{a['identity']}\0{a.get('sha256', 'unavailable')}\n".encode())
            compared += 1
            readable += "report" in a
            unreadable += "unreadable" in a
            unavailable += "unavailable" in a
            if a == b:
                exact += 1
            else:
                changed.append({"identity": a["identity"], "inputSha256": a.get("sha256")})
    result = {
        "comparison": "Exact decoded JSON equality; no numeric tolerances or field exclusions.",
        "cliCases": cases,
        "corpus": {
            "compared": compared, "readable": readable, "unreadable": unreadable,
            "unavailable": unavailable, "exactlyEqual": exact, "changed": changed,
            "inputIdentitySha256Aggregate": identities.hexdigest(),
            "beforeSha256": digest(args.corpus_before), "afterSha256": digest(args.corpus_after),
            "sourcePathConvention": "Existing worker replaces the cache directory with a basename.",
            "scope": "Cached historical public logs; no new downloads or accuracy calibration.",
        },
        "replaySha256": {"before": digest(args.before), "after": digest(args.after)},
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["corpus"], indent=2))
    if changed:
        raise SystemExit("Inspect every changed corpus report before accepting this comparison.")


if __name__ == "__main__":
    main()
