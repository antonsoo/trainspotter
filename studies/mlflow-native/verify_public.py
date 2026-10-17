"""Compare every public sample HTML byte, including the canonical root alias."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default="https://antonsoo.github.io/trainspotter/")
    args = parser.parse_args()
    expected = [(path.name, path.read_bytes()) for path in sorted(args.build.glob("*.html"))]
    expected.append(("", (args.build / "index.html").read_bytes()))

    def check(item):
        name, wanted = item
        url = args.base + name
        try:
            with urlopen(
                Request(url, headers={"Cache-Control": "no-cache"}), timeout=20
            ) as response:
                data = response.read()
                return {
                    "file": name or "(canonical root)",
                    "url": url,
                    "status": response.status,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "expectedSha256": hashlib.sha256(wanted).hexdigest(),
                    "matches": data == wanted,
                    "etag": response.headers.get("etag"),
                    "age": response.headers.get("age"),
                }
        except HTTPError as exc:
            return {
                "file": name or "(canonical root)",
                "url": url,
                "status": exc.code,
                "matches": False,
            }

    deadline = time.monotonic() + 120
    attempts = []
    while True:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(check, expected))
        matched = sum(row["matches"] for row in rows)
        attempts.append({"matching": matched, "total": len(rows)})
        print(json.dumps(attempts[-1]), flush=True)
        if matched == len(rows) or time.monotonic() > deadline:
            break
        time.sleep(5)
    receipt = {
        "base": args.base,
        "scope": (
            "Every generated public HTML file, plus canonical root alias; "
            "exact bytes, no generated-time normalization."
        ),
        "attempts": attempts,
        "files": rows,
        "complete": all(row["matches"] for row in rows),
    }
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["complete"], [r for r in rows if not r["matches"]]


if __name__ == "__main__":
    main()
