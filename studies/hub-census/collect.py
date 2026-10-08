"""Collect public Trainer logs from the Hugging Face Hub.

    uv run python studies/hub-census/collect.py                    # the manifest's files
    uv run python studies/hub-census/collect.py --rescan --models 60000

`transformers.Trainer` writes its whole logging history to `trainer_state.json`
in every checkpoint directory, and many people upload those directories with
the model. This walks the Hub's model listing from the newest model backwards,
keeps each repository that contains such a file, and downloads the one from its
last checkpoint (the longest history).

Nothing is run and no model weights are fetched: one small JSON file per
repository, at the commit the listing reported. The manifest records exactly
which files were read; with a manifest present and no `--rescan`, this fetches
those files again so the census can be repeated on the same inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API = "https://huggingface.co/api/models?full=true&limit=1000&sort=createdAt&direction=-1"
HERE = Path(__file__).parent
MAX_BYTES = 30 * 1024 * 1024
CHECKPOINT = re.compile(r"checkpoint-(\d+)/trainer_state\.json$")


def _open(url: str, attempts: int = 5) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "trainspotter-hub-census"})
    delay = 2.0
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=90)
        except urllib.error.HTTPError as exc:
            # Gated, private or removed since the listing: not ours to read.
            if exc.code in (401, 403, 404, 451):
                raise
            if attempt == attempts - 1:
                raise
        except OSError:
            if attempt == attempts - 1:
                raise
        time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable")


def last_checkpoint(files: list[str]) -> str | None:
    """The trainer_state.json with the longest history: the highest-numbered
    checkpoint in the shallowest directory, else a top-level file."""
    # The exact name: "._trainer_state.json" is a macOS resource fork, not a log.
    states = [f for f in files if f.rsplit("/", 1)[-1] == "trainer_state.json"]
    if not states:
        return None
    depth = min(f.count("/") for f in states)
    shallow = [f for f in states if f.count("/") == depth]
    numbered = [(int(m.group(1)), f) for f in shallow if (m := CHECKPOINT.search(f))]
    return max(numbered)[1] if numbered else sorted(shallow)[0]


def scan(models: int, per_author: int, state: Path) -> list[dict[str, Any]]:
    """Walk the listing, newest first, and return one manifest row per kept repository."""
    progress = (
        json.loads(state.read_text()) if state.exists() else {"scanned": 0, "url": API, "rows": []}
    )
    counts: dict[str, int] = {}
    for row in progress["rows"]:
        counts[row["author"]] = counts.get(row["author"], 0) + 1
    while progress["scanned"] < models and progress["url"]:
        with _open(progress["url"]) as response:
            page = json.load(response)
            link = response.headers.get("Link") or ""
        for model in page:
            path = last_checkpoint([s["rfilename"] for s in model.get("siblings") or []])
            author = model["id"].split("/")[0]
            if path is None or model.get("private") or model.get("gated"):
                continue
            # One prolific account must not become the census.
            if counts.get(author, 0) >= per_author:
                continue
            counts[author] = counts.get(author, 0) + 1
            progress["rows"].append(
                {
                    "repo": model["id"],
                    "author": author,
                    "revision": model.get("sha"),
                    "path": path,
                    "created": model.get("createdAt"),
                    "library": model.get("library_name"),
                    "task": model.get("pipeline_tag"),
                    "adapter": "peft" in (model.get("tags") or [])
                    or model.get("library_name") == "peft",
                }
            )
        progress["scanned"] += len(page)
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        progress["url"] = match.group(1) if match else None
        state.write_text(json.dumps(progress))
        print(
            f"scanned {progress['scanned']:>6} models, kept {len(progress['rows'])} logs",
            flush=True,
        )
    return list(progress["rows"])


def fetch(row: dict[str, Any], cache: Path) -> str:
    target = cache / (
        hashlib.sha1(f"{row['repo']}@{row['revision']}:{row['path']}".encode()).hexdigest()
        + ".json"
    )
    if target.exists():
        return "cached"
    revision = row["revision"] or "main"
    url = (
        f"https://huggingface.co/{row['repo']}/resolve/{revision}/{urllib.parse.quote(row['path'])}"
    )
    try:
        with _open(url) as response:
            body = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return f"http {exc.code}"
    except OSError as exc:
        return f"error {type(exc).__name__}"
    if len(body) > MAX_BYTES:
        return "too large"
    target.write_bytes(body)
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--models", type=int, default=60000, help="how many of the newest models to look at"
    )
    parser.add_argument(
        "--per-author", type=int, default=3, help="most repositories kept per account"
    )
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.json")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="walk the Hub listing again and write a new manifest (default: fetch its files)",
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    if args.manifest.exists() and not args.rescan:
        # Reproduce: the same files, each at the commit recorded when it was first read.
        rows = json.loads(args.manifest.read_text())["logs"]
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            outcomes = list(pool.map(lambda row: fetch(row, args.cache), rows))
        tally = {outcome: outcomes.count(outcome) for outcome in sorted(set(outcomes))}
        print(f"{len(rows)} files in the manifest: {tally}")
        return

    rows = scan(args.models, args.per_author, args.cache / "scan-state.json")
    rows = [row for row in rows if row["path"].rsplit("/", 1)[-1] == "trainer_state.json"]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        outcomes = list(pool.map(lambda row: fetch(row, args.cache), rows))
    kept = [row for row, outcome in zip(rows, outcomes, strict=True) if outcome in ("ok", "cached")]
    tally: dict[str, int] = {}
    for outcome in outcomes:
        tally[outcome] = tally.get(outcome, 0) + 1
    scanned = json.loads((args.cache / "scan-state.json").read_text())["scanned"]
    header = {
        "listing": API,
        "models_scanned": scanned,
        "newest": max((r["created"] for r in rows), default=None),
        "oldest": min((r["created"] for r in rows), default=None),
        "per_author_cap": args.per_author,
        # "cached" counts a file already present from an earlier run of this script.
        "downloads": tally,
    }
    keep = ("repo", "revision", "path", "created", "library", "task", "adapter")
    # One log per line, so a later collection diffs row by row.
    lines = ",\n".join(json.dumps({k: r[k] for k in keep}, separators=(",", ":")) for r in kept)
    args.manifest.write_text(json.dumps(header)[:-1] + ', "logs": [\n' + lines + "\n]}\n")
    print(f"{len(kept)} logs from {scanned} models; downloads: {tally}; wrote {args.manifest}")


if __name__ == "__main__":
    main()
