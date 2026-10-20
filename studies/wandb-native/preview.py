"""Inventory native reports for the existing offline SVG browser workflow."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    reports = Path(sys.argv[1])
    captures = Path(sys.argv[2])
    cases = []
    names = [
        "stable",
        "unstable-lr",
        "logged-null",
        "custom-counter",
        "custom-selected",
        "truncated-fragment",
        "corrupt-fragment",
        "truncated-tail",
        "types",
    ]
    for name in names:
        document = (reports / (name + ".html")).read_bytes()
        exports = []
        for encoded in re.findall(rb'href="data:image/svg\+xml;base64,([^"]+)"', document):
            data = base64.b64decode(encoded, validate=True)
            root = ET.fromstring(data)
            metadata = json.loads(root.find("{http://www.w3.org/2000/svg}metadata").text)
            source = (
                captures / ("custom-counter" if name == "custom-selected" else name) / "run.wandb"
                if name in names[:5] or name == "types"
                else reports / (name + ".wandb")
            )
            assert (
                metadata["nativeWandb"]["fileSHA256"]
                == hashlib.sha256(source.read_bytes()).hexdigest()
            )
            exports.append(
                {
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "metadata": metadata,
                }
            )
        assert exports
        assert all(
            item["metadata"]["inputComplete"] == exports[0]["metadata"]["inputComplete"]
            for item in exports
        )
        cases.append(
            {
                "name": "native-" + name,
                "file": name + ".html",
                "reportSha256": hashlib.sha256(document).hexdigest(),
                "exports": exports,
                "inputComplete": exports[0]["metadata"]["inputComplete"],
            }
        )
    (reports / "capture.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(len(cases), "reports,", sum(len(case["exports"]) for case in cases), "vectors")


if __name__ == "__main__":
    main()
