"""Real CLI before/after replay of explicitly synthetic chart layouts.

No Trainspotter imports: histories with separate metric columns supply the
diagnostic control, while counts and expected names come from fixture inputs.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expect", choices=["before", "after"], required=True)
    args = parser.parse_args()
    args.python = args.python.absolute()  # Preserve the venv interpreter path.
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if args.source:
        environment["PYTHONPATH"] = str(args.source.resolve())
    else:
        environment.pop("PYTHONPATH", None)
    layouts = {
        "ambiguous": ("runA", "train - loss", "eval - loss"),
        "resumed": ("experiment - resumed", "train/loss", "eval/loss"),
        "literal": ('<run & "name">', "train - loss", "eval - loss"),
        "ordinary": ("runA", "train/loss", "eval/loss"),
    }
    train = [2 - step * .02 for step in range(50)]
    evaluation = [.5 + step * .02 for step in range(50)]

    def write(name, rows):
        with (args.artifacts / name).open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)

    write("history.csv", [["_step", "trainer/global_step", "train/loss", "eval/loss"],
                         *[[i, i, a, b] for i, (a, b) in enumerate(zip(train, evaluation))]])
    for name, (run, training_key, evaluation_key) in layouts.items():
        columns = [run + " - " + training_key, run + " - " + evaluation_key]
        values = [train, evaluation]
        if name == "ordinary":
            columns.append("other - loss")
            values.append([1] * 50)
        write(name + ".csv", [["trainer/global_step", *[
            column + suffix for column in columns for suffix in ("", "__MIN", "__MAX")
        ]], *[[i, *[series[i] for series in values for _ in range(3)]] for i in range(50)]])

    def invoke(name, options=(), output="json", command="analyze"):
        flags = ["--once"] if command == "watch" else ["--output", output]
        if output == "html":
            flags += ["--out", name + ".html", "--title", "Synthetic chart identity example"]
        process = subprocess.run(
            [str(args.python), "-m", "trainspotter.cli", command, name + ".csv",
             *options, *flags], cwd=args.artifacts, env=environment,
            capture_output=True, text=True, timeout=30,
        )
        return {"status": process.returncode, "stdout": process.stdout, "stderr": process.stderr}

    control = invoke("history")
    assert control["status"] == 0, control
    history = json.loads(control["stdout"])
    assert [f["detector"] for f in history["findings"]] == ["overfitting"]
    records = []
    for name, (run, _, _) in layouts.items():
        automatic = invoke(name)
        if args.expect == "after" and name != "ordinary":
            assert automatic["status"] == 2, automatic
            assert "ambiguous W&B chart label" in automatic["stderr"]
            assert automatic["stdout"] == ""
        else:
            assert automatic["status"] == 0, automatic
        options = ["--chart-single-run", run] if name != "ordinary" else ["--run", "runA"]
        result = invoke(name, options)
        if args.expect == "before" and name != "ordinary":
            assert result["status"] == 2
            assert "unrecognized arguments" in result["stderr"]
            result = automatic
            options = []
        assert result["status"] == 0, result
        report = json.loads(result["stdout"])
        if args.expect == "after" or name in ("resumed", "ordinary"):
            assert report["findings"] == history["findings"]
            assert report["run"]["metrics"] == history["run"]["metrics"]
        else:
            assert set(report["run"]["metrics"]) == {"train/loss"}
            assert report["findings"] == []
        (args.artifacts / (name + ".json")).write_text(result["stdout"])
        html = invoke(name, options, output="html")
        assert html["status"] == 0, html
        watch = invoke(name, options, command="watch")
        assert watch["status"] == 0, watch
        records.append({
            "name": name, "options": options, "automatic": automatic,
            "report": report, "watch": watch,
            "htmlSha256": hashlib.sha256((args.artifacts / (name + ".html")).read_bytes()).hexdigest(),
        })
    if args.expect == "after":
        wrong_layout = invoke("ordinary", ["--chart-single-run", "runA"])
        assert wrong_layout["status"] == 2
        assert "other - loss" in wrong_layout["stderr"]
        assert wrong_layout["stdout"] == ""
    else:
        wrong_layout = None
    selected_eval = invoke("ambiguous", ["--run", "runA - eval"])
    if args.expect == "before":
        selected = json.loads(selected_eval["stdout"])
        assert set(selected["run"]["metrics"]) == {"train/loss"}
    else:
        assert selected_eval["status"] == 2
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.artifacts.glob("*.csv"))}
    args.out.write_text(json.dumps({
        "expectation": args.expect, "provenance": "synthetic; not hosted W&B exports",
        "inputSha256": inputs, "historyControl": history, "records": records,
        "incorrectSingleRunDeclaration": wrong_layout, "selectApparentEvaluationRun": selected_eval,
    }, indent=2) + "\n")
    print(json.dumps({"expectation": args.expect, "charts": len(records), "historyRows": 50,
                      "bothCurvesRetained": sum(len(r["report"]["run"]["metrics"]) == 2
                                                for r in records)}))


if __name__ == "__main__":
    main()
