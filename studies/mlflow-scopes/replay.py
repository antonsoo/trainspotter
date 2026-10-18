"""Compare actual installed CLIs and readers with native SDK run histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = "studies/mlflow-scopes/native/scopes.sqlite"
SNAPSHOT = """
import json, sys
from trainspotter.readers import load_run
run = load_run(sys.argv[1], fmt='mlflow', run_name=sys.argv[2], step_key='step')
print(json.dumps({run.meta['metric_sources'].get(key, key): [
    [p.step, p.value, round(p.wall_time * 1000)] for p in series.points
] for key, series in run.metrics.items()}, allow_nan=False))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    python = str(args.python.absolute())
    cli = [python, "-c", "from trainspotter.cli import main; raise SystemExit(main())"]
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    oracle = json.loads((ROOT / "studies/mlflow-scopes/native/oracle.json").read_text())
    assert hashlib.sha256((ROOT / INPUT).read_bytes()).hexdigest() == oracle["database"]["sha256"]
    results = []
    for case in oracle["cases"]:
        common = [INPUT, "--format", "mlflow", "--run", case["run_id"], "--step-key", "step"]
        target = output / (case["name"] + ".json")
        process = subprocess.run(
            [*cli, "analyze", *common, "--output", "json", "--out", str(target)],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert process.returncode == 0, process.stderr
        report = json.loads(target.read_text())
        actual = json.loads(
            subprocess.check_output(
                [python, "-c", SNAPSHOT, INPUT, case["run_id"]], cwd=ROOT, text=True
            )
        )
        expected = {
            key: [[p["step"], p["value"], p["timestamp"]] for p in points]
            for key, points in case["sdk_run_histories"].items()
        }
        assert {key: sorted(points) for key, points in actual.items()} == {
            key: sorted(points) for key, points in expected.items()
        }, case["name"]
        gates = {}
        for command in ["analyze", "watch"]:
            gate = subprocess.run(
                [
                    *cli,
                    command,
                    *common,
                    "--fail-on",
                    "error",
                    "--no-color",
                    *(["--once"] if command == "watch" else []),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            gates[command] = gate.returncode
        if args.phase == "after":
            complete = case["name"] == "unqualified"
            assert report["run"]["input_complete"] == complete
            assert report["findings"] == []
            assert set(gates.values()) == ({0} if complete else {2})
        if case["name"] in ["mixed-dataset", "coalesced-coordinate"]:
            html = output / (case["name"] + ".html")
            rendered = subprocess.run(
                [*cli, "analyze", *common, "--output", "html", "--out", str(html)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            assert rendered.returncode == 0, rendered.stderr
        results.append(
            {
                "case": case["name"],
                "runId": case["run_id"],
                "recordedRunObservations": sum(map(len, actual.values())),
                "separateModelRows": sum(g["recorded_rows"] for g in case["sql_model_scopes"]),
                "allObservationsMatchSdk": True,
                "inputComplete": report["run"]["input_complete"],
                "diagnosticMetrics": report["run"]["metric_sources"],
                "findings": report["findings"],
                "gateExits": gates,
            }
        )
    receipt = {
        "phase": args.phase,
        "scope": (
            "Actual CLI reports and gates plus every decoded observation from the selected "
            "installed reader, compared with native MLflow 3.17.1 SDK histories. Authored "
            "logging controls, no training or production reliability claim."
        ),
        "database": oracle["database"],
        "cases": results,
    }
    (output / "replay.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "phase": args.phase,
                "runs": len(results),
                "sdkObservationsMatched": sum(r["recordedRunObservations"] for r in results),
                "findingCounts": {r["case"]: len(r["findings"]) for r in results},
            }
        )
    )


if __name__ == "__main__":
    main()
