"""Compare every native scalar/clock with independent SDK and Go decoders."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import trainspotter
from trainspotter import load_run
from trainspotter.cli import main as cli
from trainspotter.detectors import run_all
from trainspotter.report import render_html, render_terminal
from trainspotter.report.json_report import to_json_dict

NAMES = {
    "train/loss": "train/loss",
    "eval/loss": "eval/loss",
    "train/learning_rate": "lr",
    "train/grad_norm": "grad_norm",
    "epoch": "epoch",
    "optimizer_step": "optimizer_step",
    "train/global_step": "train/global_step",
}


def safe(value: object) -> object:
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def packets(rows: list[dict[str, Any]], axis: str) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        values = row["items"]
        step = int(values[axis])
        assert step == values[axis]
        clock = float(values["_runtime"])
        for key, name in NAMES.items():
            if key == axis or key not in values:
                continue
            value = None if values[key] is None else float(values[key])
            result.setdefault(name, []).append(
                {"step": step, "value": safe(value), "wall_time": clock}
            )
    for points in result.values():
        points.sort(key=lambda p: p["step"])
    return result


def observed(run: Any) -> dict[str, list[dict[str, object]]]:
    return {
        name: [
            {"step": p.step, "value": safe(p.value), "wall_time": p.wall_time}
            for p in series.points
        ]
        for name, series in run.metrics.items()
    }


def gate(path: Path, extra: list[str]) -> tuple[int, dict[str, Any]]:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        status = cli(["analyze", str(path), "--output", "json", *extra])
    return status, json.loads(output.getvalue())


def modules() -> dict[str, str]:
    root = Path(trainspotter.__file__).parent
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


@contextlib.contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("sdk_oracle", type=Path)
    parser.add_argument("leet_oracle", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--before", action="store_true")
    args = parser.parse_args()
    args.artifacts = args.artifacts.resolve()
    source = modules()
    manifest = json.loads((args.capture / "producer.json").read_text())
    sdk, leet = [json.loads(path.read_text()) for path in (args.sdk_oracle, args.leet_oracle)]
    assert sdk["capture_manifest_sha256"] == leet["capture_manifest_sha256"]
    assert (
        sdk["capture_manifest_sha256"]
        == hashlib.sha256((args.capture / "producer.json").read_bytes()).hexdigest()
    )
    args.artifacts.mkdir(parents=True, exist_ok=True)
    results = []
    for item, s, g in zip(manifest["cases"], sdk["cases"], leet["cases"], strict=True):
        name = item["case"]
        path = args.capture / item["file"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == item["sha256"]
            == s["file_sha256"]
            == g["file_sha256"]
        )
        assert s["case"] == g["case"] == name and s["record_types"] == g["record_types"]
        if args.before:
            try:
                load_run(path)
            except ValueError as exc:
                assert "cannot guess format" in str(exc)
                results.append({"case": name, "refusal": str(exc)})
            else:
                raise AssertionError("baseline unexpectedly decoded native W&B")
            continue
        axis = "_step" if name == "custom-counter" else "train/global_step"
        expected, independent = packets(s["history"], axis), packets(g["history"], axis)
        assert expected == independent, name
        run = load_run(path)
        assert observed(run) == expected, name
        native = run.meta["wandb_native"]
        assert native["record_types"] == s["record_types"]
        assert native["fragments_read"] == s["fragments_read"]
        findings = run_all(run)
        report = to_json_dict(run, findings)
        args.artifacts.joinpath(name + ".html").write_text(render_html(run, findings))
        args.artifacts.joinpath(name + ".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            render_terminal(run, findings, color=False)
        args.artifacts.joinpath(name + ".txt").write_text(terminal.getvalue())
        status, cli_report = gate(path, ["--fail-on", "error"])
        assert cli_report == report
        assert status == (1 if name == "unstable-lr" else 2 if name == "logged-null" else 0)
        result = {
            "case": name,
            "axis": axis,
            "input_complete": run.input_complete(),
            "file_sha256": item["sha256"],
            "points_sha256": digest(expected),
            "points": {key: len(points) for key, points in expected.items()},
            "record_types": native["record_types"],
            "fragments_read": native["fragments_read"],
            "cli_fail_on_error_status": status,
            "findings": [
                {
                    "detector": f.detector,
                    "severity": f.severity,
                    "step_start": f.step_start,
                    "step_end": f.step_end,
                }
                for f in findings
            ],
        }
        if name == "custom-counter":
            assert gate(path, ["--require-timing"])[0] == 2
            selected = load_run(path, step_key="optimizer_step")
            expected = packets(s["history"], "optimizer_step")
            assert observed(selected) == expected == packets(g["history"], "optimizer_step")
            assert gate(path, ["--step-key", "optimizer_step", "--require-timing"])[0] == 0
            args.artifacts.joinpath("custom-selected.html").write_text(
                render_html(selected, run_all(selected))
            )
            result["explicit_axis"] = {
                "axis": "optimizer_step",
                "points_sha256": digest(expected),
                "require_timing_status": 0,
            }
        results.append(result)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    controls = []
    if not args.before:
        original = (args.capture / "stable/run.wandb").read_bytes()
        oracle = sdk["cases"][0]
        large = next(row for row in oracle["long_records"] if row["kind"] == "history")
        prefix = packets(
            [r for r in oracle["history"] if r["record_num"] < large["record_num"]],
            "train/global_step",
        )
        for name in ["truncated-fragment", "corrupt-fragment", "truncated-tail"]:
            data = bytearray(original)
            if name == "truncated-fragment":
                data = data[: large["start"] + 40_000]
            elif name == "corrupt-fragment":
                data[large["start"] + 10] ^= 1
            else:
                data = data[:-2]
            path = args.artifacts / (name + ".wandb")
            path.write_bytes(data)
            with working_directory(args.artifacts):
                run = load_run(Path(path.name))
                status, _report = gate(Path(path.name), ["--fail-on", "error"])
            expected = (
                packets(oracle["history"], "train/global_step")
                if name == "truncated-tail"
                else prefix
            )
            assert observed(run) == expected
            assert not run.input_complete() and status == 2
            args.artifacts.joinpath(name + ".html").write_text(render_html(run, run_all(run)))
            controls.append(
                {
                    "case": name,
                    "authored_control": True,
                    "file_sha256": hashlib.sha256(data).hexdigest(),
                    "retained_points_sha256": digest(expected),
                    "last_loss_step": run.get("train/loss").steps()[-1],
                    "unread_event_bytes": run.unread_event_bytes(),
                    "tail_error": run.meta["wandb_native"]["tail_error"],
                    "cli_fail_on_error_status": 2,
                }
            )
    assert source == modules(), "auditor source changed during verification"
    payload = {
        "scope": (
            "All scalar values, coordinates and runtime clocks compared with independent official "
            "SDK/Go decoders; original SDK captures unchanged; binary64 metric semantics and "
            "exact step coordinates."
        ),
        "before": args.before,
        "producer_manifest_sha256": sdk["capture_manifest_sha256"],
        "oracle_script_sha256": sdk["script_sha256"],
        "sdk_decoder_source_sha256": sdk["cases"][0]["decoder_source_sha256"],
        "go_decoder_binary_sha256": leet["cases"][0]["decoder_binary_sha256"],
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime_modules": source,
        "cases": results,
        "authored_damage_controls": controls,
    }
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("Verified", len(results), "native captures; before=", args.before)


if __name__ == "__main__":
    main()
