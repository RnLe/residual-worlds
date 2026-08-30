"""Command-line entry point.

One console command, ``residual-worlds``, with subcommands that mirror
the experiment stages. Heavy imports happen inside each handler so that
``--help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    """Commands report one JSON object on stdout for scripting."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_doctor(args: argparse.Namespace) -> int:
    from residual_worlds.doctor import run_doctor

    report = run_doctor()
    _emit(report)
    return 0 if report["ok"] else 1


def _cmd_verify_simulator(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.physics.verification import run_simulator_verification

    contract = load_contract(Path(args.config))
    result = run_simulator_verification(contract)
    _emit(result)
    return 0 if result["all_passed"] else 1


def _cmd_generate_scenarios(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.task.scenarios import generate_bank, write_bank_manifest

    contract = load_contract(Path(args.config))
    scenarios = generate_bank(contract, args.bank)
    path = write_bank_manifest(contract, args.bank, scenarios, Path(args.output_dir))
    _emit({"bank": args.bank, "count": len(scenarios), "path": str(path)})
    return 0


def _cmd_generate_data(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.data.generate import generate_world_dataset

    contract = load_contract(Path(args.config))
    result = generate_world_dataset(
        contract,
        world_id=args.world,
        replicate=args.replicate,
        scenario_dir=Path(args.scenario_dir),
    )
    _emit(result)
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.smoke import run_smoke

    contract = load_contract(Path(args.config))
    result = run_smoke(contract)
    _emit(result)
    return 0 if result["ok"] else 1


def _cmd_bundle_build_core(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.release.build_core import build_core_bundle

    contract = load_contract(Path(args.config))
    result = build_core_bundle(
        contract,
        Path(args.analysis),
        Path(args.figures),
        args.content_status,
    )
    _emit(result)
    return 0


def _cmd_bundle_verify(args: argparse.Namespace) -> int:
    from residual_worlds.release.verify_bundle import BundleError, verify_bundle

    try:
        result = verify_bundle(Path(args.bundle), args.require_content_status)
    except BundleError as error:
        _emit({"ok": False, "error": str(error)})
        return 1
    _emit({"ok": True, **result})
    return 0


def _cmd_fixture_schematic(args: argparse.Namespace) -> int:
    from residual_worlds.release.schematic import build_schematic_fixture

    destination = build_schematic_fixture(Path(args.destination))
    _emit({"destination": str(destination)})
    return 0


def _cmd_report_stage(args: argparse.Namespace) -> int:
    from residual_worlds.reporting.stage import stage_report

    result = stage_report(Path(args.bundle))
    _emit(result)
    return 0


def _cmd_report_build(args: argparse.Namespace) -> int:
    from residual_worlds.reporting.build import build_report

    result = build_report(Path(args.output) if args.output else None)
    _emit(result)
    return 0


def _cmd_render_preview(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.media.preview import RenderOptions, render_preview

    contract = load_contract(Path(args.config))
    options = RenderOptions(
        width=args.width, height=args.height, fps=args.fps, gif_stride=args.gif_stride
    )
    result = render_preview(contract, Path(args.output_dir), args.world, options)
    _emit(result)
    return 0


def _cmd_fixture_arm_golden(args: argparse.Namespace) -> int:
    from residual_worlds.config import load_contract
    from residual_worlds.media.golden import write_arm_golden

    contract = load_contract(Path(args.config))
    destination = write_arm_golden(contract, Path(args.destination), args.world)
    _emit({"destination": str(destination)})
    return 0


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to the experiment contract YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="residual-worlds",
        description="Residual dynamics learning and MPC on a two-link arm",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="report the toolchain and compute environment")
    doctor.set_defaults(handler=_cmd_doctor)

    verify = subparsers.add_parser(
        "verify-simulator", help="run the analytic and numerical simulator verification suite"
    )
    _add_config_argument(verify)
    verify.set_defaults(handler=_cmd_verify_simulator)

    scenarios = subparsers.add_parser(
        "generate-scenarios", help="generate one frozen scenario bank"
    )
    _add_config_argument(scenarios)
    scenarios.add_argument(
        "--bank",
        required=True,
        choices=["calibration", "pilot", "training_task", "protected"],
    )
    scenarios.add_argument("--output-dir", default="scenarios")
    scenarios.set_defaults(handler=_cmd_generate_scenarios)

    data = subparsers.add_parser("generate-data", help="generate one target-world dataset")
    _add_config_argument(data)
    data.add_argument("--world", required=True)
    data.add_argument("--replicate", type=int, required=True)
    data.add_argument("--scenario-dir", default="scenarios")
    data.set_defaults(handler=_cmd_generate_data)

    smoke = subparsers.add_parser("smoke", help="run the CPU end-to-end smoke pipeline")
    _add_config_argument(smoke)
    smoke.set_defaults(handler=_cmd_smoke)

    build_core = subparsers.add_parser(
        "bundle-build-core", help="build the public result bundle from an analysis"
    )
    _add_config_argument(build_core)
    build_core.add_argument("--analysis", required=True, help="analysis artifact directory")
    build_core.add_argument("--figures", required=True, help="rendered figures directory")
    build_core.add_argument(
        "--content-status", required=True, choices=["schematic", "pilot", "final"]
    )
    build_core.set_defaults(handler=_cmd_bundle_build_core)

    bundle_verify = subparsers.add_parser("bundle-verify", help="verify a public result bundle")
    bundle_verify.add_argument("--bundle", required=True)
    bundle_verify.add_argument(
        "--require-content-status", choices=["schematic", "pilot", "final"], default=None
    )
    bundle_verify.set_defaults(handler=_cmd_bundle_verify)

    fixture = subparsers.add_parser(
        "fixture-schematic", help="regenerate the checked-in schematic fixture bundle"
    )
    fixture.add_argument("--destination", default="tests/fixtures/public_result_schematic")
    fixture.set_defaults(handler=_cmd_fixture_schematic)

    report_stage = subparsers.add_parser(
        "report-stage", help="stage a verified bundle into report/generated"
    )
    report_stage.add_argument("--bundle", required=True)
    report_stage.set_defaults(handler=_cmd_report_stage)

    report_build = subparsers.add_parser(
        "report-build", help="compile the Typst report from staged data"
    )
    report_build.add_argument("--output", default=None)
    report_build.set_defaults(handler=_cmd_report_build)

    preview = subparsers.add_parser(
        "render-preview", help="render the animated arm preview for the site"
    )
    _add_config_argument(preview)
    preview.add_argument("--world", default="composite_standard")
    preview.add_argument("--output-dir", default="site/public")
    preview.add_argument("--width", type=int, default=960)
    preview.add_argument("--height", type=int, default=540)
    preview.add_argument("--fps", type=int, default=20)
    preview.add_argument("--gif-stride", type=int, default=1)
    preview.set_defaults(handler=_cmd_render_preview)

    arm_golden = subparsers.add_parser(
        "fixture-arm-golden", help="write the golden numbers that pin the site physics"
    )
    _add_config_argument(arm_golden)
    arm_golden.add_argument("--world", default="composite_standard")
    arm_golden.add_argument("--destination", default="tests/fixtures/arm_golden.json")
    arm_golden.set_defaults(handler=_cmd_fixture_arm_golden)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from residual_worlds.runtime import configure_torch_cpu

    configure_torch_cpu()
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except Exception as error:  # surface a compact error, keep traceback for -X dev
        print(f"error: {error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
