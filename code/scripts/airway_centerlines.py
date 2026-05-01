#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lungvolseg.airway import run_vmtk_centerlines


def _parse_point(text: str) -> tuple[float, float, float]:
    values = [float(value) for value in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Points must be formatted as x,y,z")
    return values[0], values[1], values[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, help="Airway surface mesh (.vtp, .vtk, .stl, .ply, .obj)")
    parser.add_argument("--output", required=True, help="Output centerline file (.vtp)")
    parser.add_argument("--source-point", type=_parse_point, help="Trachea source point as x,y,z")
    parser.add_argument(
        "--target-point",
        action="append",
        type=_parse_point,
        default=[],
        help="Terminal target point as x,y,z. Repeat for multiple airway branches.",
    )
    parser.add_argument(
        "--seed-selector",
        choices=["pointlist", "openprofiles"],
        default="pointlist",
        help="Use pointlist for explicit seeds or openprofiles for cap-based seeding.",
    )
    parser.add_argument("--source-profile-id", action="append", type=int, default=[])
    parser.add_argument("--target-profile-id", action="append", type=int, default=[])
    args = parser.parse_args()

    if args.seed_selector == "pointlist":
        if args.source_point is None:
            raise SystemExit("--source-point is required for pointlist mode.")
        if not args.target_point:
            raise SystemExit("At least one --target-point is required for pointlist mode.")
        run_vmtk_centerlines(
            args.surface,
            args.output,
            source_points=[args.source_point],
            target_points=args.target_point,
            seed_selector="pointlist",
        )
    else:
        if not args.source_profile_id or not args.target_profile_id:
            raise SystemExit("--source-profile-id and --target-profile-id are required for openprofiles mode.")
        run_vmtk_centerlines(
            args.surface,
            args.output,
            source_profile_ids=args.source_profile_id,
            target_profile_ids=args.target_profile_id,
            seed_selector="openprofiles",
        )

    print(json.dumps({"centerlines": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
