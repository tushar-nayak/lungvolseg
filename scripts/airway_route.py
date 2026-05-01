#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lungvolseg.airway import read_polydata, shortest_path_route, write_polydata


def _parse_point(text: str) -> tuple[float, float, float]:
    values = [float(value) for value in text.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Points must be formatted as x,y,z")
    return values[0], values[1], values[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--centerlines", required=True, help="Centerline polydata (.vtp)")
    parser.add_argument("--source-point", required=True, type=_parse_point, help="Trachea seed as x,y,z")
    parser.add_argument("--target-point", required=True, type=_parse_point, help="Target coordinate as x,y,z")
    parser.add_argument("--output", required=True, help="Output route polyline (.vtp)")
    args = parser.parse_args()

    centerlines = read_polydata(args.centerlines)
    route, summary = shortest_path_route(centerlines, args.source_point, args.target_point)
    write_polydata(route, args.output)

    print(json.dumps({"route": str(Path(args.output).resolve()), **summary}, indent=2))


if __name__ == "__main__":
    main()
