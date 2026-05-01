from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def _as_point(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("Expected a 3D point.")
    return float(values[0]), float(values[1]), float(values[2])


def read_polydata(path: str | Path) -> vtk.vtkPolyData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vtp":
        reader: vtk.vtkAlgorithm = vtk.vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
    elif suffix == ".ply":
        reader = vtk.vtkPLYReader()
    elif suffix == ".obj":
        reader = vtk.vtkOBJReader()
    else:
        raise ValueError(f"Unsupported polydata format: {path}")

    reader.SetFileName(str(path))
    reader.Update()
    output = vtk.vtkPolyData()
    output.ShallowCopy(reader.GetOutput())
    return output


def write_polydata(polydata: vtk.vtkPolyData, path: str | Path) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".vtp":
        writer: vtk.vtkAlgorithm = vtk.vtkXMLPolyDataWriter()
    elif suffix == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif suffix == ".stl":
        writer = vtk.vtkSTLWriter()
    else:
        raise ValueError(f"Unsupported polydata output format: {path}")

    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def _flatten_points(points: Iterable[Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for point in points:
        flattened.extend(_as_point(point))
    return flattened


def run_vmtk_centerlines(
    surface_path: str | Path,
    output_path: str | Path,
    source_points: Sequence[Sequence[float]] | None = None,
    target_points: Sequence[Sequence[float]] | None = None,
    source_profile_ids: Sequence[int] | None = None,
    target_profile_ids: Sequence[int] | None = None,
    endpoints: bool = True,
    seed_selector: str | None = None,
) -> None:
    surface_path = Path(surface_path)
    output_path = Path(output_path)

    if seed_selector is None:
        seed_selector = "pointlist" if source_points and target_points else "openprofiles"

    if seed_selector not in {"pointlist", "openprofiles"}:
        raise ValueError("seed_selector must be 'pointlist' or 'openprofiles'.")

    command = ["vmtkcenterlines", "-ifile", str(surface_path), "-ofile", str(output_path)]
    command += ["-seedselector", seed_selector]

    if seed_selector == "pointlist":
        if not source_points or not target_points:
            raise ValueError("pointlist mode requires source_points and target_points.")
        command += ["-sourcepoints", *map(str, _flatten_points(source_points))]
        command += ["-targetpoints", *map(str, _flatten_points(target_points))]
    else:
        if not source_profile_ids or not target_profile_ids:
            raise ValueError("openprofiles mode requires source_profile_ids and target_profile_ids.")
        command += ["-sourceids", *map(str, source_profile_ids)]
        command += ["-targetids", *map(str, target_profile_ids)]

    if endpoints:
        command += ["-endpoints", "1"]

    try:
        import subprocess

        subprocess.run(command, check=True)
        return
    except FileNotFoundError:
        pass

    try:
        from vmtk import pypes
    except Exception as exc:  # pragma: no cover - depends on local vmtk install
        raise RuntimeError(
            "vmtkcenterlines was not found on PATH and the Python vmtk package is unavailable."
        ) from exc

    pypes.PypeRun(" ".join(command))


def build_adjacency(centerlines: vtk.vtkPolyData) -> tuple[np.ndarray, dict[int, list[tuple[int, float]]]]:
    points = vtk_to_numpy(centerlines.GetPoints().GetData())
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)

    lines = centerlines.GetLines()
    lines.InitTraversal()
    cell = vtk.vtkIdList()
    while lines.GetNextCell(cell):
        ids = [cell.GetId(i) for i in range(cell.GetNumberOfIds())]
        for left, right in zip(ids, ids[1:]):
            weight = float(np.linalg.norm(points[left] - points[right]))
            adjacency[left].append((right, weight))
            adjacency[right].append((left, weight))

    return points, adjacency


def _nearest_point_id(points: np.ndarray, query_point: Sequence[float]) -> int:
    query = np.asarray(_as_point(query_point), dtype=float)
    deltas = points - query
    return int(np.argmin(np.sum(deltas * deltas, axis=1)))


def shortest_path_route(
    centerlines: vtk.vtkPolyData,
    source_point: Sequence[float],
    target_point: Sequence[float],
) -> tuple[vtk.vtkPolyData, dict[str, float]]:
    points, adjacency = build_adjacency(centerlines)
    source_id = _nearest_point_id(points, source_point)
    target_id = _nearest_point_id(points, target_point)

    distances: dict[int, float] = {source_id: 0.0}
    previous: dict[int, int] = {}
    heap: list[tuple[float, int]] = [(0.0, source_id)]

    while heap:
        current_distance, node_id = heappop(heap)
        if node_id == target_id:
            break
        if current_distance != distances.get(node_id):
            continue
        for neighbor_id, edge_weight in adjacency.get(node_id, []):
            candidate = current_distance + edge_weight
            if candidate < distances.get(neighbor_id, float("inf")):
                distances[neighbor_id] = candidate
                previous[neighbor_id] = node_id
                heappush(heap, (candidate, neighbor_id))

    if target_id not in distances:
        raise RuntimeError("No path found between the selected source and target points.")

    path_ids = [target_id]
    while path_ids[-1] != source_id:
        path_ids.append(previous[path_ids[-1]])
    path_ids.reverse()

    route_points = vtk.vtkPoints()
    route_lines = vtk.vtkCellArray()
    route_line = vtk.vtkPolyLine()
    route_line.GetPointIds().SetNumberOfIds(len(path_ids))
    cumulative = vtk.vtkDoubleArray()
    cumulative.SetName("ArcLength")
    cumulative.SetNumberOfComponents(1)
    cumulative.SetNumberOfTuples(len(path_ids))

    total_length = 0.0
    for index, point_id in enumerate(path_ids):
        route_points.InsertNextPoint(points[point_id])
        route_line.GetPointIds().SetId(index, index)
        if index > 0:
            total_length += float(np.linalg.norm(points[path_ids[index]] - points[path_ids[index - 1]]))
        cumulative.SetValue(index, total_length)

    route_lines.InsertNextCell(route_line)
    route = vtk.vtkPolyData()
    route.SetPoints(route_points)
    route.SetLines(route_lines)
    route.GetPointData().AddArray(cumulative)

    return route, {
        "source_point_id": float(source_id),
        "target_point_id": float(target_id),
        "path_length": float(total_length),
    }

