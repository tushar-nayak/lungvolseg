#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import vtk

from lungvolseg.airway import read_polydata


def _make_actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float], opacity: float, width: float = 1.0) -> vtk.vtkActor:
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(width)
    return actor


def _surface_actor(polydata: vtk.vtkPolyData) -> vtk.vtkActor:
    actor = _make_actor(polydata, (0.78, 0.86, 0.97), 0.20)
    actor.GetProperty().SetRepresentationToSurface()
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().EdgeVisibilityOff()
    return actor


def _line_actor(polydata: vtk.vtkPolyData, color: tuple[float, float, float], width: float) -> vtk.vtkActor:
    actor = _make_actor(polydata, color, 1.0, width=width)
    actor.GetProperty().SetRepresentationToWireframe()
    return actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--surface", required=True, help="Airway surface mesh (.vtp, .vtk, .stl, .ply, .obj)")
    parser.add_argument("--centerlines", help="Centerline polydata (.vtp or .vtk)")
    parser.add_argument("--route", help="Optional route polyline (.vtp or .vtk)")
    parser.add_argument("--output", required=True, help="Output image path (.png)")
    parser.add_argument("--camera-elevation", type=float, default=25.0)
    parser.add_argument("--camera-azimuth", type=float, default=35.0)
    parser.add_argument("--show", action="store_true", help="Open an interactive window instead of only saving a PNG.")
    args = parser.parse_args()

    surface = read_polydata(args.surface)
    centerlines = read_polydata(args.centerlines) if args.centerlines else None
    route = read_polydata(args.route) if args.route else None

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)

    surface_actor = _surface_actor(surface)
    renderer.AddActor(surface_actor)

    if centerlines is not None:
        renderer.AddActor(_line_actor(centerlines, (0.10, 0.45, 0.85), 4.0))

    if route is not None:
        renderer.AddActor(_line_actor(route, (0.88, 0.20, 0.18), 6.0))

    bounds = surface.GetBounds()
    center = ((bounds[0] + bounds[1]) * 0.5, (bounds[2] + bounds[3]) * 0.5, (bounds[4] + bounds[5]) * 0.5)
    radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.SetPosition(center[0], center[1] - 2.5 * radius, center[2] + 0.8 * radius)
    camera.Azimuth(args.camera_azimuth)
    camera.Elevation(args.camera_elevation)
    camera.Zoom(1.25)

    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1600, 1200)
    window.SetOffScreenRendering(1)

    window.Render()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()

    if args.show:
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.Initialize()
        interactor.Start()


if __name__ == "__main__":
    main()
