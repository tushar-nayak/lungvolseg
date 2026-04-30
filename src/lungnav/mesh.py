from __future__ import annotations

import os
from pathlib import Path

import vtk

from .config import CLASS_NAMES, ensure_dir


def _make_surface(reader_output: vtk.vtkImageData, label_value: int) -> vtk.vtkPolyData:
    extractor = vtk.vtkDiscreteMarchingCubes()
    extractor.SetInputData(reader_output)
    extractor.GenerateValues(1, label_value, label_value)
    extractor.Update()

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(extractor.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.Update()
    return smoother.GetOutput()


def _write_polydata(polydata: vtk.vtkPolyData, stem: Path) -> None:
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(str(stem.with_suffix(".stl")))
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

    vtp_writer = vtk.vtkXMLPolyDataWriter()
    vtp_writer.SetFileName(str(stem.with_suffix(".vtp")))
    vtp_writer.SetInputData(polydata)
    vtp_writer.Write()


def _render_preview(polydata_map: dict[int, vtk.vtkPolyData], output_path: Path) -> None:
    colors = {
        1: (0.4, 0.7, 1.0),
        2: (1.0, 0.6, 0.2),
    }
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(1000, 800)

    for label_value, polydata in polydata_map.items():
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*colors.get(label_value, (0.8, 0.8, 0.8)))
        actor.GetProperty().SetOpacity(0.85 if label_value == 1 else 1.0)
        renderer.AddActor(actor)

    renderer.ResetCamera()
    window.Render()

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()


def extract_meshes(
    label_image_path: str | Path,
    output_dir: str | Path,
    class_names: dict[int, str] | None = None,
) -> list[str]:
    output_dir = ensure_dir(output_dir)
    class_names = class_names or CLASS_NAMES
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(label_image_path))
    reader.Update()

    outputs: list[str] = []
    preview_map: dict[int, vtk.vtkPolyData] = {}
    for label_value, label_name in class_names.items():
        if label_value == 0:
            continue
        polydata = _make_surface(reader.GetOutput(), label_value)
        if polydata.GetNumberOfPoints() == 0:
            continue
        preview_map[label_value] = polydata
        stem = Path(output_dir) / label_name
        _write_polydata(polydata, stem)
        outputs.extend([str(stem.with_suffix(".stl")), str(stem.with_suffix(".vtp"))])

    if preview_map and os.environ.get("LUNGNAV_RENDER_PREVIEW") == "1":
        preview_path = Path(output_dir) / "mesh_preview.png"
        _render_preview(preview_map, preview_path)
        outputs.append(str(preview_path))

    return outputs
