from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lungvolseg.inference import infer_case
from lungvolseg.metrics import summarize_metrics
from lungvolseg.training import train_model
from lungvolseg.zenodo_covid_lung import prepare_zenodo_lung_cases, run_zenodo_lung_pipeline


class PipelineValidationTests(unittest.TestCase):
    def test_prepare_cases_rejects_non_positive_target_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "target_shape depth must be a positive integer"):
                prepare_zenodo_lung_cases(
                    raw_dir=tmpdir,
                    output_dir=tmpdir,
                    target_shape=(0, 128, 128),
                )

    def test_prepare_cases_rejects_non_positive_max_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "max_cases must be a positive integer"):
                prepare_zenodo_lung_cases(
                    raw_dir=tmpdir,
                    output_dir=tmpdir,
                    max_cases=0,
                )

    def test_run_pipeline_rejects_non_positive_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "epochs must be a positive integer"):
                run_zenodo_lung_pipeline(workspace=tmpdir, epochs=0)

    def test_train_model_rejects_empty_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "cases must contain at least one training example"):
                train_model(cases=[], output_dir=tmpdir)

    def test_train_model_rejects_invalid_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cases = [{"image": "fake_image.nii.gz", "label": "fake_label.nii.gz"}]
            with self.assertRaisesRegex(ValueError, "num_classes must be greater than 1"):
                train_model(cases=cases, output_dir=tmpdir, num_classes=1)

    def test_infer_case_rejects_invalid_patch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "prediction.nii.gz"
            with self.assertRaisesRegex(ValueError, "patch_size depth must be positive"):
                infer_case(
                    checkpoint_path="fake_checkpoint.pt",
                    image_path="fake_image.nii.gz",
                    output_path=output_path,
                    patch_size=(0, 96, 96),
                )

    def test_infer_case_rejects_invalid_num_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "prediction.nii.gz"
            with self.assertRaisesRegex(ValueError, "num_classes must be greater than 1"):
                infer_case(
                    checkpoint_path="fake_checkpoint.pt",
                    image_path="fake_image.nii.gz",
                    output_path=output_path,
                    num_classes=1,
                )

    def test_summarize_metrics_rejects_empty_case_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            with self.assertRaisesRegex(ValueError, "case_metrics must contain at least one case"):
                summarize_metrics({}, output_path)


if __name__ == "__main__":
    unittest.main()
