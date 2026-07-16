import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# These tests exercise input selection and calibration assembly without importing the
# heavy image-processing stack. CI and installed environments use the real modules.
for module_name in ("biom", "cv2", "yaml"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))

sys.modules["cv2"].IMREAD_GRAYSCALE = 0
sys.modules["cv2"].IMREAD_COLOR = 1

sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))
figure_module = sys.modules.setdefault(
    "matplotlib.figure", types.ModuleType("matplotlib.figure")
)
figure_module.Figure = type("Figure", (), {})
sys.modules.setdefault("scipy", types.ModuleType("scipy"))
stats_module = sys.modules.setdefault("scipy.stats", types.ModuleType("scipy.stats"))
stats_module.f = object()
tqdm_module = sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
tqdm_auto_module = sys.modules.setdefault("tqdm.auto", types.ModuleType("tqdm.auto"))
tqdm_auto_module.tqdm = lambda iterable: iterable
hyperspectral_module = types.ModuleType("camiira.hyperspectral_processing")
hyperspectral_module.process_bil2npz = lambda **_kwargs: None
hyperspectral_module.process_npz2png = lambda **_kwargs: None
sys.modules.setdefault("camiira.hyperspectral_processing", hyperspectral_module)

from camiira import calc_calib_params  # noqa: E402
from camiira import data_transform  # noqa: E402
from camiira.detector import CAMIIColonyDetector  # noqa: E402
from camiira.utils import (  # noqa: E402
    _get_time_points,
    parse_dir_for_time_series,
    read_file_list,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TimePointSelectionTests(unittest.TestCase):
    def test_unlabelled_images_work_with_min_and_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "red_rgb" / "plate1.png")
            _touch(root / "white_rgb" / "plate1.png")

            for selector in ("min", "max"):
                labels, red_images, white_images = read_file_list(
                    str(root), selector
                )
                self.assertEqual(labels, ["plate1"])
                self.assertTrue(red_images[0].endswith("red_rgb/plate1.png"))
                self.assertTrue(white_images[0].endswith("white_rgb/plate1.png"))

    def test_min_includes_unlabelled_and_max_selects_latest_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            for name in ("plate1.png", "plate1_d1.png", "plate1_d5.png"):
                _touch(image_dir / name)

            _, earliest_images = _get_time_points(str(image_dir), "min")
            _, latest_images = _get_time_points(str(image_dir), "max")

            self.assertTrue(earliest_images[0].endswith("plate1.png"))
            self.assertTrue(latest_images[0].endswith("plate1_d5.png"))

    def test_default_selector_argument_prefers_unlabelled_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _touch(image_dir / "plate1.png")
            _touch(image_dir / "plate1_d5.png")

            _, images = _get_time_points(str(image_dir))

            self.assertTrue(images[0].endswith("plate1.png"))

    def test_default_selector_uses_latest_when_unlabelled_image_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _touch(image_dir / "plate1_d1.png")
            _touch(image_dir / "plate1_d5.png")

            _, images = _get_time_points(str(image_dir), "default")

            self.assertTrue(images[0].endswith("plate1_d5.png"))

    def test_max_uses_latest_when_only_labelled_images_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _touch(image_dir / "plate1_d1.png")
            _touch(image_dir / "plate1_d5.png")

            _, images = _get_time_points(str(image_dir), "max")

            self.assertTrue(images[0].endswith("plate1_d5.png"))

    def test_minus_one_explicitly_selects_unlabelled_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _touch(image_dir / "plate1.png")
            _touch(image_dir / "plate1_d5.png")

            _, images = _get_time_points(str(image_dir), "-1")

            self.assertTrue(images[0].endswith("plate1.png"))

    def test_missing_specific_time_point_is_not_silently_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "red_rgb" / "plate1.png")
            _touch(root / "white_rgb" / "plate1.png")

            with self.assertRaisesRegex(ValueError, "No paired"):
                read_file_list(str(root), 3)

    def test_email_report_demo_directory_works_with_max_selector(self):
        labels, red_images, white_images = read_file_list(
            str(REPO_ROOT / "test_data" / "png"), "max"
        )

        self.assertEqual(len(labels), 8)
        self.assertEqual(len(red_images), 8)
        self.assertEqual(len(white_images), 8)

    def test_barcode_with_underscore_raises_parser_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_dir = Path(tmp)
            _touch(image_dir / "plate_one.png")

            with self.assertRaises(ValueError):
                _get_time_points(str(image_dir), "min")

    def test_compound_metadata_suffix_is_removed_before_time_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp)
            _touch(metadata_dir / "plate_metadata.csv")
            _touch(metadata_dir / "plate_d5_metadata.csv")

            series = parse_dir_for_time_series(
                str(metadata_dir), ext="_metadata.csv"
            )

            self.assertEqual(set(series), {"plate"})
            self.assertEqual(set(series["plate"]), {-1, 5})


class CalibrationInputTests(unittest.TestCase):
    @staticmethod
    def _config():
        return {
            "crop_x_min": None,
            "crop_x_max": None,
            "crop_y_min": None,
            "crop_y_max": None,
            "calib_gaussian_kernel": (3, 3),
            "calib_gaussian_iteration": 1,
        }

    def test_calibration_stacks_materialized_image_sequences(self):
        grayscale = np.ones((2, 2), dtype=np.uint8)
        color = np.ones((2, 2, 3), dtype=np.uint8)

        def fake_imread(_path, flags):
            return grayscale if flags == 0 else color

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "calibration.npz")
            with (
                mock.patch.object(
                    calc_calib_params,
                    "read_config",
                    return_value=self._config(),
                ),
                mock.patch.object(
                    calc_calib_params,
                    "read_file_list",
                    return_value=(
                        ["plate1", "plate2"],
                        ["red1.png", "red2.png"],
                        ["white1.png", "white2.png"],
                    ),
                ),
                mock.patch.object(
                    calc_calib_params.cv, "imread", fake_imread, create=True
                ),
                mock.patch.object(
                    calc_calib_params.cv,
                    "GaussianBlur",
                    side_effect=lambda image, _kernel, _sigma: image,
                    create=True,
                ),
            ):
                calc_calib_params.calib("images", output_path, "config.yaml")

            with np.load(output_path) as calibration:
                self.assertEqual(calibration["image_trans_calib"].shape, (2, 2))
                self.assertEqual(calibration["image_epi_calib"].shape, (2, 2, 3))

    def test_output_dir_remains_a_cli_alias_for_output_path(self):
        with mock.patch.object(calc_calib_params, "calib") as calib_mock:
            with mock.patch.object(
                sys,
                "argv",
                [
                    "calc_calib_params.py",
                    "--input_dir",
                    "images",
                    "--output_dir",
                    "calibration.npz",
                    "--config_path",
                    "config.yaml",
                ],
            ):
                calc_calib_params.main()

        calib_mock.assert_called_once_with(
            "images", "calibration.npz", "config.yaml"
        )

    def test_unreadable_calibration_image_has_clear_error(self):
        with mock.patch.object(
            calc_calib_params.cv, "imread", return_value=None, create=True
        ):
            with self.assertRaisesRegex(ValueError, "Could not read calibration image"):
                calc_calib_params._read_and_crop(
                    "missing.png", 0, None, None, None, None
                )

    def test_single_image_detector_loads_npz_calibration(self):
        expected = np.arange(4, dtype=float).reshape(2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            calibration_path = os.path.join(tmp, "calibration.npz")
            np.savez(calibration_path, image_trans_calib=expected)

            detector = CAMIIColonyDetector.__new__(CAMIIColonyDetector)
            observed = detector.load_calibration(calibration_path)

        np.testing.assert_array_equal(observed, expected)


class BmpPairValidationTests(unittest.TestCase):
    def test_incomplete_bmp_pair_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            output_dir = Path(tmp) / "output"
            _touch(input_dir / "plate1_001.bmp")

            with self.assertRaisesRegex(ValueError, "exactly two BMP images"):
                data_transform.process_bmp(str(input_dir), str(output_dir))

            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
