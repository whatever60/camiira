import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SingleImageConfigTests(unittest.TestCase):
    def _read_options(self, config_name: str) -> dict:
        script = """
import json
import sys
from camiira.detect_colonies import _single_detector_options
from camiira.utils import read_config

print(json.dumps(_single_detector_options(read_config(sys.argv[1]))))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(REPO_ROOT / "test_data" / "configs" / config_name),
            ],
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_documented_rgb_config_uses_single_image_defaults(self):
        options = self._read_options("configure_small.yaml")

        self.assertEqual(options["bg_gaussian_kernel"], 18)
        self.assertEqual(options["fg_gaussian_kernel"], 5)
        self.assertEqual(options["clahe_clip_limit"], 2.0)
        self.assertEqual(options["clahe_tile_grid_size"], [4, 4])
        self.assertEqual(options["calib_contrast_alpha"], 1)
        self.assertEqual(options["calib_contrast_beta"], 0)

    def test_single_bmp_bhi_profile_is_complete(self):
        options = self._read_options("configure_single_bmp_bhi.yaml")

        self.assertEqual(options["crop_x_min"], 220)
        self.assertEqual(options["crop_x_max"], 3820)
        self.assertEqual(options["crop_y_min"], 220)
        self.assertEqual(options["crop_y_max"], 2450)


if __name__ == "__main__":
    unittest.main()
