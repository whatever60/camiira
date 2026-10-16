import csv
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from camiira._selection import choose_replacement_rows  # noqa: E402


class ReplacementSelectionTests(unittest.TestCase):
    def test_candidate_positions_are_mapped_to_global_rows(self):
        selected = choose_replacement_rows(
            candidate_order=[1, 0],
            target_rows=[10, 42],
            target_barcodes=["plate_a", "plate_b"],
            source_barcodes=[],
            plate_limits={"plate_a": -1, "plate_b": -1},
            requested=1,
        )

        np.testing.assert_array_equal(selected, [42])

    def test_plate_limit_is_updated_after_each_replacement(self):
        selected = choose_replacement_rows(
            candidate_order=[0, 1, 2],
            target_rows=[5, 6, 7],
            target_barcodes=["plate_a", "plate_a", "plate_b"],
            source_barcodes=[],
            plate_limits={"plate_a": 1, "plate_b": 1},
            requested=3,
        )

        np.testing.assert_array_equal(selected, [5, 7])

    def test_existing_picks_count_toward_plate_limit(self):
        selected = choose_replacement_rows(
            candidate_order=[0, 1],
            target_rows=[8, 9],
            target_barcodes=["plate_a", "plate_b"],
            source_barcodes=["plate_a"],
            plate_limits={"plate_a": 1, "plate_b": 1},
            requested=2,
        )

        np.testing.assert_array_equal(selected, [9])


class ReleaseContractTests(unittest.TestCase):
    def test_release_version_and_pyarrow_dependency(self):
        project = (REPO_ROOT / "pyproject.toml").read_text()

        self.assertRegex(project, r'(?m)^version = "0\.4\.0"$')
        self.assertRegex(project, r'(?m)^\s*"pyarrow[><=]')

    def test_small_rgb_config_includes_calibration_settings(self):
        config = (
            REPO_ROOT / "test_data" / "configs" / "configure_small.yaml"
        ).read_text()

        self.assertRegex(
            config, r"(?m)^calib_gaussian_kernel:\s*\[13,\s*13\]\s*$"
        )
        self.assertRegex(config, r"(?m)^calib_gaussian_iteration:\s*20\s*$")

    def test_metadata_matches_camera_pipeline_fixtures(self):
        metadata_path = REPO_ROOT / "test_data" / "metadata" / "metadata.csv"
        with metadata_path.open(newline="") as metadata_file:
            metadata = list(csv.DictReader(metadata_file))

        metadata_barcodes = {row["barcode"] for row in metadata}
        bmp_barcodes = {
            path.name.split("_", 1)[0]
            for path in (REPO_ROOT / "test_data" / "bmp").glob("*.bmp")
        }
        detection_barcodes = {
            path.name.removesuffix("_metadata.csv")
            for path in (REPO_ROOT / "test_data" / "01_detect").glob(
                "*_metadata.csv"
            )
        }
        selection_barcodes = {
            path.name.removesuffix("_metadata_init.csv")
            for path in (REPO_ROOT / "test_data" / "02_select").glob(
                "*_metadata_init.csv"
            )
        }

        self.assertEqual(
            metadata_barcodes,
            bmp_barcodes,
        )
        self.assertEqual(metadata_barcodes, detection_barcodes)
        self.assertEqual(metadata_barcodes, selection_barcodes)
        self.assertEqual(
            {
                row["barcode"]: (
                    row["pick_group"],
                    int(row["num_picks_group"]),
                    int(row["max_picks_plate"]),
                )
                for row in metadata
            },
            {
                "FT1AR80": ("G1", 123, -1),
                "FT1AR81": ("G2", 45, -1),
                "FT4AR80": ("G1", 123, -1),
            },
        )

    def test_debugger_and_uninstalled_absolute_import_are_absent(self):
        source = (REPO_ROOT / "src" / "camiira" / "select_colonies.py").read_text()

        self.assertNotIn("pdb.set_trace", source)
        self.assertNotIn("from detect_colonies import", source)

    def test_isolate_cli_resolves_aligner_inside_the_package(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "camiira.isolate_interaction", "--help"],
            cwd=REPO_ROOT.parent,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:",completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("polars"),
        "requires the complete project environment",
    )
    def test_post_shortage_selects_all_targets_without_stopping(self):
        script = textwrap.dedent(
            """
            import json
            import tempfile
            import warnings
            from pathlib import Path
            from unittest import mock

            import polars as pl
            from camiira import select_colonies

            with tempfile.TemporaryDirectory() as temporary_directory:
                data_dir = Path(temporary_directory)
                pl.DataFrame(
                    {
                        "pca1": [0.0, 1.0],
                        "pca2": [0.0, 1.0],
                        "picking_status": [1, 2],
                    }
                ).write_csv(data_dir / "plate_metadata_init.csv")
                initial = {
                    "annotations": [
                        {"id": 0, "category_id": 1},
                        {"id": 1, "category_id": 2},
                    ]
                }
                reviewed = {
                    "annotations": [
                        {"id": 0, "category_id": 2},
                        {"id": 1, "category_id": 2},
                    ]
                }
                (data_dir / "plate_annot_init.json").write_text(
                    json.dumps(initial)
                )
                (data_dir / "plate_annot_init_post.json").write_text(
                    json.dumps(reviewed)
                )

                with (
                    mock.patch.object(
                        select_colonies,
                        "process_metadata",
                        return_value={"group": (["plate"], [-1], 3)},
                    ),
                    mock.patch.object(
                        select_colonies,
                        "read_file_list",
                        return_value=(["plate"], ["red.png"], ["white.png"]),
                    ),
                    mock.patch.object(
                        select_colonies, "_save_modified"
                    ) as save_modified,
                    warnings.catch_warnings(record=True) as observed_warnings,
                ):
                    warnings.simplefilter("always")
                    select_colonies.pick_colony_post(
                        "images",
                        str(data_dir),
                        "metadata.csv",
                    )

                saved_frame = save_modified.call_args.args[2]
                assert saved_frame["picking_status"].to_list() == [1, 1]
                warning_messages = [
                    str(item.message) for item in observed_warnings
                ]
                assert any(
                    "No picked colonies remain" in message
                    for message in warning_messages
                )
                assert any(
                    "Only 2 of 3 requested" in message
                    for message in warning_messages
                )
            """
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT / "src")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_v03_colab_uses_small_config_for_calibration_and_detection(self):
        notebook_path = (
            REPO_ROOT / "examples" / "camiira_v0.3.0_colab.ipynb"
        )
        notebook = json.loads(notebook_path.read_text())
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        calibration_cell = next(
            code for code in code_cells if "camiira.calc_calib_params" in code
        )
        detection_cell = next(
            code for code in code_cells if "camiira.detect_colonies" in code
        )
        self.assertIn('RELEASE = "v0.3.0"', "\n".join(code_cells))
        self.assertIn("SMALL_CONFIG_PATH", calibration_cell)
        self.assertIn("SMALL_CONFIG_PATH", detection_cell)
        self.assertNotIn('CONFIG_DIR / "configure.yaml"', calibration_cell)


if __name__ == "__main__":
    unittest.main()
