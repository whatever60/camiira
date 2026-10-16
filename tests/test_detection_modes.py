import json
import tempfile
import unittest
import sys
import warnings
from pathlib import Path
from unittest import mock

import cv2 as cv
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camiira import detect_colonies as detection, select_colonies as selection
from camiira.utils import read_config, read_file_list, read_plate_inputs, resolve_plate_inputs


class DetectionModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_four_combinations_share_one_processor_and_write_relative_inputs(self):
        red = self.image("images/red_rgb/plate_d5.png")
        white = self.image("images/white_rgb/plate_d5.bmp")
        for mode, batch in (("dual", False), ("dual", True), ("epi-only", False), ("epi-only", True)):
            root = self.root / "images"
            if mode == "epi-only" and batch:
                root = self.root / "epi"
                white = self.image("epi/white_rgb/plate_d5.bmp")
            path = str(root) if batch else (red if mode == "dual" else white)
            out = self.root / f"out-{mode}-{batch}"
            with self.subTest(mode=mode, batch=batch), mock.patch.object(detection, "_detect_plates") as processor, mock.patch.object(detection, "read_config", return_value={}) as config:
                detection.detect_colonies(path, str(out), config_path="config", mode=mode,
                                          white_path=white if mode == "dual" and not batch else None)
                processor.assert_called_once()
                config.assert_called_once()
                plates = processor.call_args.args[0]
                self.assertEqual(set(plates), {"plate"})
                self.assertEqual(plates["plate"]["trans_path"] is None, mode == "epi-only")
                manifest = json.loads((out / "inputs.json").read_text())
                self.assertEqual(manifest["mode"], mode)
                self.assertFalse(Path(manifest["plates"]["plate"]["epi_path"]).is_absolute())
                image_root = root if batch else Path(path).parent
                self.assertEqual(read_plate_inputs(out, image_root), plates)

    def test_auto_only_recognizes_explicit_channel_directories(self):
        self.image("white_rgb/plate.bmp")
        self.assertEqual(resolve_plate_inputs(self.root, "auto")[0], "epi-only")
        self.image("red_rgb/plate.png")
        self.assertEqual(resolve_plate_inputs(self.root, "auto")[0], "dual")
        image = self.image("plate.png")
        with self.assertRaisesRegex(ValueError, "Single plates require"):
            resolve_plate_inputs(image, "auto")

    def test_bad_layouts_are_rejected_before_output(self):
        roots = []
        for index, files in enumerate((
            ("plate.png",), ("plate.bmp",), ("red_rgb/plate.png",),
            ("white_rgb/plate.png", "plate.bmp"),
            ("white_rgb/nested/plate.png",),
            ("white_rgb/plate.png", "unexpected/plate.png"),
        )):
            root = self.root / str(index)
            for name in files:
                self.image(f"{index}/{name}")
            roots.append(root)
        for root in roots:
            with self.subTest(root=root), mock.patch.object(detection, "_detect_plates") as processor:
                out = self.root / "invalid-output"
                with self.assertRaises(ValueError):
                    detection.detect_colonies(str(root), str(out), mode="auto", config_path="config")
                processor.assert_not_called()
                self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            resolve_plate_inputs(self.root / "2", "epi-only")

    def test_all_formats_share_time_selection(self):
        first = self.image("white_rgb/plate.png")
        self.image("white_rgb/plate_d1.BMP")
        last = self.image("white_rgb/plate_d5.tiff")
        for selector, expected in (("min", first), ("max", last), ("default", first), ("-1", first)):
            _, plates = resolve_plate_inputs(self.root, "epi-only", time_label=selector)
            self.assertEqual(plates["plate"]["epi_path"], expected)

    def test_timestamped_bmps_are_epi_only_not_a_camera_mode(self):
        first = self.image("white_rgb/P1_20260709112041.bmp")
        last = self.image("white_rgb/P1_2026-09-03_12-01-00.bmp")
        for selector, expected in (("min", first), ("max", last)):
            mode, plates = resolve_plate_inputs(self.root, "auto", time_label=selector)
            self.assertEqual(mode, "epi-only")
            self.assertEqual(plates["P1"]["epi_path"], expected)

    def test_duplicate_time_point_and_underscore_barcode_fail(self):
        self.image("white_rgb/plate_d5.png")
        self.image("white_rgb/plate_d5.bmp")
        with self.assertRaises(ValueError):
            resolve_plate_inputs(self.root, "epi-only")
        image = self.image("plate_one.bmp")
        with self.assertRaises(ValueError):
            resolve_plate_inputs(image, "epi-only")

    def test_dual_single_requires_matching_partner_and_epi_rejects_partner(self):
        red = self.image("red/plate.png")
        white = self.image("white/other.png")
        with self.assertRaisesRegex(ValueError, "white_path"):
            resolve_plate_inputs(red, "dual")
        with self.assertRaisesRegex(ValueError, "matching barcodes"):
            resolve_plate_inputs(red, "dual", white)
        with self.assertRaisesRegex(ValueError, "white_path"):
            resolve_plate_inputs(red, "epi-only", white)
        with self.assertRaisesRegex(ValueError, "distinct"):
            resolve_plate_inputs(red, "dual", red)

    def test_dual_pairs_by_barcode_not_independent_filename_order(self):
        for name in ("red_rgb/P1_d5.png", "red_rgb/P10.png",
                     "white_rgb/P1.png", "white_rgb/P10_d5.png"):
            self.image(name)
        _, plates = resolve_plate_inputs(self.root, "dual")
        self.assertTrue(plates["P1"]["epi_path"].endswith("P1.png"))
        self.assertTrue(plates["P10"]["epi_path"].endswith("P10_d5.png"))
        labels, red, white = read_file_list(str(self.root), "max")
        for label, r, w in zip(labels, red, white):
            self.assertEqual(Path(r).stem.split("_")[0], label)
            self.assertEqual(Path(w).stem.split("_")[0], label)

    def test_pair_pipeline_accepts_null_crop_offsets(self):
        red = self.image("red/plate.png")
        white = self.image("white/plate.png")
        image = np.full((64, 64, 3), 100, np.uint8)
        cv.imwrite(red, image)
        cv.imwrite(white, image)
        config = {"crop_x_min": None, "crop_x_max": None, "crop_y_min": None,
                  "crop_y_max": None, "calib_contrast_trans_alpha": 1,
                  "calib_contrast_trans_beta": 0}
        contour = np.array([[[20, 20]], [[40, 20]], [[40, 40]], [[20, 40]]], np.int32)
        frame = pl.DataFrame({"center_x": [30.], "center_y": [30.]})
        with mock.patch.object(detection, "detect_colony", return_value=[contour]), mock.patch.object(detection, "colony_qc", return_value=([contour], frame)):
            result = detection._detect_colony_pairs([("plate", red, white)], None, None, config)
        np.testing.assert_array_equal(result[0][1][0], contour)
        self.assertEqual(result[0][0]["center_x"].to_list(), [30.])


    def test_manifest_does_not_reselect_time_points_and_is_validated(self):
        self.image("white_rgb/plate_d1.png")
        out = self.root / "detect"
        with mock.patch.object(detection, "_detect_plates"), mock.patch.object(detection, "read_config", return_value={}):
            detection.detect_colonies(str(self.root), str(out), config_path="config", mode="epi-only")
        self.image("white_rgb/plate_d9.png")
        selected = selection._selection_inputs(str(self.root), str(out))
        self.assertTrue(selected["plate"]["epi_path"].endswith("plate_d1.png"))
        manifest = json.loads((out / "inputs.json").read_text())
        manifest["mode"] = "dual"
        (out / "inputs.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Invalid"):
            selection._selection_inputs(str(self.root), str(out))

    def test_omitted_mode_preserves_legacy_cli_dispatch(self):
        image = self.image("plate.png")
        for path, target in ((image, "detect_colony_single"), (str(self.root), "detect_colony_batch")):
            with self.subTest(path=path), mock.patch.object(detection, target) as legacy, mock.patch.object(detection, "detect_colonies") as router, mock.patch("sys.argv", ["detect_colonies.py", "-i", path, "-o", "out", "-c", "config", "--toss_red"]):
                with self.assertWarnsRegex(FutureWarning, "deprecated legacy CLI interface"):
                    detection.main()
                legacy.assert_called_once()
                router.assert_not_called()

    def test_new_cli_has_only_explicit_modes(self):
        image = self.image("plate.png")
        with mock.patch.object(detection, "detect_colonies") as router, mock.patch("sys.argv", ["detect_colonies.py", "-i", image, "-o", "out", "-c", "config", "--mode", "epi-only"]):
            with warnings.catch_warnings(record=True) as caught:
                detection.main()
            self.assertEqual(caught, [])
            self.assertEqual(router.call_args.kwargs["mode"], "epi-only")


class PipelineModeTests(unittest.TestCase):
    def test_four_modes_through_detection_selection_manual_review_and_picking(self):
        config = read_config(str(Path(__file__).resolve().parents[1] / "test_data/configs/configure_single_bmp_bhi.yaml"))
        config.update(crop_x_min=None, crop_x_max=None, crop_y_min=None, crop_y_max=None,
                      calib_contrast_trans_alpha=1, calib_contrast_trans_beta=0,
                      bg_threshold_block_size=31, bg_threshold_offset=5, skip_clahe=True,
                      segment_bias=2, filter_bias=2, circularity_threshold=0,
                      circularity_threshold_very_bad=0, small_size_circularity=0)
        for mode, batch in (("dual", False), ("epi-only", False), ("dual", True), ("epi-only", True), ("legacy", True)):
            with self.subTest(mode=mode, batch=batch), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                images, detected, selected, final = [root / n for n in ("images", "detect", "select", "final")]
                barcodes = ["P1", "P2"] if batch else ["P1"]
                dual = mode in {"dual", "legacy"}
                extension = "png" if mode == "legacy" else "bmp"
                for light in (("red", "white") if dual else ("white",)):
                    folder = images / f"{light}_rgb"
                    folder.mkdir(parents=True)
                    for barcode in barcodes:
                        image = np.full((192, 192, 3), 200, np.uint8)
                        for center, radius in (((45, 45), 8), ((95, 80), 13), ((145, 145), 18)):
                            cv.circle(image, center, radius, (20, 20, 20), -1)
                        cv.imwrite(str(folder / f"{barcode}_d1.{extension}"), image)
                input_path = images if batch else images / ("red_rgb" if dual else "white_rgb") / "P1_d1.bmp"
                image_root = images if batch else input_path.parent
                partner = str(images / "white_rgb/P1_d1.bmp") if mode == "dual" and not batch else None
                with mock.patch.object(detection, "read_config", return_value=config):
                    if mode == "legacy":
                        detection.detect_colony_batch(str(images), str(detected), config_path="config")
                    else:
                        detection.detect_colonies(str(input_path), str(detected), config_path="config", mode=mode, white_path=partner)
                metadata = root / "metadata.csv"
                pl.DataFrame({"barcode": barcodes, "pick_group": ["G"] * len(barcodes),
                              "max_picks_plate": [-1] * len(barcodes),
                              "num_picks_group": [1000] * len(barcodes)}).write_csv(metadata)
                reader = selection.read_file_list if mode == "legacy" else AssertionError("Must use the recorded inputs")
                with mock.patch.object(selection, "read_config", return_value=config), mock.patch.object(selection, "read_file_list", side_effect=reader), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    selection.pick_colony_init(str(image_root), str(detected), str(selected), str(metadata), "config")
                    for barcode in barcodes:
                        annotation = json.loads((selected / f"{barcode}_annot_init.json").read_text())
                        self.assertGreaterEqual(len(annotation["annotations"]), 2)
                        annotation["annotations"][-1]["category_id"] = 3
                        added = dict(annotation["annotations"][-1], id=len(annotation["annotations"]),
                                     segmentation=[[20, 145, 30, 145, 30, 155, 20, 155]])
                        annotation["annotations"].append(added)
                        (selected / f"{barcode}_annot_init_post.json").write_text(json.dumps(annotation))
                    selection.pick_colony_post(str(image_root), str(selected), str(metadata))
                    selection.pick_colony_final(str(image_root), str(selected), str(metadata), str(final), tsp_method="heuristic")
                if mode != "legacy":
                    self.assertEqual((detected / "inputs.json").read_bytes(), (selected / "inputs.json").read_bytes())
                    self.assertEqual((detected / "inputs.json").read_bytes(), (final / "inputs.json").read_bytes())
                for barcode in barcodes:
                    frame = pl.read_csv(final / f"{barcode}_metadata.csv")
                    picks = pl.read_csv(final / f"{barcode}_picking.csv", has_header=False)
                    self.assertEqual(len(picks), frame.filter(pl.col("picking_status") == 1).height)
                    self.assertEqual(frame.filter(pl.col("picking_status") == 3).height, 2)
                    self.assertTrue((final / f"{barcode}_rgb_white_contour.jpg").is_file())
                    self.assertTrue((final / f"{barcode}_gs_white_contour.jpg").is_file())
                    self.assertEqual((final / f"{barcode}_rgb_red_contour.jpg").exists(), dual)
                    self.assertTrue(np.isfinite(picks.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
