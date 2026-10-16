import tempfile
import unittest
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camiira._builtin_contours import contour_from_box, reconstruct_builtin_contours


class BuiltinContourTests(unittest.TestCase):
    def test_extracts_dark_and_bright_objects(self):
        for background, foreground in ((210, 30), (30, 210)):
            with self.subTest(foreground=foreground):
                image = np.full((64, 64, 3), background, np.uint8)
                cv.circle(image, (32, 32), 10, (foreground,) * 3, -1)
                contour, status = contour_from_box(image, [21, 21, 43, 43], [32, 32])
                self.assertEqual(status, "ok")
                self.assertGreater(cv.contourArea(contour), 250)
                self.assertLess(cv.contourArea(contour), 350)
                self.assertGreaterEqual(cv.pointPolygonTest(contour.astype(np.float32), (32, 32), False), 0)

    def test_failure_does_not_fabricate_a_rectangle(self):
        image = np.full((64, 64, 3), 100, np.uint8)
        for box, center, expected in (([20, 20, 40, 40], [30, 30], "low_contrast"), ([30, 30, 20, 20], [25, 25], "invalid_box"), ([-3, 20, 40, 40], [30, 30], "outside_image"), ([20, 20, 40, 40], [50, 50], "center_outside_box")):
            with self.subTest(expected=expected):
                contour, status = contour_from_box(image, box, center)
                self.assertIsNone(contour)
                self.assertEqual(status, expected)

    def test_exports_source_ids_scale_and_failure_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((64, 64, 3), 210, np.uint8)
            cv.circle(image, (32, 32), 10, (30,) * 3, -1)
            cv.imwrite(str(root / "image.png"), image)
            pd.DataFrame([
                {"Id": 17, "CentroidX": 16, "CentroidY": 16, "BBoxMinX": 10.5, "BBoxMinY": 10.5, "BBoxMaxX": 21.5, "BBoxMaxY": 21.5},
                {"Id": 23, "CentroidX": 3, "CentroidY": 3, "BBoxMinX": 2, "BBoxMinY": 2, "BBoxMaxX": 4, "BBoxMaxY": 4},
            ]).to_csv(root / "input.csv", index=False)
            result = reconstruct_builtin_contours(root / "image.png", root / "input.csv", root / "out", .5)
            self.assertEqual([a["id"] for a in result["annotations"]], [17, 23])
            self.assertEqual(result["annotations"][0]["source_center"], [32, 32])
            self.assertEqual(result["annotations"][0]["status"], "ok")
            self.assertEqual(result["annotations"][1]["segmentation"], [])
            self.assertIn("not native", result["info"]["contour_source"])
            self.assertTrue((root / "out" / "builtin_reconstructed.json").is_file())


if __name__ == "__main__":
    unittest.main()
