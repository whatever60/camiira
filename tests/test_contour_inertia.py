import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camiira.detect_colonies import _calc_inertia_ratio


class ContourInertiaTests(unittest.TestCase):
    def test_elongation_is_rotation_and_scale_invariant(self):
        for scale in (1, 4):
            for angle in (0, 15, 30, 45, 60, 75, 90):
                with self.subTest(scale=scale, angle=angle):
                    contour = cv.boxPoints(((200, 200), (100*scale,20*scale),angle)).astype(np.float32)
                    moments = cv.moments(contour)
                    matrix = [[moments["mu20"],moments["mu11"]],
                              [moments["mu11"],moments["mu02"]]]
                    eigenvalues = np.linalg.eigvalsh(matrix)
                    expected = eigenvalues[0]/eigenvalues[1]
                    scalar = _calc_inertia_ratio(moments)
                    vector = _calc_inertia_ratio(pl.from_dicts([moments]))["inertia_ratio"][0]
                    self.assertAlmostEqual(scalar, expected, places=12)
                    self.assertAlmostEqual(vector, expected, places=12)
                    self.assertAlmostEqual(scalar, .04, places=6)
                    self.assertLess(scalar, .2)

    def test_circle_and_zero_moments(self):
        for moments,expected in (({"mu20":10.,"mu02":10.,"mu11":0.},1.),
                                 ({"mu20":0.,"mu02":0.,"mu11":0.},1.)):
            self.assertEqual(_calc_inertia_ratio(moments),expected)
            self.assertEqual(_calc_inertia_ratio(pl.from_dicts([moments]))["inertia_ratio"][0],expected)

    def test_almost_degenerate_nonnegative_moments_remain_bounded(self):
        moments = {"mu20":100.,"mu02":1e-10,"mu11":0.}
        result = _calc_inertia_ratio(moments)
        self.assertGreaterEqual(result,0)
        self.assertLessEqual(result,1)


if __name__=="__main__":
    unittest.main()
