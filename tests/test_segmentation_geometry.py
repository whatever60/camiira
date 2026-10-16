import sys
import unittest
import warnings
import json
import tempfile
from pathlib import Path
from unittest import mock

import cv2 as cv
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from camiira.detect_colonies import colony_qc, postprocess_contour, detect_colonies, _detect
from camiira.detector import CAMIIColonyDetector


class SegmentationGeometryTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (ROOT / "test_data/configs/configure_single_bmp_bhi.yaml").read_text()
        )

    def test_internal_dark_stripe_does_not_create_extra_geometric_seeds(self):
        mask = np.zeros((100, 100), np.uint8)
        cv.circle(mask, (50, 50), 20, 255, -1)
        contour = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0][0]
        uniform = np.where(mask, 100, 0).astype(np.uint8)
        striped = uniform.copy()
        striped[:, 49:52] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = postprocess_contour(contour, uniform, self.config)
            second = postprocess_contour(contour, striped, self.config)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        np.testing.assert_array_equal(first[0], second[0])

    def test_zero_area_contours_are_removed_before_centroid_indexing(self):
        image = np.zeros((120, 120), np.uint8)
        valid = np.array([[45, 45], [65, 45], [65, 65], [45, 65]], np.int32).reshape(-1, 1, 2)
        point = np.array([[[40, 40]]], np.int32)
        line = np.array([[[70, 70]], [[80, 80]]], np.int32)
        self.config.update(circularity_threshold=.1, circularity_threshold_very_bad=.1,
                           post_min_circularity=.1, post_min_convexity=.1)
        kept, frame = colony_qc(image, [point, valid, line], self.config)
        self.assertEqual(len(kept), 1)
        self.assertEqual(frame.height, 1)
        np.testing.assert_array_equal(kept[0], valid)
        self.assertTrue(np.isfinite(frame[["center_x", "center_y"]].to_numpy()).all())
        kept, frame = colony_qc(image, [point, line], self.config)
        self.assertEqual(kept, [])
        self.assertEqual(frame.height, 0)

    def test_empty_input_does_not_export_placeholder_with_permissive_filters(self):
        self.config.update(min_size=1, small_size_area=1, segment_bias=0,
                           filter_bias=0, min_dist=0, min_dist_pin=0,
                           circularity_threshold=.1, circularity_threshold_very_bad=.1,
                           post_min_circularity=.1, post_min_convexity=.1,
                           postprocess=False)
        for contours in ([], [np.array([[[40, 40]]], np.int32)]):
            kept, frame = colony_qc(np.zeros((120, 120), np.uint8), contours, self.config)
            self.assertEqual(kept, [])
            self.assertEqual(frame.height, 0)

    def test_existing_contrast_config_inverts_bright_foreground(self):
        dark = np.full((200, 200), 100, np.uint8)
        cv.circle(dark, (100, 100), 20, 30, -1)
        options = dict(fg_gaussian_kernel=3,
                       bg_threshold_block_size=101, bg_threshold_offset=5)
        first = CAMIIColonyDetector(**options).detect(dark)
        second = CAMIIColonyDetector(**options, calib_contrast_alpha=-1,
                                    calib_contrast_beta=255).detect(255-dark)
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_single_and_legacy_detectors_keep_matching_original_steps(self):
        image = np.full((200,200),100,np.uint8)
        cv.circle(image,(100,100),20,30,-1)
        single = CAMIIColonyDetector(skip_clahe=True).detect(image)
        legacy = _detect(image,18,501,18,5,80)
        self.assertEqual(len(single),len(legacy))
        for a,b in zip(single,legacy):
            np.testing.assert_array_equal(a,b)

    def test_canny_percentile_excludes_background_without_random_sampling(self):
        expected = np.percentile(np.arange(10,101,10,dtype=np.uint8),80)
        for single in (True,False):
            for side in (32,64):
                with self.subTest(single=single,side=side):
                    image = np.full((side,side),100,np.uint8)
                    blurred = np.zeros_like(image)
                    blurred.flat[:10] = np.arange(10,101,10,dtype=np.uint8)
                    with mock.patch.object(cv,"GaussianBlur",return_value=blurred), \
                            mock.patch.object(cv,"Canny",wraps=cv.Canny) as canny, \
                            mock.patch.object(np.random,"choice",side_effect=AssertionError("No sampling")):
                        for _ in range(2):
                            if single:
                                CAMIIColonyDetector(skip_clahe=True).detect(image)
                            else:
                                _detect(image,18,101,18,5,80)
                    self.assertEqual(canny.call_count,2)
                    for call in canny.call_args_list:
                        self.assertEqual(call.kwargs["threshold2"],expected)

    def test_empty_foreground_uses_zero_without_computing_percentile(self):
        image = np.zeros((100,100),np.uint8)
        for single in (True,False):
            with self.subTest(single=single), \
                    mock.patch.object(np,"percentile") as percentile, \
                    mock.patch.object(cv,"Canny",wraps=cv.Canny) as canny:
                if single:
                    contours = CAMIIColonyDetector(skip_clahe=True).detect(image)
                else:
                    contours = _detect(image,18,101,18,5,80)
                percentile.assert_not_called()
                self.assertEqual(canny.call_args.kwargs["threshold2"],0)
                self.assertEqual(len(contours),0)

    def test_legacy_percentile_preserves_nonzero_float_intensities(self):
        image = np.zeros((32,32),np.float32)
        blurred = image.copy()
        blurred.flat[:3] = [.25,.5,1.5]
        with mock.patch.object(cv,"GaussianBlur",return_value=blurred), \
                mock.patch.object(cv,"Canny",wraps=cv.Canny) as canny:
            _detect(image,18,101,18,5,80)
        self.assertAlmostEqual(canny.call_args.kwargs["threshold2"],1.1)

    def test_original_threshold_blur_and_fixed_morphology_order(self):
        image = np.full((200,200),100,np.uint8)
        cv.circle(image,(100,100),20,30,-1)
        original_blur, original_threshold = cv.GaussianBlur, cv.adaptiveThreshold
        original_dilate, original_erode = cv.dilate, cv.erode
        for single in (True,False):
            events = []
            def blur(*args,**kwargs):
                kernel = kwargs.get("ksize",args[1] if len(args)>1 else None)
                events.append(("blur",kernel))
                return original_blur(*args,**kwargs)
            def threshold(*args,**kwargs):
                events.append(("threshold",None))
                return original_threshold(*args,**kwargs)
            def dilate(*args,**kwargs):
                events.append(("dilate",kwargs["iterations"]))
                return original_dilate(*args,**kwargs)
            def erode(*args,**kwargs):
                events.append(("erode",kwargs["iterations"]))
                return original_erode(*args,**kwargs)
            with mock.patch.object(cv,"GaussianBlur",side_effect=blur), \
                    mock.patch.object(cv,"adaptiveThreshold",side_effect=threshold), \
                    mock.patch.object(cv,"dilate",side_effect=dilate), \
                    mock.patch.object(cv,"erode",side_effect=erode):
                if single:
                    CAMIIColonyDetector(skip_clahe=True).detect(image)
                else:
                    _detect(image,18,101,5,5,80)
            self.assertEqual(events,[("threshold",None),("blur",(5,5)),
                                      ("dilate",3),("erode",1)])

    def test_existing_algorithm_runs_all_four_acquisition_combinations(self):
        image = np.full((200, 200, 3), 100, np.uint8)
        cv.circle(image, (100, 100), 20, (30, 30, 30), -1)
        self.config.update(fg_gaussian_kernel=3,
                           bg_threshold_block_size=101,bg_threshold_offset=5,
                           crop_x_min=None,crop_x_max=None,crop_y_min=None,crop_y_max=None,
                           calib_contrast_trans_alpha=1,calib_contrast_trans_beta=0,
                           circularity_threshold=.1,circularity_threshold_very_bad=.1,
                           post_min_circularity=.1,post_min_convexity=.1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = root/"config.yaml"
            cfg.write_text(yaml.safe_dump(self.config))
            outputs = []
            for mode in ("dual","epi-only"):
                source = root/mode
                white = source/"white_rgb"/"plate.png"
                white.parent.mkdir(parents=True)
                cv.imwrite(str(white),image)
                red = source/"red_rgb"/"plate.png"
                if mode=="dual":
                    red.parent.mkdir()
                    cv.imwrite(str(red),image)
                for batch in (False,True):
                    target = root/f"{mode}-{batch}"
                    path = source if batch else red if mode=="dual" else white
                    detect_colonies(str(path),str(target),config_path=str(cfg),mode=mode,
                                    white_path=str(white) if mode=="dual" and not batch else None)
                    data = json.loads((target/"plate_annot.json").read_text())["annotations"]
                    self.assertEqual(len(data),1)
                    outputs.append(data[0]["segmentation"])
            self.assertEqual(len(outputs),4)


if __name__ == "__main__":
    unittest.main()
