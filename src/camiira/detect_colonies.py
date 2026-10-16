#!/usr/bin/env python3

import argparse
import os
import json
import warnings
from functools import singledispatch
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree
import pandas as pd
import polars as pl
import cv2 as cv
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from rich import print as rprint

from .utils import (
    read_config, read_file_list, add_contours, resolve_plate_inputs, write_plate_inputs,
)
from .detector import CAMIIColonyDetector


def _single_detector_options(config: dict) -> dict:
    """Translate a pipeline config into single-image detector options.

    Older RGB configs predate :class:`CAMIIColonyDetector` and therefore do not
    contain every detector-specific key.  Keep those configs usable for a single
    image by applying the detector's established defaults.
    """
    return {
        "bg_gaussian_kernel": config.get("bg_gaussian_kernel", 18),
        "bg_threshold_block_size": config["bg_threshold_block_size"],
        "bg_threshold_offset": config["bg_threshold_offset"],
        "fg_gaussian_kernel": config.get("fg_gaussian_kernel", 5),
        "canny_upper_percentile": config["canny_upper_percentile"],
        "clahe_clip_limit": config.get("clahe_clip_limit", 2.0),
        "clahe_tile_grid_size": config.get("clahe_tile_grid_size", [4, 4]),
        "calib_contrast_alpha": config.get("calib_contrast_alpha", 1),
        "calib_contrast_beta": config.get("calib_contrast_beta", 0),
        "crop_x_min": config["crop_x_min"],
        "crop_x_max": config["crop_x_max"],
        "crop_y_min": config["crop_y_min"],
        "crop_y_max": config["crop_y_max"],
        "skip_clahe": config.get("skip_clahe", False),
    }


def detect_colony_batch(
    input_dir: str,
    output_dir: str | None = None,
    calib_param: str | dict[str, np.ndarray] | None = None,
    config_path: str | None = None,
    time_label: str | int = "max",
    toss_red: bool = False,
) -> list[tuple[pl.DataFrame, list[np.ndarray]]]:
    if config_path is None:
        raise ValueError("config_path is required for batch colony detection.")
    labels, red, white = read_file_list(input_dir, time_label)
    return _detect_colony_pairs(zip(labels, red, white), output_dir, calib_param,
                                read_config(config_path), toss_red)


def _detect_colony_pairs(image_pairs, output_dir, calib_param, config, toss_red=False):
    plates = {b: {"trans_path": r, "epi_path": w} for b, r, w in image_pairs}
    return _detect_plates(plates, output_dir, calib_param, config, toss_red)


def _detect_plates(plates, output_dir, calib_param, config, toss_red=False):
    """One per-plate loop; retain the established detector for each acquisition."""
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    ret = []
    for image_label, paths in plates.items():
        epi_raw = cv.imread(paths["epi_path"], cv.IMREAD_COLOR)
        if epi_raw is None:
            raise ValueError(f"Could not read image: {paths['epi_path']}")
        trans_raw = epi_raw
        image_epi = None
        if paths["trans_path"] is not None:
            trans_raw = cv.imread(paths["trans_path"], 0)
            if trans_raw is None:
                raise ValueError(f"Could not read image: {paths['trans_path']}")
            trans_raw = trans_raw.astype(np.float32)
            epi_raw = epi_raw.astype(np.float32)
            image_trans, image_epi = correct_image(
                config, trans_raw, epi_raw, calib_param, toss_red
            )
            contours = detect_colony(image_trans, config)
        else:
            detector = CAMIIColonyDetector(
                calib_param=calib_param, **_single_detector_options(config)
            )
            contours = detector.detect(epi_raw)
            image_trans = detector.image_corrected_gs
        rprint("Detecting colonies for", image_label)
        contours, df = colony_qc(image_trans, contours, config)
        if image_epi is not None:
            df = pl.concat([
                df, pl.from_dict(_calc_contour_channel_mean_std(contours, image_trans, image_epi))
            ], how="horizontal")
        df = _modify_output_object_colony_detection(df, image_label, config)
        offset = np.array([config["crop_x_min"] or 0, config["crop_y_min"] or 0])
        contours = [cnt + offset for cnt in contours]
        ret.append((df, contours))
        if output_dir is None:
            continue
        centers = df[["center_x", "center_y"]].to_numpy()
        pixel = config["plate_qc_colony_contour_pixel"]
        trans_overlay = add_contours(trans_raw, contours, centers, pixel, pixel)
        epi_overlay = (add_contours(epi_raw, contours, centers, pixel, pixel)
                       if image_epi is not None else None)
        _save_outputs_colony_detection(
            df, contours, image_label, paths["epi_path"],
            trans_overlay, epi_overlay, output_dir,
        )
    return ret


def detect_colonies(
    input_path: str,
    output_dir: str,
    calib_param: str | None = None,
    config_path: str | None = None,
    *,
    mode: str = "auto",
    white_path: str | None = None,
    time_label: str | int = "max",
    toss_red: bool = False,
) -> None:
    """Process dual or epi-only plates, singly or in a strictly laid-out batch."""
    source = Path(input_path)
    mode, plates = resolve_plate_inputs(input_path, mode, white_path, time_label)
    if toss_red and mode != "dual":
        raise ValueError("--toss_red applies only to dual mode.")
    if config_path is None:
        raise ValueError("config_path is required for colony detection.")
    _detect_plates(plates, output_dir, calib_param, read_config(config_path), toss_red)
    write_plate_inputs(output_dir, source.parent if source.is_file() else source, mode, plates)


def detect_colony_single(
    input_path: str,
    output_dir: str,
    calib_param: str | None = None,
    config_path: str | None = None,
) -> None:
    """Compatibility entry point for the historical single-image detector."""
    if config_path is None:
        raise ValueError("config_path is required for single-image colony detection.")
    label = os.path.basename(input_path).split(".")[0].split("_")[0]
    _detect_plates(
        {label: {"epi_path": input_path, "trans_path": None}},
        output_dir if output_dir is not None else os.path.dirname(input_path),
        calib_param, read_config(config_path),
    )


def correct_image(
    config: dict,
    image_trans_raw: np.ndarray | None = None,
    image_epi_raw: np.ndarray | None = None,
    calib_param: str | dict[str, np.ndarray] | None = None,  # path to a npz file
    toss_red: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # load two images and calib data
    if calib_param is None:
        _calib_param = {
            "image_trans_calib": np.array(1),
            "image_epi_calib": np.array(1),
        }
    elif isinstance(calib_param, dict):
        _calib_param = calib_param
    else:
        _calib_param = np.load(calib_param)
    _calib_param: dict[str, np.ndarray]

    # crop images
    crop_x_min = config["crop_x_min"]
    crop_x_max = config["crop_x_max"]
    crop_y_min = config["crop_y_min"]
    crop_y_max = config["crop_y_max"]

    if image_trans_raw is None:
        image_trans_corr = None
    else:
        image_trans = crop(
            image_trans_raw, crop_x_min, crop_x_max, crop_y_min, crop_y_max
        )
        if calib_param is None:
            _calib_param["image_trans_calib"] = np.ones_like(image_trans)
        image_dim = image_trans.shape[:2]
        calib_dim = _calib_param["image_trans_calib"].shape[:2]
        if image_dim != calib_dim:
            rprint(
                "WARNING: Image dimension does not match calibration parameters."
                f"Resizing image from {image_dim} to {calib_dim}."
            )
            image_trans = cv.resize(image_trans, calib_dim[::-1])
        image_trans_corr: np.ndarray = (
            image_trans
            / _calib_param["image_trans_calib"]
            * config["calib_contrast_trans_alpha"]
            + config["calib_contrast_trans_beta"]
        )

    if image_epi_raw is None:
        image_epi_corr = None
    else:
        image_epi = crop(image_epi_raw, crop_x_min, crop_x_max, crop_y_min, crop_y_max)
        if calib_param is None:
            _calib_param["image_epi_calib"] = np.ones_like(image_epi)
        image_dim = image_epi.shape[:2]
        calib_dim = _calib_param["image_epi_calib"].shape[:2]
        if image_dim != calib_dim:
            rprint(
                "WARNING: Image dimension does not match calibration parameters."
                f"Resizing image from {image_dim} to {calib_dim}."
            )
            image_epi = cv.resize(image_epi, calib_dim[::-1])
        image_epi_corr: np.ndarray = image_epi / _calib_param["image_epi_calib"]
        if toss_red:
            image_trans_corr: np.ndarray = cv.cvtColor(
                image_epi.max() - image_epi, cv.COLOR_BGR2GRAY
            )
    return image_trans_corr, image_epi_corr


def _detect_simple(arr: np.ndarray):
    if len(arr.shape) == 3:
        gray = cv.cvtColor(arr, cv.COLOR_BGR2GRAY)
    else:
        gray = arr
    blurred = cv.GaussianBlur(gray, (7, 7), 0).astype(np.uint8)
    edged = cv.Canny(blurred, 50, 150)
    contours, _ = cv.findContours(
        edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt)]
    return filtered_contours


def _detect(
    arr: np.ndarray,
    bg_gaussian_kernel: int,
    bg_threshold_block_size: int,
    bg_threshold_offset: int,
    fg_gaussian_kernel: int,
    canny_upper_percentile: float,
) -> list[np.ndarray]:
    arr_bg_mask = cv.adaptiveThreshold(
        cv.filter2D(
            arr,
            -1,
            np.array([[-1, -1, -1], [-1, bg_gaussian_kernel, -1], [-1, -1, -1]])
            / (bg_gaussian_kernel - 8),
        ).astype(np.uint8),
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        bg_threshold_block_size,
        bg_threshold_offset,
    )

    # gaussian blur
    arr_res_bg = cv.GaussianBlur(
        src=np.where(arr_bg_mask, arr, 0),
        ksize=(fg_gaussian_kernel, fg_gaussian_kernel),
        sigmaX=0,
    )

    # edge detection and contouring
    # Match CAMII's nonzero population without random sampling.
    foreground = arr_res_bg[arr_res_bg != 0]
    threshold2 = (
        np.percentile(foreground, canny_upper_percentile)
        if foreground.size else 0
    )
    contours, _ = cv.findContours(
        cv.erode(
            src=cv.dilate(
                src=cv.Canny(
                    image=arr_res_bg.astype(np.uint8),
                    threshold1=1,
                    threshold2=threshold2,
                ),
                kernel=None,
                iterations=3,
            ),
            kernel=None,
            iterations=1,
        ),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )
    return contours


def detect_colony(image_trans: np.ndarray, config: dict) -> pl.DataFrame:
    if config.get("simple_detect"):
        contours = _detect_simple(image_trans)
    else:
        contours = _detect(
            image_trans,
            bg_gaussian_kernel=18,
            bg_threshold_block_size=config["bg_threshold_block_size"],
            bg_threshold_offset=config["bg_threshold_offset"],
            fg_gaussian_kernel=5,
            canny_upper_percentile=config["canny_upper_percentile"],
        )
    return contours


def colony_qc(
    image: np.ndarray, contours: list[np.ndarray], config: dict
) -> tuple[list[np.ndarray], pl.DataFrame]:
    # Point/line contours have no region or finite centroid. Remove them before
    # calculating moments and constructing the spatial index.
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 0]
    no_cnt_flag = 0
    if not contours:
        contours = [np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).reshape(-1, 1, 2)]
        no_cnt_flag = 1
        rprint("\tNo colony detected even before any filtering.")

    df_contour = _calc_contours_stats(
        contours, image, config["segment_bias"] + config["filter_bias"]
    )
    min_dist_pass, min_dist_pin_pass = _check_dist_criteria(
        df_contour[["center_x", "center_y"]].to_numpy(),
        contours,
        config["min_dist"],
        config["min_dist_pin"],
    )

    # initial filtering
    df_contour = (
        df_contour.with_columns(
            min_dist_pass=pl.Series(min_dist_pass),
            min_dist_pin_pass=pl.Series(min_dist_pin_pass),
        )
        .with_columns(
            pass_initial=pl.col("area").is_between(
                config["min_size"], config["max_size"]
            )
            & pl.col("circularity").is_between(
                config["min_circularity"], config["max_circularity"]
            )
            & pl.col("convexity").is_between(
                config["min_convexity"], config["max_convexity"]
            )
            & pl.col("inertia_ratio").is_between(
                config["min_inertia"], config["max_inertia"]
            )
            & pl.col("min_dist_pass")
            & pl.col("min_dist_pin_pass")
            & (
                (pl.col("circularity") >= config["small_size_circularity"])
                | (pl.col("area") >= config["small_size_area"])
            )
            & ~pl.col("close_to_border")
        )
        .with_columns(
            need_pp=pl.col("pass_initial")
            & (
                (
                    pl.col("circularity").is_between(
                        config["circularity_threshold_very_bad"],
                        config["circularity_threshold"],
                        closed="left",
                    )
                    & pl.col("area").is_between(
                        config["area_segment_min"],
                        config["area_segment_max"],
                    )
                )
                | (pl.col("circularity") < config["circularity_threshold_very_bad"])
            ),
        )
        .with_columns(
            direct_pass=pl.col("pass_initial")
            & ~pl.col("need_pp")
            & (pl.col("circularity") > config["post_min_circularity"])
            & (pl.col("convexity") > config["post_min_convexity"])
        )
    )
    contours_dp = [
        cnt
        for cnt, direct_pass in zip(contours, df_contour["direct_pass"])
        if direct_pass
    ]

    # post-processing
    if config.get("postprocess", True):
        contours_pp = [
            j
            for cnt, need_pp in zip(contours, df_contour["need_pp"])
            if need_pp
            for j in postprocess_contour(cnt, image, config)
            if cv.contourArea(j) > 0
        ]
    else:
        contours_pp = []
    if not no_cnt_flag:
        rprint(f"\t{len(contours)} colonies detected.")
        rprint(f"\t{len(contours_dp)} colonies passed initial filtering.")
        rprint(f"\t{len(contours_pp)} colonies need post-processing.")

    df_contour = df_contour.filter(pl.col("direct_pass"))
    if contours_pp:
        df_contour_pp = _calc_contours_stats(
            contours_pp,
            image,
            config["segment_bias"] + config["filter_bias"],
        ).with_columns(
            post_pass=pl.col("area").is_between(
                config["post_min_size"], config["max_size"]
            )
            & (pl.col("circularity") >= config["post_min_circularity"])
            & (pl.col("convexity") >= config["post_min_convexity"])
            & pl.col("inertia_ratio").is_between(
                config["min_inertia"], config["max_inertia"]
            )
        )
        contours_pp = [
            cnt
            for cnt, post_pass in zip(contours_pp, df_contour_pp["post_pass"])
            if post_pass
        ]
        df_contour = pl.concat(
            [df_contour, df_contour_pp.filter(pl.col("post_pass"))],
            how="diagonal",
        )
        if not no_cnt_flag:
            rprint(f"\t\t{len(contours_pp)} colonies passed post-processing.")
    else:
        df_contour = df_contour.with_columns(
            post_pass=pl.Series([False] * len(df_contour))
        )
    # collect contours
    contours = contours_dp + contours_pp

    rprint(f"\t{len(contours)} colonies passed first filtering.")

    if not contours:
        return contours, df_contour

    min_dist_pass, min_dist_pin_pass = _check_dist_criteria(
        df_contour[["center_x", "center_y"]].to_numpy(),
        contours,
        config["min_dist"],
        config["min_dist_pin"],
    )

    # second filtering
    contours = [
        cnt
        for cnt, min_dist_pass, min_dist_pin_pass in zip(
            contours, min_dist_pass, min_dist_pin_pass
        )
        if min_dist_pass and min_dist_pin_pass
    ]
    df_contour = df_contour.with_columns(
        min_dist_pass=pl.Series(min_dist_pass),
        min_dist_pin_pass=pl.Series(min_dist_pin_pass),
    ).filter(pl.col("min_dist_pass") & pl.col("min_dist_pin_pass"))

    if no_cnt_flag:
        contours = []
        df_contour = df_contour.clear()
    else:
        rprint(f"\t{len(contours)} colonies passed second filtering.")
    return contours, df_contour


def _modify_output_object_colony_detection(
    df: pl.DataFrame, barcode: str, config: dict
) -> pl.DataFrame:
    return df.with_columns(
        center_x=pl.col("center_x") + (config["crop_x_min"] or 0),
        center_y=pl.col("center_y") + (config["crop_y_min"] or 0),
        plate_barcode=pl.lit(barcode),
    ).with_row_index("colony_index")


def _save_outputs_colony_detection(
    df: pl.DataFrame,
    contours: list[np.ndarray],
    barcode: str,
    image_epi_path: str,
    image_trans: np.ndarray,
    image_epi: np.ndarray | None,
    output_dir: str,
    toss_red: bool = False,
) -> None:
    cv.imwrite(f"{output_dir}/{barcode}_gs_red_contour.jpg", image_trans)
    if image_epi is not None:
        cv.imwrite(f"{output_dir}/{barcode}_rgb_white_contour.jpg", image_epi)
    df.write_csv(f"{output_dir}/{barcode}_metadata.csv")
    contour_border_coco_dict = _contours_to_coco(contours)
    # contour_border_coco_dict["images"][0] = {
    #     "id": barcode,
    #     "width": image_trans.shape[1] + config["crop_x_min"],
    #     "height": image_trans.shape[0] + config["crop_y_min"],
    #     "file_name": None,
    # }
    contour_border_coco_dict["images"][0]["file_name"] = (
        f"{barcode}_rgb_white.png" if toss_red else os.path.basename(image_epi_path)
    )
    with open(f"{output_dir}/{barcode}_annot.json", "w") as f:
        json.dump(contour_border_coco_dict, f)

    # num_contours = [len(cnt) for cnt in contours]
    # contour_id = np.repeat(np.arange(len(contours)), num_contours)
    # contours = np.concatenate(contours)
    # df_contour_border = pl.DataFrame(
    #     {"contour_idx": contour_id, "x": contours[:, 0, 0], "y": contours[:, 0, 1]}
    # )
    # df_contour_border.write_csv(f"{output_dir}/{barcode}_contour_border.csv")


def _contours_to_coco(contours: list[np.ndarray]) -> dict:
    """Convert contours to COCO format."""
    if not isinstance(contours, list) or not all(
        isinstance(cnt, np.ndarray) for cnt in contours
    ):
        raise ValueError("Contours must be a list of numpy arrays.")

    if not all(
        len(cnt.shape) == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2
        for cnt in contours
    ):
        raise ValueError("Contours must be a list of shape (n, 1, 2).")

    data = {
        "images": [{"id": 1, "width": None, "height": None, "file_name": None}],
        "annotations": [
            {
                "id": i,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [contour.flatten().tolist()],
                "bbox": contour.min(0)[0].tolist()
                + (contour.max(0) - contour.min(0))[0].tolist(),
                "iscrowd": 0,
            }
            for i, contour in enumerate(contours)
        ],
        "categories": [
            {"id": 1, "name": "picked colony", "supercategory": "colony"},
            {"id": 2, "name": "unpicked colony", "supercategory": "colony"},
            {"id": 3, "name": "excluded colony", "supercategory": "colony"},
        ],
    }
    return data


def crop(image, crop_x_min, crop_x_max, crop_y_min, crop_y_max):
    # replace with extreme values if None
    if crop_x_min is None:
        crop_x_min = 0
    if crop_x_max is None:
        crop_x_max = image.shape[1]
    if crop_y_min is None:
        crop_y_min = 0
    if crop_y_max is None:
        crop_y_max = image.shape[0]
    return image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]


@singledispatch
def _calc_inertia_ratio(arg):
    raise NotImplementedError(f"Unsupported type: {type(arg)}")


@_calc_inertia_ratio.register
def _(moms: dict) -> float:
    # Eigenvalue ratio of the second-central-moment matrix. The discriminant
    # needs the square root of BOTH terms; otherwise elongation depends on
    # contour orientation and image scale.
    trace = moms["mu20"] + moms["mu02"]
    discriminant = np.hypot(moms["mu20"] - moms["mu02"], 2 * moms["mu11"])
    if trace + discriminant <= 0:
        return 1.0
    return float(np.clip((trace - discriminant) / (trace + discriminant), 0, 1))


@_calc_inertia_ratio.register
def _(df_contour: pl.DataFrame) -> pl.DataFrame:
    df_contour = (
        df_contour.with_columns(
            moment_trace=pl.col("mu20") + pl.col("mu02"),
            moment_discriminant=((pl.col("mu20") - pl.col("mu02")) ** 2
                                 + 4 * pl.col("mu11") ** 2).sqrt(),
        )
        .with_columns(
            inertia_ratio=pl.when(pl.col("moment_trace") + pl.col("moment_discriminant") > 0)
            .then(((pl.col("moment_trace") - pl.col("moment_discriminant")) /
                   (pl.col("moment_trace") + pl.col("moment_discriminant"))).clip(0, 1))
            .otherwise(1)
        )
        .drop(["moment_trace", "moment_discriminant"])
    )
    return df_contour


def _check_dist_criteria(
    centers: np.ndarray,
    contours: list[np.ndarray],
    min_dist: float,
    min_dist_pin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Given one contour center, check if the closest contour center and the closest
    contour boundary (must be on a different contour) satisfy the distance criteria.

    That is, for each contour center:
        - We want to find the closest contour center and check the distance;
        - We also want to find all the contour boundaries with max distance
            `min_dist_pin` and check if all such boundaries are on the same contour.

    Algorithmically, we use KDTree.
    """
    tree_center, tree_contour = KDTree(centers), KDTree(np.concatenate(contours)[:, 0])
    dists = tree_center.sparse_distance_matrix(tree_center, max_distance=min_dist)
    dists_pin = tree_center.sparse_distance_matrix(
        tree_contour, max_distance=min_dist_pin
    )
    # mark boundary on the same contour as zero
    idx_contours = np.cumsum([0] + list(map(len, contours)))
    for idx, (start, end) in enumerate(zip(idx_contours[:-1], idx_contours[1:])):
        dists_pin[idx, start:end] = 0
    return (
        ~dists.sum(1).A.flatten().astype(bool),
        ~dists_pin.sum(1).A.flatten().astype(bool),
    )


def _calc_contour_channel_mean_std(
    contours: list[np.ndarray],
    image_gray: np.ndarray,
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculate the mean and std of the pixels in the contour.
    We do this by selecting convex hull of each contour and summing up the pixels
        within. Mean is calculated by dividing the sum by how many pixels are in the
        contour. Standard deviation (actually dispersion) is calculated using the
        formula Var(X) = E(X^2) - E(X)^2.
    Formally, E(X) = sum(X) / n, std(X) = (sum(X^2) / n / E(X)^2 - 1) ** 0.5

    Args:
        contours (list[np.ndarray]): The contours.
        image_gray (np.ndarray): The gray image. Must have the shape (height, width).
        image (np.ndarray): The color image. Must have the shape (height, width, 3).
            Channels are in the order of BGR.

    Returns:
        dict[str, float]: The mean and std of the pixels in the contour.
    """
    num_cnts = len(contours)
    height, width = image_gray.shape
    # [height x width, 4], GrBGR
    image = np.concatenate([image_gray[..., None], image], axis=-1).reshape(-1, 4)
    means, stds = np.empty((2, num_cnts, 4))
    for idx, cnt in enumerate(contours):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.fillConvexPoly(mask, cnt, 1)
        i = image[mask.flatten().astype(bool)]
        means[idx] = i.mean(0)
        stds[idx] = i.std(0)
    stds /= means + 1e-6

    return {
        "Graymean": means[:, 0],
        "Bepimean": means[:, 1],
        "Gepimean": means[:, 2],
        "Repimean": means[:, 3],
        "Graystd": stds[:, 0],
        "Bepistd": stds[:, 1],
        "Gepistd": stds[:, 2],
        "Repistd": stds[:, 3],
    }


def postprocess_contours(
    final_df: pd.DataFrame,
    final_contours: list[np.ndarray],
    image_trans_crop: np.ndarray,
    config: dict,
) -> list[np.ndarray]:
    """For contours identified in the initial detection stage, we decide whether
    post-processing is needed using the following criteria:
    - All contours too close to the image border do not need post-processing.
    - All contours with large circularity do not need post-processing.
    - For the rest of contours:
        - If it has small enough circularity, it needs post-processing.
        - If its circularity is not small enough but has reasonable area, it needs
            post-processing.

    For contours that do not need post-processing, we further check their convexity and
    circularity. If both are large enough, we keep them. Otherwise, we discard them.

    For contours that need post-processing, we zoom in on them and segment them using
    random walker. We keep contours that have large enough are, circularity and convexity.
    """
    raise NotImplementedError


def zoom_in_contours_box(
    contour: np.ndarray, bias: int | float
) -> tuple[np.ndarray, int, int, int, int]:
    x_cor, y_cor = contour[:, 0].T
    x_start = int(x_cor.min() - bias)
    x_end = int(x_cor.max() + bias)
    y_start = int(y_cor.min() - bias)
    y_end = int(y_cor.max() + bias)
    modified_contour = contour - np.array([x_start, y_start])
    return modified_contour, x_start, x_end, y_start, y_end


def _calc_contours_stats(
    contours: list[np.ndarray], image: np.ndarray, bias: float | int
) -> pl.DataFrame:
    """Calculate statistics related to contours."""
    df_contour = (
        pl.from_dicts(list(map(cv.moments, contours)))
        .with_columns(
            perim=pl.Series([cv.arcLength(c, True) for c in contours]),
            hull_area=pl.Series([cv.contourArea(cv.convexHull(c)) for c in contours]),
            close_to_border=pl.Series(
                [close_to_border(c, image, bias) for c in contours]
            ),
        )
        .with_columns(area=pl.col("m00"))
        .with_columns(
            center_x=pl.col("m10") / pl.col("area"),
            center_y=pl.col("m01") / pl.col("area"),
            circularity=4 * np.pi * pl.col("area") / (pl.col("perim") ** 2),
            convexity=pl.col("area") / pl.col("hull_area"),
        )
    )
    df_contour = df_contour.with_columns(
        radius=pl.Series(
            _calc_contours_radius(
                df_contour[["center_x", "center_y"]].to_numpy(), contours
            )
        )
    )
    df_contour = _calc_inertia_ratio(df_contour)
    return df_contour.select(
        pl.all().exclude(r"^m\d\d$").exclude(r"^mu\d\d$").exclude(r"^nu\d\d$")
    )


def _calc_contours_radius(
    centers: np.ndarray, contours: list[np.ndarray]
) -> list[float]:
    """Calculate radius of a contour as the median of the distances of center to points
    on the contours.
    """
    radius = [
        np.median(np.linalg.norm(c - cnt[:, 0], axis=1))
        for c, cnt in zip(centers, contours)
    ]
    return radius


def postprocess_contour(
    contour: np.ndarray, image_trans_crop: np.ndarray, config: dict
) -> list[np.ndarray]:
    ret = []
    modified_contour, x_start, x_end, y_start, y_end = zoom_in_contours_box(
        contour, config["segment_bias"]
    )
    image_sub_contour_binary = np.zeros(
        (y_end - y_start + 1, x_end - x_start + 1), dtype=np.uint8
    )
    cv.drawContours(image_sub_contour_binary, [modified_contour], 0, 255, -1)
    cv.drawContours(image_sub_contour_binary, [modified_contour], 0, 255, 1)
    image_sub_contour = np.where(
        image_sub_contour_binary,
        image_trans_crop[y_start : y_end + 1, x_start : x_end + 1].astype(np.uint8),
        0,
    )
    # Centers describe the geometric footprint, not internal dark pigmentation
    # or zero-valued pixels in an otherwise single colony.
    distance = ndimage.distance_transform_edt(image_sub_contour_binary)
    peak_idx = peak_local_max(
        distance,
        footprint=np.ones(
            (
                config["random_walker_maxi_size"],
                config["random_walker_maxi_size"],
            )
        ),
        labels=image_sub_contour_binary,
    )
    local_maxi = np.zeros_like(distance, dtype=bool)
    local_maxi[tuple(peak_idx.T)] = True
    markers = morphology.label(local_maxi)
    markers[image_sub_contour_binary == 0] = -1
    image_sub_segmentation = random_walker(
        image_sub_contour,
        markers,
        beta=config["random_walker_beta"],
        mode=config["random_walker_method"],
    )
    segment_num = image_sub_segmentation.max()
    offset = np.array([x_start, y_start])
    for j in range(1, segment_num + 1):
        cnts = cv.findContours(
            cv.threshold(
                cv.GaussianBlur(
                    np.where(image_sub_segmentation == j, 255, 0).astype(np.uint8),
                    (3, 3),
                    0,
                ),
                100,
                255,
                cv.THRESH_BINARY,
            )[1],
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )[0]
        if len(cnts) == 0:
            continue
        ret.append(cnts[0] + offset)
    return ret


def close_to_border(contour: np.ndarray, image: np.ndarray, bias: int) -> bool:
    if (
        contour[:, 0, 0].min() < bias
        or contour[:, 0, 1].min() < bias
        or contour[:, 0, 0].max() >= image.shape[1] - bias
        or contour[:, 0, 1].max() >= image.shape[0] - bias
    ):
        return True
    else:
        return False


def main():
    """Main CLI entry point for detect_colonies."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        "--input_path",
        dest="input_path",
        type=str,
        required=True,
        help=(
            "Either one image readable by OpenCV (for example BMP or PNG), or a "
            "transformed-image directory containing red_rgb and white_rgb "
            "subdirectories. Epi-only batches require a white_rgb subdirectory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save output images.",
    )
    parser.add_argument(
        "-b",
        "--calib_param_path",
        type=str,
        default=None,
        help=(
            "Optional path to a calibration .npz file. Omit it to skip calibration."
        ),
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-t",
        "--time_label",
        default="max",
        type=str,
        help=(
            "Image time selector. Unsuffixed images have internal time point -1, so "
            "'min' selects them when present; 'max' (the default) selects the latest "
            "available image. Use -1 or a non-negative integer for an exact time. "
            "'default' retains the original unsuffixed-else-latest behavior."
        ),
    )
    parser.add_argument("--toss_red", action="store_true", help="Toss the red channel.")
    parser.add_argument(
        "--mode", choices=("dual", "epi-only", "auto"), default=None,
        help=(
            "Dual or epi-only inputs. Auto recognizes batch channel folders only. "
            "Omitting mode uses deprecated legacy file/directory dispatch."
        ),
    )
    parser.add_argument(
        "--white_path", default=None,
        help="White epi image paired with a single red transmission image.",
    )

    args = parser.parse_args()
    if args.mode is None and args.white_path is None:
        warnings.warn(
            "Detection without --mode uses a deprecated legacy CLI interface. "
            "Its behavior is unchanged; specify --mode dual or --mode epi-only.",
            FutureWarning,
            stacklevel=2,
        )
    if args.mode is not None or args.white_path is not None:
        detect_colonies(
            args.input_path, args.output_dir, args.calib_param_path, args.config_path,
            mode=args.mode or "auto", white_path=args.white_path,
            time_label=args.time_label, toss_red=args.toss_red,
        )
    elif os.path.isdir(args.input_path):
        detect_colony_batch(
            args.input_path,
            args.output_dir,
            args.calib_param_path,
            args.config_path,
            time_label=args.time_label,
            toss_red=args.toss_red,
        )
    elif os.path.isfile(args.input_path):
        detect_colony_single(
            args.input_path,
            args.output_dir,
            calib_param=args.calib_param_path,
            config_path=args.config_path,
        )
    else:
        raise ValueError(
            f"Input path: {args.input_path} is neither a file nor a directory."
        )


if __name__ == "__main__":
    main()
