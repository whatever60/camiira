import sys
import os
import re
from os import path as op
from collections import defaultdict
import yaml
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import cv2 as cv
import biom

IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _parse_image_name(image):
    match = re.fullmatch(
        r"([^_]+)(?:_(d\d+|\d{14}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}))?",
        Path(image).stem,
    )
    if match is None:
        raise ValueError
    barcode, time = match.groups()
    return barcode, int(time.removeprefix("d").replace("-", "").replace("_", "")) if time else -1


def read_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    platform_mapping = {"darwin": "macos", "win32": "mswin10"}

    platform = platform_mapping.get(sys.platform, "mswin10")

    keys_to = [
        "plate_qc_colony_contour_pixel",
        "plate_qc_image_scale_factor",
        "plate_qc_image_width_bias",
        "plate_qc_image_height_bias",
        "plate_qc_confirm_window",
        "plate_qc_text_size_large",
        "plate_qc_text_size_small",
        "plate_qc_text_size_button",
        "colony_qc_image_scale_factor",
        "colony_qc_image_width_bias",
        "colony_qc_image_height_bias",
        "colony_qc_confirm_window",
        "colony_qc_colony_show_bias",
        "colony_qc_colony_window_bias",
        "colony_qc_colony_contour_pixel",
        "colony_qc_colony_label_size",
        "colony_qc_colony_label_thickness",
        "colony_qc_text_size_large",
        "colony_qc_text_size_mid",
        "colony_qc_text_size_small",
        "colony_qc_text_size_button",
    ]
    keys_from = []
    for k in keys_to:
        tmp = k.split("_")
        tmp.insert(2, platform)
        keys_from.append("_".join(tmp))
    ret = config.copy()
    for key_from, key_to in zip(keys_from, keys_to):
        if key_from in config:
            ret[key_to] = ret[key_from]
            del ret[key_from]
    return ret


def parse_dir_for_time_series(input_dir, ext: str | tuple = "png") -> dict[str, dict[int, str]]:
    """
    Parses a directory containing time series images.

    File names may be either ``{barcode}_d{time_point}.png`` or ``{barcode}.png``.
    Time points must be non-negative integers. An image without the ``_dN`` suffix is
    assigned time point ``-1``, so it naturally participates in ``min``/``max``
    selection. Barcodes must not contain underscores.

    Args:
        input_dir (str): The path to the input directory.

    Returns:
        dict[str, dict[int, str]]: A dictionary containing the image labels as keys
            and a dictionary containing the time points as keys and the file paths as
            values.
    """
    if not op.isdir(input_dir):
        raise ValueError(f"Directory {input_dir} not found.")
    image_time_series = defaultdict(dict)
    suffix = ext if isinstance(ext, tuple) else (ext if ext.startswith((".", "_")) else f".{ext}")
    images = sorted(p for p in Path(input_dir).iterdir()
                    if p.is_file() and p.name.lower().endswith(suffix))
    for image_path in images:
        image = str(image_path)
        if isinstance(ext, tuple):
            barcode, time_point = _parse_image_name(image)
        else:
            filename = op.basename(image)
            parts = filename[: -len(suffix)].split("_")
            if len(parts) == 1:
                barcode, time_point = parts[0], -1
            else:
                barcode, time_label = parts
                if not barcode or not time_label.startswith("d"):
                    raise ValueError
                time_point = int(time_label[1:])
                if time_point < 0:
                    raise ValueError
        if time_point in image_time_series[barcode]:
            previous = image_time_series[barcode][time_point]
            raise ValueError(
                f"Multiple images map to barcode {barcode!r} and time point "
                f"{time_point}: {previous!r}, {image!r}."
            )
        image_time_series[barcode][time_point] = image
    return image_time_series


def _get_time_points(
    input_dir: str,
    time: int | str | dict[str, int | str] = "default",
    ext: str | tuple = "png",
    missing_tp: str = "silence",
) -> tuple[list[str], list[str]]:
    """
    Selects one image per barcode from a directory of optional time-series images.

    Args:
        input_dir: Directory containing images named ``{barcode}.png`` or
            ``{barcode}_d{time_point}.png``.
        time: Time point ``-1`` (the unsuffixed image), a non-negative time point,
            ``min``, ``max``, or a per-barcode mapping of those selectors. ``default``
            is retained as a compatibility alias that prefers the unsuffixed image.
        ext: Image filename extension.
        missing_tp: Behavior when an explicitly requested integer time point is absent.

    Returns:
        Two lists: image labels and their selected image paths.
    """
    image_label_list = []
    image_list = []
    ts_dict = parse_dir_for_time_series(input_dir, ext=ext)

    if not ts_dict:
        raise ValueError(f"No {ext} images found in {input_dir}.")

    for barcode, time_points in ts_dict.items():
        if isinstance(time, dict):
            if barcode not in time:
                raise ValueError(f"No time selector provided for barcode {barcode}.")
            time_b = time[barcode]
        else:
            time_b = time

        if isinstance(time_b, str) and re.fullmatch(r"-?\d+", time_b):
            time_b = int(time_b)

        if isinstance(time_b, int):
            if time_b < -1:
                raise ValueError("Time point must be -1 or a non-negative integer.")
            if time_b in time_points:
                image_label_list.append(barcode)
                image_list.append(time_points[time_b])
            else:
                if missing_tp == "raise":
                    raise ValueError(
                        f"Time point {time_b} not found for barcode {barcode} in {input_dir}."
                    )
                elif missing_tp == "print":
                    print(
                        f"Time point {time_b} not found for barcode {barcode} in {input_dir}."
                    )
                elif missing_tp == "silence":
                    pass
                else:
                    raise ValueError(
                        f"Invalid missing_tp argument {missing_tp}. "
                        "Must be 'raise', 'print', or 'silence'."
                    )
        else:
            if time_b not in {"default", "min", "max"}:
                raise ValueError(
                    f"Invalid time argument {time_b}. Must be -1, a non-negative "
                    "integer, 'default', 'min', or 'max'."
                )

            if time_b == "default" and -1 in time_points:
                selected_time = -1
            else:
                selector = max if time_b in {"default", "max"} else min
                selected_time = selector(time_points)

            image_label_list.append(barcode)
            image_list.append(time_points[selected_time])

    return image_label_list, image_list


def read_file_list(
    input_dir: str, time: int | str = "default", ext: str | tuple = "png"
) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a list of image files from a directory, ensuring consistency between red and
        white RGB images, and returns the file paths. It provides detailed information
        on any inconsistencies found between the labels in the red_rgb and white_rgb
        directories.

    Args:
        input_dir (str): The directory from which to read images.
        time (int | str): The time point or criteria for image selection.

    Returns:
        tuple[list[str], list[str], list[str]]: Lists of image labels, transmission,
            and epifluorescence image paths.
    """
    if not op.isdir(input_dir):
        raise ValueError(f"Directory {input_dir} not found.")
    red_labels, red_images = _get_time_points(f"{input_dir}/red_rgb", time, ext=ext)
    white_labels, white_images = _get_time_points(f"{input_dir}/white_rgb", time, ext=ext)

    # Determine the differences between red and white labels
    missing_in_red = set(white_labels) - set(red_labels)
    missing_in_white = set(red_labels) - set(white_labels)

    if missing_in_red or missing_in_white:
        error_messages = []
        if missing_in_red:
            error_messages.append(
                f"{len(missing_in_red)} labels found in white_rgb but not in red_rgb: "
                f"{', '.join(sorted(missing_in_red))}"
            )
        if missing_in_white:
            error_messages.append(
                f"{len(missing_in_white)} labels found in red_rgb but not in white_rgb: "
                f"{', '.join(sorted(missing_in_white))}"
            )
        raise ValueError(
            "Inconsistent labels between red_rgb and white_rgb directories: "
            + "; ".join(error_messages)
        )

    if not red_labels:
        raise ValueError(
            f"No paired red_rgb and white_rgb images matched time selector {time!r} "
            f"in {input_dir}."
        )

    white_by_label = dict(zip(white_labels, white_images))
    return red_labels, red_images, [white_by_label[label] for label in red_labels]


def resolve_plate_inputs(input_path, mode, white_path=None, time_label="max"):
    """Resolve only declared single inputs or the two supported batch layouts."""
    source = Path(input_path)
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"Input path {input_path} is neither a file nor a directory.")
    if mode not in {"dual", "epi-only", "auto"}:
        raise ValueError(f"Unknown detection mode: {mode}")
    if mode == "auto":
        if source.is_file():
            raise ValueError("Single plates require --mode dual or --mode epi-only.")
        if (source / "red_rgb").is_dir():
            mode = "dual"
        elif (source / "white_rgb").is_dir():
            mode = "epi-only"
        else:
            raise ValueError("Unrecognized layout; specify a single-plate mode or provide white_rgb.")
    if white_path is not None and (mode != "dual" or source.is_dir()):
        raise ValueError("--white_path is only for a dual single plate.")
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image: {source}")
        label = _parse_image_name(source)[0]
        if mode == "epi-only":
            return mode, {label: {"epi_path": str(source), "trans_path": None}}
        if white_path is None or not Path(white_path).is_file():
            raise ValueError("Dual single-plate detection requires --white_path.")
        if Path(white_path).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image: {white_path}")
        if _parse_image_name(white_path)[0] != label:
            raise ValueError("Red and white images must have matching barcodes.")
        if source.resolve() == Path(white_path).resolve():
            raise ValueError("Dual mode requires two distinct images.")
        return mode, {label: {"epi_path": str(white_path), "trans_path": str(source)}}
    required = {"white_rgb"} | ({"red_rgb"} if mode == "dual" else set())
    allowed = required | {name.replace("rgb", "grayscale") for name in required}
    directories = {p.name for p in source.iterdir() if p.is_dir()}
    if not required <= directories or directories - allowed or any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in source.iterdir()
    ):
        raise ValueError(f"{mode} batch requires {', '.join(sorted(required))}; no other image layout is accepted.")
    if any(p.is_dir() for name in required for p in (source / name).iterdir()):
        raise ValueError("Image folders must be flat.")
    if mode == "dual":
        labels, red, white = read_file_list(str(source), time_label, ext=IMAGE_EXTENSIONS)
    else:
        labels, white = _get_time_points(str(source / "white_rgb"), time_label, ext=IMAGE_EXTENSIONS)
        red = [None] * len(labels)
    if not labels:
        raise ValueError(f"No plates selected in {input_path}.")
    return mode, {b: {"epi_path": w, "trans_path": r} for b, r, w in zip(labels, red, white)}


def write_plate_inputs(output_dir, image_dir, mode, plates):
    """Store paths relative to the root later supplied as --image_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    relative = {b: {k: os.path.relpath(p, image_dir) if p is not None else None
                    for k, p in images.items()} for b, images in plates.items()}
    with open(Path(output_dir) / "inputs.json", "w") as stream:
        json.dump({"mode": mode, "plates": relative}, stream, indent=2)


def read_plate_inputs(data_dir, image_dir):
    """Read an existing selection, never infer its mode or select new time points."""
    with open(Path(data_dir) / "inputs.json") as stream:
        inputs = json.load(stream)
    mode, plates = inputs["mode"], inputs["plates"]
    if mode not in {"dual", "epi-only"} or not plates:
        raise ValueError("Invalid inputs.json.")
    resolved = {}
    for barcode, images in plates.items():
        if set(images) != {"epi_path", "trans_path"} or not images["epi_path"] or (
            (images["trans_path"] is not None) != (mode == "dual")
        ):
            raise ValueError(f"Invalid inputs.json entry for {barcode}.")
        resolved[barcode] = {k: os.path.normpath(Path(image_dir) / p) if p is not None else None
                             for k, p in images.items()}
        for path in resolved[barcode].values():
            if path is not None and not Path(path).is_file():
                raise ValueError(f"Input image not found: {path}")
            if path is not None and (Path(path).suffix.lower() not in IMAGE_EXTENSIONS
                                     or _parse_image_name(path)[0] != barcode):
                raise ValueError(f"Input image does not match barcode {barcode}: {path}")
    return resolved


# def read_file_list(
#     input_dir: str, time: int | str = "max"
# ) -> tuple[list[str], list[str], list[str]]:
#     """
#     Reads a list of image files from a directory and returns the file paths.

#     Args:
#         input_dir (str): The path to the input directory.
#         time (int | str): The time point to read images from if int. Otherwise, must be
#             "min" or "max", in which case the minimum or maximum time point will be
#             selected. By default we select the maximum time point.

#     Returns:
#         tuple[int, list[str], list[str], list[str]]: A tuple containing the total number
#             of images, a list of image labels, a list of transmission image file paths,
#             and a list of epifluorescence image file paths.
#     """
#     image_label_list = []
#     image_trans_list = []
#     image_epi_list = []
#     rgb_red_dict = parse_dir_for_time_series(f"{input_dir}/red_rgb")
#     rgb_white_dict = parse_dir_for_time_series(f"{input_dir}/white_rgb")
#     # take the union of all barcodes in the two dictionaries and raise error during loop
#     # if not all barcodes are present in both dictionaries
#     for barcode in set(rgb_red_dict.keys()) | set(rgb_white_dict.keys()):
#         rgb_red_b = rgb_red_dict.get(barcode, {})
#         rgb_white_b = rgb_white_dict.get(barcode, {})
#         if not rgb_red_b:
#             raise ValueError(f"Barcode {barcode} not found in rgb_red directory.")
#         if not rgb_white_b:
#             raise ValueError(f"Barcode {barcode} not found in rgb_white directory.")
#         if isinstance(time, int):
#             time_point_red = time
#             time_point_white = time
#         else:
#             if time == "min":
#                 func = min
#             elif time == "max":
#                 func = max
#             else:
#                 raise ValueError(
#                     f"Invalid time argument {time}. Must be int, 'min', or 'max'."
#                 )
#             time_point_red = func(rgb_red_b.keys())
#             time_point_white = func(rgb_white_b.keys())
#         # if -1 is one of the time points, use it as the default
#         if -1 in rgb_red_b:
#             time_point_red = -1
#         if -1 in rgb_white_b:
#             time_point_white = -1
#         image_label_list.append(barcode)
#         image_trans_list.append(rgb_red_b[time_point_red])
#         image_epi_list.append(rgb_white_b[time_point_white])
#     return image_label_list, image_trans_list, image_epi_list


def read_file_list_(input_dir: str) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a list of image files from a directory and returns the file paths.

    DEPRECATED:
        We used to store all the images in a single directory, using file names like
        "{barcode}_{condition}.png", where condition indicates light condition and
        channel. Now we store conditions in separate directories, and incorporate
        time series label into the file names.

    Args:
        input_dir (str): The path to the input directory.

    Returns:
        tuple[int, list[str], list[str], list[str]]: A tuple containing the total number
            of images, a list of image labels, a list of transmission image file paths,
            and a list of epifluorescence image file paths.
    """
    image_label_list = []
    image_trans_list = []
    image_epi_list = []
    for image in sorted(glob.glob(f"{input_dir}/*_rgb_red.png")):
        # red light rgb image
        barcode = op.splitext(op.basename(image))[0].split("_")[0]
        image_label_list.append(barcode)
        image_trans_list.append(image)
    for image in sorted(glob.glob(f"{input_dir}/*_rgb_white.png")):
        # white light rgb image
        image_epi_list.append(image)
    return image_label_list, image_trans_list, image_epi_list


def add_contours(
    image: np.ndarray,
    contours: list[np.ndarray],
    centers: np.ndarray,  # [num_contours, 2]
    contour_pixel: int,
    center_pixel: int,
    border_color: tuple[int, int, int] = (0, 0, 0),
    center_color: tuple[int, int, int] = (0, 255, 0),
    annot_index: bool = False,
) -> np.ndarray:
    if image.ndim == 2:
        image_contours = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    else:
        image_contours = image.copy()

    cv.drawContours(image_contours, contours, -1, border_color, contour_pixel).astype(
        np.uint8
    )

    for idx, c in enumerate(np.round(centers).astype(int)):
        cv.circle(image_contours, tuple(c), center_pixel, center_color, -1)
        if annot_index:
            cv.putText(
                image_contours,
                str(idx + 1),
                tuple(c),
                cv.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv.LINE_AA,
            )

    return image_contours


def _coco_to_contours(coco: dict, category_id: int = None) -> list[np.ndarray]:
    return [
        np.array(anno["segmentation"][0]).reshape(-1, 1, 2).astype(np.int32)
        for anno in coco["annotations"]
        if category_id is None or anno["category_id"] == category_id
    ]


def read_table(
    table_path: str,
    index_col: str | int = 0,
    comment: str = None,
    dtype: str = "int",
) -> pd.DataFrame:
    """Read a table from a file and return it as a DataFrame.

    Args:
        table_path: Path to the table file. Could be a tsv, csv or biom.

    Returns:
        pd.DataFrame: DataFrame containing the table data.
    """
    if table_path.endswith(".biom"):
        df_biom = biom.load_table(table_path)
        df = df_biom.to_dataframe().astype(dtype)
        df.index.name = df_biom.table_id
        return df
    else:
        if table_path.endswith(".csv") or table_path.endswith(".csv.gz"):
            method = pd.read_csv
        elif table_path.endswith(".tsv") or table_path.endswith(".tsv.gz"):
            method = pd.read_table
        else:
            raise ValueError("Unsupported table file format.")
        return method(table_path, index_col=index_col, comment=comment)


def write_table(table: pd.DataFrame, table_path: str) -> None:
    """Write a table to a file.

    Args:
        table: DataFrame to be written to a file.
        table_path: Path to the output file. Could be a tsv, csv or biom.
    """
    if table_path.endswith(".biom"):
        data = biom.Table(
            table.to_numpy(), table.index, table.columns, table_id=table.index.name
        )
        with biom.util.biom_open(table_path, "w") as f:
            data.to_hdf5(f, "whatever60")
    elif table_path.endswith(".csv") or table_path.endswith(".csv.gz"):
        table.to_csv(table_path)
    elif table_path.endswith(".tsv") or table_path.endswith(".tsv.gz"):
        table.to_csv(table_path, sep="\t")
    else:
        raise ValueError("Unsupported table file format.")
