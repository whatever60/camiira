#!/usr/bin/env python3

import argparse

import numpy as np
import cv2 as cv

from .utils import read_config, read_file_list


def calib(input_dir: str, output_path: str, config_path: str) -> None:
    """Calculate calibration parameters using calibration images (stored in the
    input_dir)
    """
    config = read_config(config_path)
    image_label_list, image_trans_list, image_epi_list = read_file_list(input_dir)

    # do calibration
    crop_x_min = config["crop_x_min"]
    crop_x_max = config["crop_x_max"]
    crop_y_min = config["crop_y_min"]
    crop_y_max = config["crop_y_max"]
    gaussian_kernel = config["calib_gaussian_kernel"]
    gaussian_iteration = config["calib_gaussian_iteration"]
    images_trans = np.stack(
        [
            _read_and_crop(
                image_path,
                cv.IMREAD_GRAYSCALE,
                crop_x_min,
                crop_x_max,
                crop_y_min,
                crop_y_max,
            )
            for image_path in image_trans_list
        ]
    ).astype(np.float32)
    images_epi = np.stack(
        [
            _read_and_crop(
                image_path,
                cv.IMREAD_COLOR,
                crop_x_min,
                crop_x_max,
                crop_y_min,
                crop_y_max,
            )
            for image_path in image_epi_list
        ]
    ).astype(np.float32)

    image_trans_calib = _calc_calib_bg(
        images_trans, gaussian_kernel, gaussian_iteration
    )
    image_epi_calib = np.stack(
        [
            _calc_calib_bg(i, gaussian_kernel, gaussian_iteration)
            for i in images_epi.transpose(3, 0, 1, 2)
        ],
        axis=-1,
    )
    np.savez(
        output_path,
        image_trans_calib=image_trans_calib,
        image_epi_calib=image_epi_calib,
    )


def crop(image, crop_x_min, crop_x_max, crop_y_min, crop_y_max):
    return image[crop_y_min:crop_y_max, crop_x_min:crop_x_max]


def _read_and_crop(
    image_path: str,
    flags: int,
    crop_x_min: int | None,
    crop_x_max: int | None,
    crop_y_min: int | None,
    crop_y_max: int | None,
) -> np.ndarray:
    image = cv.imread(image_path, flags)
    if image is None:
        raise ValueError(f"Could not read calibration image: {image_path}")
    return crop(image, crop_x_min, crop_x_max, crop_y_min, crop_y_max)


def _calc_calib_bg(
    images: np.ndarray, gaussian_kernel: tuple[int, int], gaussian_iteration: int
) -> np.ndarray:
    gaussian_input = np.mean(images, axis=0)
    for _ in range(gaussian_iteration):
        gaussian_input = cv.GaussianBlur(gaussian_input, gaussian_kernel, 0)
    return gaussian_input / gaussian_input.mean()


def main():
    """Main CLI entry point for calc_calib_params."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Transformed reference-image directory containing red_rgb and white_rgb "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--output_path",
        "--output_dir",
        dest="output_path",
        type=str,
        required=True,
        help=(
            "Path of the calibration .npz file to create. The .npz extension is "
            "optional. --output_dir is retained as a deprecated compatibility alias."
        ),
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )

    args = parser.parse_args()
    calib(args.input_dir, args.output_path, args.config_path)


if __name__ == "__main__":
    main()
