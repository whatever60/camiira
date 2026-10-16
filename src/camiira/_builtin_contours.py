"""Private evaluation helper: image-derived contours guided by built-in boxes.

Run with ``python -m camiira._builtin_contours --help``. These are reconstructed
contours, not the robot's unexported native segmentation.
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd


def contour_from_box(image, box, center, padding=4):
    """Local Otsu threshold, retaining the component at the supplied center."""
    box = np.asarray(box, dtype=float)
    center = np.asarray(center, dtype=float)
    if not np.all(np.isfinite(np.r_[box, center])) or np.any(box[2:] <= box[:2]):
        return None, "invalid_box"
    height, width = image.shape[:2]
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int) + 1
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        return None, "outside_image"
    if not (x0 <= center[0] < x1 and y0 <= center[1] < y1):
        return None, "center_outside_box"
    left, top = max(0, x0 - padding), max(0, y0 - padding)
    right, bottom = min(width, x1 + padding), min(height, y1 + padding)
    patch = image[top:bottom, left:right].copy()
    if np.ptp(image[y0:y1, x0:x1].astype(float), axis=(0, 1)).max() < 2:
        return None, "low_contrast"
    sx = int(np.clip(round(center[0]) - left, x0-left, x1-left-1))
    sy = int(np.clip(round(center[1]) - top, y0-top, y1-top-1))
    gray = cv.GaussianBlur(cv.cvtColor(patch, cv.COLOR_BGR2GRAY), (3, 3), 0)
    border = np.r_[gray[0], gray[-1], gray[:, 0], gray[:, -1]]
    seed = gray[max(0, sy-1):sy+2, max(0, sx-1):sx+2]
    contrast = float(np.median(seed)) - float(np.median(border))
    if abs(contrast) < 2:
        return None, "low_contrast"
    polarity = cv.THRESH_BINARY if contrast > 0 else cv.THRESH_BINARY_INV
    _, foreground = cv.threshold(gray, 0, 1, polarity | cv.THRESH_OTSU)
    # Restrict each recovered object to its source box. Nearby colonies outside
    # the box must not be merged into this one.
    allowed = np.zeros(gray.shape, np.uint8)
    allowed[y0-top:y1-top, x0-left:x1-left] = 1
    foreground &= allowed
    _, components = cv.connectedComponents(foreground)
    target = components[sy, sx]
    if target == 0:
        return None, "no_center_component"
    contours, _ = cv.findContours(
        (components == target).astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv.contourArea)
    if cv.contourArea(contour) <= 0:
        return None, "degenerate_contour"
    return contour + np.array([left, top]), "ok"


def reconstruct_builtin_contours(input_path, csv_path, output_dir, units_per_pixel):
    """Preserve every built-in ID and record failures instead of inventing masks."""
    if not np.isfinite(units_per_pixel) or units_per_pixel <= 0:
        raise ValueError("units_per_pixel must be finite and positive.")
    image = cv.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    source = pd.read_csv(csv_path)
    required = {"Id", "CentroidX", "CentroidY", "BBoxMinX", "BBoxMinY", "BBoxMaxX", "BBoxMaxY"}
    if not required.issubset(source.columns):
        raise ValueError(f"Missing built-in CSV fields: {sorted(required - set(source.columns))}")
    if source.Id.duplicated().any():
        raise ValueError("Built-in IDs must be unique.")
    annotations = []
    overlay = image.copy()
    for row in source.itertuples(index=False):
        box = np.array([row.BBoxMinX, row.BBoxMinY, row.BBoxMaxX, row.BBoxMaxY]) / units_per_pixel
        center = np.array([row.CentroidX, row.CentroidY]) / units_per_pixel
        contour, status = contour_from_box(image, box, center)
        annotation = {
            "id": int(row.Id), "image_id": 1, "category_id": 1, "iscrowd": 0,
            "source_bbox": [*box[:2].tolist(), *(box[2:] - box[:2]).tolist()],
            "source_center": center.tolist(), "segmentation": [],
            "status": status, "area": None, "bbox": None,
        }
        if contour is not None:
            annotation.update(
                segmentation=[contour.flatten().tolist()],
                area=float(cv.contourArea(contour)), bbox=list(cv.boundingRect(contour)),
            )
            cv.drawContours(overlay, [contour], -1, (0, 170, 255), 1)
        annotations.append(annotation)
    result = {
        "info": {
            "contour_source": "bbox-guided image resegmentation (local Otsu), not native built-in masks",
            "source_csv": str(csv_path), "units_per_pixel": float(units_per_pixel),
            "coordinate_assumption": "original-image origin, same axes",
        },
        "images": [{"id": 1, "file_name": Path(input_path).name,
                    "width": image.shape[1], "height": image.shape[0]}],
        "categories": [{"id": 1, "name": "colony"}], "annotations": annotations,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "builtin_reconstructed.json").write_text(json.dumps(result), encoding="utf-8")
    if not cv.imwrite(str(destination / "builtin_reconstructed.jpg"), overlay):
        raise OSError(f"Could not write contour overlay to {destination}.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--units_per_pixel", required=True, type=float)
    args = parser.parse_args()
    result = reconstruct_builtin_contours(**vars(args))
    success = sum(a["status"] == "ok" for a in result["annotations"])
    print(f"Reconstructed {success}/{len(result['annotations'])} built-in contours.")


if __name__ == "__main__":
    main()
