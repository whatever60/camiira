"""Small helpers for post-inspection colony replacement."""

from collections import Counter
from collections.abc import Sequence

import numpy as np


def choose_replacement_rows(
    candidate_order: Sequence[int],
    target_rows: Sequence[int],
    target_barcodes: Sequence[str],
    source_barcodes: Sequence[str],
    plate_limits: dict[str, int],
    requested: int,
) -> np.ndarray:
    """Return global row indices for eligible replacement colonies."""
    picked_counts = Counter(source_barcodes)
    selected_rows = []

    for target_position in candidate_order:
        barcode = target_barcodes[target_position]
        limit = plate_limits[barcode]
        if limit > 0 and picked_counts[barcode] >= limit:
            continue

        selected_rows.append(target_rows[target_position])
        picked_counts[barcode] += 1
        if len(selected_rows) == requested:
            break

    return np.asarray(selected_rows, dtype=int)
