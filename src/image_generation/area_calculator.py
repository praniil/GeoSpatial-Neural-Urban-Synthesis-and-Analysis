import numpy as np
from typing import Dict

CLASSES = {
    0: "Background",
    1: "Residential",
    2: "Road",
    3: "River",
    4: "Forest",
    5: "Unused Land",
    6: "Agriculture",
}


def calculate_area_percentages(pred_mask: np.ndarray) -> Dict[int, dict]:
    """Calculate percentage area per class from an index mask.

    Returns a dict keyed by class index with 'name' and 'percentage'.
    """
    total = pred_mask.size
    stats = {}
    for idx, name in CLASSES.items():
        count = int((pred_mask == idx).sum())
        pct = round(float(count) / float(total) * 100.0, 2)
        stats[idx] = {"name": name, "percentage": pct, "pixel_count": count}
    # sort by percentage descending
    stats = dict(sorted(stats.items(), key=lambda x: x[1]["percentage"], reverse=True))
    return stats
