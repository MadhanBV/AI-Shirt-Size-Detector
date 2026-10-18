import json


# ============================================================
# LOAD SIZE CHART
# ============================================================

with open("size_charts.json", "r") as file:
    size_charts = json.load(file)


# ============================================================
# GET AVAILABLE BRANDS
# ============================================================

def get_brands():
    return list(size_charts.keys())


# ============================================================
# GET AVAILABLE FITS
# ============================================================

def get_fits(brand):
    return list(size_charts[brand].keys())


# ============================================================
# PREDICT SIZE
# ============================================================

def predict_shirt_size(
    shoulder_cm,
    chest_width_cm,
    brand="generic",
    fit="regular_fit"
):

    # Check brand
    if brand not in size_charts:
        raise ValueError(
            f"Unknown brand: {brand}"
        )

    # Check fit
    if fit not in size_charts[brand]:
        raise ValueError(
            f"Unknown fit: {fit}"
        )

    chart = size_charts[brand][fit]

    # Check each size
    for size, limits in chart.items():

        if (
            shoulder_cm <= limits["max_shoulder_cm"]
            and
            chest_width_cm <= limits["max_chest_width_cm"]
        ):
            return size

    return "XXL"