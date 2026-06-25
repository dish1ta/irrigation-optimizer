import json
import os
import sys

# Crop data loader
def load_crop_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    crop_data_path = os.path.abspath(os.path.join(script_dir, "..", "..", "crop-profile", "references", "crop_data.json"))
    if not os.path.exists(crop_data_path):
        raise FileNotFoundError(f"Crop data not found at {crop_data_path}")
    with open(crop_data_path, "r") as f:
        return json.load(f)

try:
    CROP_DATA = load_crop_data()
except Exception as e:
    print(json.dumps({"error": f"Failed to load crop data: {str(e)}"}), file=sys.stderr)
    sys.exit(1)

# Hard safety ceiling: never recommend more than this many mm of irrigation in
# a single day for any crop, regardless of what the calculation produces.
MAX_DAILY_MM = 60

# Liters per mm of depth per hectare (1mm over 1ha = 10 m^3 = 10,000 L).
LITERS_PER_MM_PER_HA = 10_000

def kc_for_day(crop: str, days_after_planting: int) -> float:
    """FAO-56 style crop coefficient curve: flat at Kc_ini through the initial
    stage, linear ramp to Kc_mid through development, flat at Kc_mid through
    mid-season, linear taper to Kc_end through late season."""
    crop_info = CROP_DATA[crop]
    kc_ini = crop_info["Kc_ini"]
    kc_mid = crop_info["Kc_mid"]
    kc_end = crop_info["Kc_end"]
    stages = crop_info["stage_lengths"]
    d_ini, d_dev, d_mid, d_late = stages
    d = days_after_planting

    if d <= d_ini:
        return kc_ini
    if d <= d_ini + d_dev:
        frac = (d - d_ini) / d_dev
        return kc_ini + frac * (kc_mid - kc_ini)
    if d <= d_ini + d_dev + d_mid:
        return kc_mid
    if d <= d_ini + d_dev + d_mid + d_late:
        frac = (d - d_ini - d_dev - d_mid) / d_late
        return kc_mid + frac * (kc_end - kc_mid)
    return kc_end  # past the modeled season -- caller should flag harvest

def effective_rainfall(precip_mm: float) -> float:
    """Simple USDA SCS-style approximation: light rain is mostly usable by the
    crop; heavy rain mostly runs off or drains past the root zone."""
    if precip_mm <= 5:
        return precip_mm * 0.9
    if precip_mm <= 25:
        return 4.5 + (precip_mm - 5) * 0.7
    return 4.5 + 20 * 0.7  # cap -- excess assumed lost to runoff/deep drainage

def build_schedule(crop: str, days_after_planting: int, field_size_ha: float,
                    forecast: list) -> list:
    if crop not in CROP_DATA:
        raise ValueError(f"Unknown crop '{crop}'. Known: {list(CROP_DATA.keys())}")
    if field_size_ha <= 0:
        raise ValueError("field_size_ha must be positive")

    schedule = []
    dap = days_after_planting
    for day in forecast:
        kc = kc_for_day(crop, dap)
        etc_mm = day["et0_mm"] * kc
        eff_rain_mm = effective_rainfall(day["precip_mm"])
        net_mm = max(0.0, etc_mm - eff_rain_mm)
        net_mm = min(net_mm, MAX_DAILY_MM)  # safety ceiling -- never skip this
        liters = net_mm * field_size_ha * LITERS_PER_MM_PER_HA

        schedule.append({
            "date": day["date"],
            "kc": round(kc, 2),
            "etc_mm": round(etc_mm, 2),
            "effective_rain_mm": round(eff_rain_mm, 2),
            "net_irrigation_mm": round(net_mm, 2),
            "liters_needed": round(liters, 1),
        })
        dap += 1
    return schedule

def water_saved_vs_naive(schedule: list, field_size_ha: float,
                          naive_mm_per_day: float = 6.0) -> dict:
    """Compare against a naive 'irrigate a fixed amount every day' baseline."""
    actual_total_mm = sum(d["net_irrigation_mm"] for d in schedule)
    naive_total_mm = naive_mm_per_day * len(schedule)
    saved_mm = max(0.0, naive_total_mm - actual_total_mm)
    saved_liters = saved_mm * field_size_ha * LITERS_PER_MM_PER_HA
    pct_saved = (saved_mm / naive_total_mm * 100) if naive_total_mm else 0.0
    return {
        "naive_total_mm": round(naive_total_mm, 1),
        "actual_total_mm": round(actual_total_mm, 1),
        "saved_liters": round(saved_liters, 1),
        "pct_saved": round(pct_saved, 1),
    }

def main():
    try:
        input_data = json.load(sys.stdin)
    except Exception as e:
        print(json.dumps({"error": f"Invalid JSON input on stdin: {str(e)}"}), file=sys.stderr)
        sys.exit(1)

    required_fields = ["crop", "days_after_planting", "field_size_ha", "forecast"]
    missing = [f for f in required_fields if f not in input_data]
    if missing:
        print(json.dumps({"error": f"Missing required fields: {missing}"}), file=sys.stderr)
        sys.exit(1)

    crop = input_data["crop"]
    days_after_planting = input_data["days_after_planting"]
    field_size_ha = input_data["field_size_ha"]
    forecast = input_data["forecast"]

    if not forecast or not isinstance(forecast, list):
        print(json.dumps({"error": "forecast must be a non-empty list"}), file=sys.stderr)
        sys.exit(1)

    # Validate elements of forecast list
    for idx, day in enumerate(forecast):
        if not isinstance(day, dict):
            print(json.dumps({"error": f"forecast item at index {idx} must be a dictionary"}), file=sys.stderr)
            sys.exit(1)
        required_day_fields = ["date", "et0_mm", "precip_mm"]
        missing_day = [f for f in required_day_fields if f not in day]
        if missing_day:
            print(json.dumps({"error": f"forecast item at index {idx} missing fields: {missing_day}"}), file=sys.stderr)
            sys.exit(1)

    try:
        schedule = build_schedule(crop, days_after_planting, field_size_ha, forecast)
        savings = water_saved_vs_naive(schedule, field_size_ha)

        output = {
            "schedule": schedule,
            "savings": savings
        }
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
