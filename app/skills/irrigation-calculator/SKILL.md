---
name: irrigation-calculator
description: Calculate the FAO-56 crop water balance 7-day irrigation schedule and water savings.
---

# Irrigation Schedule Calculator

## Goal
Calculate a 7-day irrigation schedule and water savings relative to a naive baseline by executing a standard calculation script.

## Instructions
1. **CRITICAL**: Do **NOT** attempt to redo or reason about the FAO-56 mathematical formulas (e.g. crop coefficient interpolation, effective rainfall USDA approximations, water balance, conversions to liters, etc.) directly in the prompt.
2. Prepare a JSON payload containing:
   - `crop`: The crop name (e.g., `"sugarcane"`, `"wheat"`).
   - `days_after_planting`: Integer representing the days elapsed since planting.
   - `field_size_ha`: Float representing the field area in hectares.
   - `forecast`: A list of daily forecast dictionaries, each containing:
     - `date`: `"YYYY-MM-DD"`
     - `et0_mm`: Float reference evapotranspiration in mm.
     - `precip_mm`: Float precipitation in mm.
3. Pipe the JSON payload directly via standard input (stdin) to the execution script `scripts/calc_schedule.py` using the shell or command execution tools (e.g., `python scripts/calc_schedule.py`).
4. Read and parse the resulting JSON stdout from the script. The output contains:
   - `schedule`: The 7-day irrigation schedule with daily parameters (`kc`, `etc_mm`, `effective_rain_mm`, `net_irrigation_mm`, and `liters_needed`).
   - `savings`: A comparison baseline of water saved (`naive_total_mm`, `actual_total_mm`, `saved_liters`, `pct_saved`).
5. Present the schedule and savings to the user or requesting node.

## Constraints
- Standard forecast input is **required**. If forecast data is missing, the script will exit with an error. Do not attempt to run calculation without complete forecast data.
- The calculator strictly enforces a daily ceiling of `MAX_DAILY_MM = 60` for net irrigation to prevent overwatering.
- Do not duplicate the Kc table curve definitions; the script loads crop profiles dynamically from the `crop-profile` skill.
