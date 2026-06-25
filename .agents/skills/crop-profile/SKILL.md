---
name: crop-profile
description: Look up FAO-56 crop coefficients (Kc) and growth stage lengths.
---

# Crop Profile Lookup

## Goal
Retrieve the FAO-56 crop coefficients (`Kc_ini`, `Kc_mid`, `Kc_end`) and stage lengths (initial, development, mid-season, late season) for a specified crop.

## Instructions
1. State clearly which crops are supported. Supported crops are: `wheat`, `maize`, `cotton`, `sugarcane`, `tomato`, `chickpea`, `groundnut`.
   - **Sugarcane** is the project's headline crop (cultivated in Maharashtra, India, where it occupies ~4% of the state's cultivated area but consumes ~70% of its irrigation water).
   - **Wheat** is fully supported as the secondary crop.
2. Read the crop profiles from the references file at `references/crop_data.json` relative to this skill. Do not hardcode or reason about these crop coefficients.
3. Look up and return the correct row corresponding to the crop name requested by the user.

## Constraints
- Never make up or hallucinate crop coefficient curves or stage lengths. Always query the references file.
- If a crop is not found, state clearly that it is not currently supported, list the supported crops, and request the user to specify a supported crop.
