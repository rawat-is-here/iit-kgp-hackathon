# Data Check & Audit (Devansh)

## Overview of the Datasets
- `train_dataset.csv`: 150 rows, 5 raw features, and `overall_yield` target.
- `trainer_new.csv` (Akash's clean file): 140 rows.
- `test_dataset.csv`: 51 rows.
- Clean dataset, no missing/null values or duplicate rows in train or test.

## Target Check (overall_yield)
- Mean yield: 36.22
- Median: 15.31
- Std dev: 38.43
- Range: 0.0 to 99.97
- Heavily skewed towards low yield (71 rows under 10 yield, 37 exact zeros). Only 18 rows above 90.

## Raw Correlation with Target
Jacket temp and inlet temp have the strongest negative correlation with yield (around -0.50 and -0.40). Length and flow rate are slightly positive but very low.
- jacket_temperature_K: -0.4984
- inlet_temperature_K: -0.4051
- temp_diff: -0.1292
- length_m: 0.0800
- flow_rate_L_min: 0.0381
- concentration_mol_L: 0.0087

## Checking the 10 "Outliers" Akash Removed
- Akash dropped indices: `[97, 77, 149, 86, 30, 23, 142, 18, 57, 26]`
- Checked Z-scores for all these rows on raw features, exposure proxy, temp diff, and target. None of them have a Z-score greater than 2!
- In terms of yield: 2 are near zero yield, 2 are mid yield (10-30), 5 are high yield (70-90), and 1 is extreme high yield (>90).
- Conclusion: These are not bad data or feature outliers. They are just difficult extreme chemistry regimes. Removing them is model-disagreement filtering, which ruins the model's performance on these boundaries. 

## Dataset Shift / Test Extrapolation
Did a KS test and compared min/max bounds between train and test features.
- Most features align really well (KS p-values are high, no significant shift).
- Test set is almost entirely within the interpolation range of train, only 5 points have mild range excursions.
- So we don't have to worry about extreme extrapolation issues in the hidden test.

## Final Notes
- Re-running test_piml.py on the 140 cleaned rows gives ~12.3 RMSE.
- But if you train on all 150 rows and validate properly, it is much more robust.
- The 10 rows must be kept.
