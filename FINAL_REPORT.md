# Reactor Yield Prediction Project Final Report (Devansh)

## Executive Summary & Data Stats
- Had 150 total training rows and 51 rows for the hidden test.
- Raw features are flow, conc, inlet temp, reactor length, and jacket temp.
- Target is yield of B (labeled as overall_yield).
- 37 rows have exactly 0 yield. A lot of points have low yield (< 10) - around 71 rows. Only 18 rows have high yield (> 90).

## Chemistry / Physics Basis
- Non-isothermal continuous flow reactor with A -> B -> C reaction network.
- Used the PFR discussion and material balance design equations to construct a physical surrogate.
- Since we don't have the reactor volume, I used L / F (length over flow rate) as a proxy for exposure/residence time.
- Temperature is super critical - inlet and jacket temp compete. If it gets too hot or exposure is too long, B converts to C (undesired).

## Feature Engineering
We generated a bunch of physics-derived features to help the models:
- Basic: residence time (L/F), temp differences, inverse temps, centered squared temperatures.
- Kinetic exposure features: c * tau, tau / c, dt * tau, etc.
- Arrhenius terms: exp(-E/T) * tau using a small energy grid (3000, 5000, 7000).
- Fitted ODE terms: solved the PFR mass balance equations numerically to fit alpha (heat transfer), activation energies (E1, E2), and reaction orders. Generated effective temperatures, effective rate constants k1 & k2, k1*tau, k2*tau, selectivity ratio k1/k2, and raw physical prediction (y_phys).

## Model Leaderboard (5-Fold CV across 5 seeds)
These are the CV RMSE values:

| Model | CV Mean RMSE | CV Std | High Yield RMSE |
|---|---|---|---|
| PFR + Residual HGB | 8.4842 | 2.5715 | 8.2352 |
| Physics ExtraTrees | 8.5843 | 2.8845 | 8.1001 |
| Gaussian Process | 8.7276 | 2.7990 | 9.7370 |
| Physics HGB Logit | 8.7788 | 2.7928 | 7.4245 |
| Physics HGB | 8.9804 | 2.9348 | 8.1723 |
| Mechanistic PFR | 10.0989 | 2.6728 | 9.8681 |
| Physics MLP | 10.1075 | 3.2777 | 10.5272 |
| ExtraTrees | 14.8091 | 2.8701 | 20.2786 |
| Random Forest | 17.1697 | 3.5155 | 22.0898 |
| Small MLP | 17.8273 | 4.5596 | 23.0850 |
| HistGradientBoosting | 17.9230 | 4.7684 | 23.2468 |
| Polynomial Ridge | 23.8455 | 4.4956 | 26.1857 |
| Ridge | 26.7520 | 2.2802 | 33.3372 |

## Model Selection & Ensemble Strategy
- Best single model: PFR + Residual HGB (8.48 RMSE).
- Decided to build an ensemble of the top 3 models: PFR + Residual HGB, Physics ExtraTrees, and Gaussian Process.
- Evaluated on held-out seeds: Simple average got ~8.65 RMSE, while the simplex optimized weights got ~8.68 RMSE. The simple average is very robust.
- The logit target transform helped a lot at extreme yields.

## Outlier Analysis & Conclusion
- Looked at the 10 rows Akash deleted in his trainer_new.csv.
- They are NOT feature outliers (all Z-scores are under 2). They are just difficult extreme-yield points (e.g. yield > 70 or close to 0).
- If we drop these 10 rows from the training set, the CV RMSE on the best model goes from 8.48 to 10.55. Deleting them is a mistake! Kept them in.
- The final submission predictions are saved to `final_predictions.csv` and `final_submission.csv`.
