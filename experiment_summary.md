# Model Experiment Log (Devansh)

## Quick CV Leaderboard
Here's the results from running CV across all the different models we tried. The physics-informed models are way ahead of standard ML models.

| Model | Mean CV RMSE | Std | High Yield RMSE | Notes |
|---|---|---|---|---|
| PFR + Residual HGB | 8.4842 | 2.5715 | 8.2352 | Best single model, hybrid PFR + HGB |
| Physics ExtraTrees | 8.5843 | 2.8845 | 8.1001 | ExtraTrees on the ODE features |
| Gaussian Process | 8.7276 | 2.7990 | 9.7370 | GP with RBF kernel, solid interpolation |
| Physics HGB Logit | 8.7788 | 2.7928 | 7.4245 | HGB with logit transform on yield |
| Physics HGB | 8.9804 | 2.9348 | 8.1723 | HGB without logit |
| Mechanistic PFR | 10.0989 | 2.6728 | 9.8681 | Just the ODE model |
| Physics MLP | 10.1075 | 3.2777 | 10.5272 | Legacy style MLP |
| ExtraTrees | 14.8091 | 2.8701 | 20.2786 | Baseline without physics features |
| Random Forest | 17.1697 | 3.5155 | 22.0898 | |
| Small MLP | 17.8273 | 4.5596 | 23.0850 | |
| HistGradientBoosting | 17.9230 | 4.7684 | 23.2468 | |
| Polynomial Ridge | 23.8455 | 4.4956 | 26.1857 | |
| Ridge | 26.7520 | 2.2802 | 33.3372 | |

## Trying Ensembles
We blended the top 3 models: PFR + Residual HGB, Physics ExtraTrees, and Gaussian Process.
- Simple average across seeds: ~8.6500 RMSE
- Simplex optimized blend weights: ~8.6870 RMSE
Pretty close, the simple average or optimized blend are both solid. Decided to stick with the ensemble.

## Outlier Check
Tried removing the 10 rows Akash deleted. When we do that, the CV RMSE on the best model actually gets worse by +2.0674!
So dropping those points is definitely hurting the model's ability to generalize, because they aren't noise, they are just high/low yield regions. Keep them in!

## Decision
Selected the ensemble for the final submission. Hard CV estimate is around 8.68.
