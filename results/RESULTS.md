# Experiment results (re-run)

Generated 2026-09-27 03:50 UTC with Python 3.11.15, scikit-learn 1.8.0.

Task: forecast the next calendar day's close using only information available at the
end of the current day. Split: chronological; last 20% of usable rows = hold-out test.
`MAE/Pers.` < 1 means the model beat the persistence baseline (tomorrow = today).
DM p-value: Diebold-Mariano test of squared errors vs persistence (two-sided).
Directional accuracy excludes days where the model predicts no move (n/a for persistence).

## Bitcoin

Train 2013-06-03 → 2024-04-02 (3060 days); test 2024-04-03 → 2026-05-07 (765 days).

| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 1459.5460 | 2023.2220 | 1.75 | 1.000 | n/a | n/a | n/a | 1.000 (0/5) |
| Drift | 1470.7536 | 2032.3802 | 1.77 | 1.008 | 0.502 | 1.34 | 0.180 | 1.004 (1/5) |
| Ridge[candle] | 1472.8855 | 2041.5985 | 1.77 | 1.009 | 0.493 | 2.55 | 0.011 | 1.006 (1/5) |
| Ridge[candle_lags] | 1478.3109 | 2042.9326 | 1.78 | 1.013 | 0.480 | 2.70 | 0.007 | 1.010 (0/5) |
| HistGradientBoosting[candle_lags] | 1491.5690 | 2037.9125 | 1.79 | 1.022 | 0.515 | 0.72 | 0.474 | 1.067 (0/5) |
| RandomForest[candle_lags] | 1501.3835 | 2044.1452 | 1.80 | 1.029 | 0.489 | 1.33 | 0.185 | 1.041 (0/5) |
| RandomForest[candle] | 1503.6495 | 2047.4374 | 1.81 | 1.030 | 0.489 | 1.38 | 0.167 | 1.040 (0/5) |
| HistGradientBoosting[candle] | 1519.6657 | 2075.5380 | 1.83 | 1.041 | 0.494 | 2.17 | 0.030 | 1.058 (0/5) |
| LinearRegression[legacy_level] | 1575.9985 | 2136.1517 | 1.89 | 1.080 | 0.529 | 2.91 | 0.004 | 1.087 (1/5) |
| RandomForest[legacy_level] | 18253.5542 | 24274.4203 | 18.15 | 12.506 | 0.511 | 22.54 | 0.000 | 5.007 (0/5) |
| DecisionTree[legacy_level] | 18394.9843 | 24220.1965 | 18.39 | 12.603 | 0.502 | 22.75 | 0.000 | 6.121 (0/5) |

Best learned model on hold-out: **Ridge[candle]** (MAE/Pers. 1.009). No learned model beats persistence on hold-out MAE.

## Ethereum

Train 2015-09-05 → 2024-09-13 (2400 days); test 2024-09-14 → 2026-05-07 (601 days).

| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 76.4038 | 110.1759 | 2.64 | 1.000 | n/a | n/a | n/a | 1.000 (0/5) |
| Drift | 76.9462 | 110.6958 | 2.66 | 1.007 | 0.509 | 1.29 | 0.199 | 1.007 (1/5) |
| Ridge[candle] | 77.2521 | 110.9361 | 2.68 | 1.011 | 0.469 | 1.63 | 0.105 | 1.014 (0/5) |
| Ridge[candle_lags] | 77.9332 | 111.3154 | 2.69 | 1.020 | 0.461 | 2.13 | 0.034 | 1.021 (0/5) |
| RandomForest[candle] | 78.2977 | 112.1654 | 2.71 | 1.025 | 0.484 | 2.34 | 0.020 | 1.034 (0/5) |
| RandomForest[candle_lags] | 78.4041 | 111.7385 | 2.71 | 1.026 | 0.488 | 1.79 | 0.074 | 1.034 (0/5) |
| HistGradientBoosting[candle_lags] | 80.0423 | 113.8816 | 2.78 | 1.048 | 0.484 | 3.13 | 0.002 | 1.055 (0/5) |
| HistGradientBoosting[candle] | 81.9221 | 115.3158 | 2.82 | 1.072 | 0.496 | 3.19 | 0.002 | 1.047 (1/5) |
| LinearRegression[legacy_level] | 83.9649 | 117.2670 | 2.91 | 1.099 | 0.514 | 3.38 | 0.001 | 1.135 (0/5) |
| RandomForest[legacy_level] | 128.0790 | 198.0862 | 4.01 | 1.676 | 0.514 | 7.30 | 0.000 | 3.673 (0/5) |
| DecisionTree[legacy_level] | 147.6222 | 217.3428 | 4.65 | 1.932 | 0.498 | 8.56 | 0.000 | 4.075 (0/5) |

Best learned model on hold-out: **Ridge[candle]** (MAE/Pers. 1.011). No learned model beats persistence on hold-out MAE.

## Litecoin

Train 2013-06-03 → 2024-04-02 (3060 days); test 2024-04-03 → 2026-05-07 (765 days).

| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 2.3647 | 3.7429 | 2.64 | 1.000 | n/a | n/a | n/a | 1.000 (0/5) |
| Drift | 2.3684 | 3.7493 | 2.65 | 1.002 | 0.502 | 1.14 | 0.255 | 1.001 (2/5) |
| Ridge[candle] | 2.4178 | 3.8178 | 2.69 | 1.022 | 0.482 | 2.36 | 0.018 | 1.018 (1/5) |
| RandomForest[candle] | 2.4208 | 3.7847 | 2.70 | 1.024 | 0.508 | 1.44 | 0.151 | 1.027 (0/5) |
| RandomForest[candle_lags] | 2.4247 | 3.8152 | 2.70 | 1.025 | 0.485 | 2.11 | 0.035 | 1.028 (0/5) |
| Ridge[candle_lags] | 2.4409 | 3.8456 | 2.72 | 1.032 | 0.467 | 2.85 | 0.005 | 1.026 (0/5) |
| HistGradientBoosting[candle] | 2.4796 | 3.8810 | 2.76 | 1.049 | 0.484 | 2.19 | 0.029 | 1.057 (0/5) |
| HistGradientBoosting[candle_lags] | 2.4817 | 3.9597 | 2.76 | 1.049 | 0.481 | 2.96 | 0.003 | 1.062 (0/5) |
| LinearRegression[legacy_level] | 2.7664 | 4.1730 | 3.08 | 1.170 | 0.520 | 3.99 | 0.000 | 1.167 (0/5) |
| RandomForest[legacy_level] | 3.0225 | 4.6853 | 3.32 | 1.278 | 0.525 | 5.06 | 0.000 | 1.270 (0/5) |
| DecisionTree[legacy_level] | 3.5655 | 5.6351 | 3.97 | 1.508 | 0.499 | 5.09 | 0.000 | 1.578 (0/5) |

Best learned model on hold-out: **Ridge[candle]** (MAE/Pers. 1.022). No learned model beats persistence on hold-out MAE.

## Stellar

Train 2014-09-03 → 2024-07-02 (2694 days); test 2024-07-03 → 2026-05-07 (674 days).

| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge[candle] | 0.0086 | 0.0155 | 3.05 | 0.992 | 0.525 | -1.02 | 0.308 | 0.994 (3/5) |
| Ridge[candle_lags] | 0.0086 | 0.0156 | 3.07 | 0.995 | 0.516 | -0.40 | 0.691 | 0.990 (3/5) |
| Persistence | 0.0087 | 0.0157 | 3.07 | 1.000 | n/a | n/a | n/a | 1.000 (0/5) |
| RandomForest[candle] | 0.0087 | 0.0157 | 3.09 | 1.002 | 0.496 | -0.12 | 0.902 | 1.010 (1/5) |
| Drift | 0.0087 | 0.0157 | 3.08 | 1.004 | 0.475 | 0.73 | 0.463 | 1.004 (1/5) |
| RandomForest[candle_lags] | 0.0088 | 0.0158 | 3.10 | 1.012 | 0.519 | 0.10 | 0.918 | 1.016 (0/5) |
| HistGradientBoosting[candle] | 0.0089 | 0.0159 | 3.14 | 1.025 | 0.491 | 0.43 | 0.665 | 1.041 (0/5) |
| HistGradientBoosting[candle_lags] | 0.0093 | 0.0165 | 3.25 | 1.071 | 0.493 | 1.17 | 0.241 | 1.042 (0/5) |
| LinearRegression[legacy_level] | 0.0098 | 0.0182 | 3.44 | 1.124 | 0.516 | 1.90 | 0.058 | 1.203 (0/5) |
| RandomForest[legacy_level] | 0.0134 | 0.0229 | 4.81 | 1.545 | 0.501 | 4.26 | 0.000 | 1.559 (0/5) |
| DecisionTree[legacy_level] | 0.0161 | 0.0272 | 5.62 | 1.851 | 0.488 | 4.00 | 0.000 | 2.126 (0/5) |

Best learned model on hold-out: **Ridge[candle]** (MAE/Pers. 0.992). It beats persistence on hold-out MAE.

## XRP

Train 2013-09-02 → 2024-04-20 (2987 days); test 2024-04-21 → 2026-05-07 (747 days).

| Model | MAE | RMSE | MAPE % | MAE/Pers. | Dir. acc. | DM stat | DM p | WF MAE/Pers. (folds won) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 0.0516 | 0.0893 | 2.77 | 1.000 | n/a | n/a | n/a | 1.000 (0/5) |
| Drift | 0.0518 | 0.0894 | 2.78 | 1.004 | 0.482 | 0.70 | 0.482 | 1.005 (1/5) |
| Ridge[candle] | 0.0532 | 0.0917 | 2.83 | 1.032 | 0.519 | 2.12 | 0.034 | 1.026 (0/5) |
| RandomForest[candle_lags] | 0.0533 | 0.0909 | 2.86 | 1.032 | 0.491 | 1.11 | 0.267 | 1.048 (0/5) |
| Ridge[candle_lags] | 0.0536 | 0.0922 | 2.84 | 1.038 | 0.521 | 2.44 | 0.015 | 1.032 (0/5) |
| RandomForest[candle] | 0.0536 | 0.0917 | 2.86 | 1.040 | 0.513 | 1.99 | 0.047 | 1.048 (0/5) |
| HistGradientBoosting[candle] | 0.0542 | 0.0927 | 2.89 | 1.051 | 0.493 | 2.74 | 0.006 | 1.107 (0/5) |
| HistGradientBoosting[candle_lags] | 0.0551 | 0.0933 | 2.94 | 1.068 | 0.510 | 1.68 | 0.093 | 1.120 (0/5) |
| LinearRegression[legacy_level] | 0.0574 | 0.0967 | 3.07 | 1.113 | 0.541 | 2.75 | 0.006 | 2.026 (0/5) |
| RandomForest[legacy_level] | 0.1781 | 0.2588 | 8.57 | 3.451 | 0.494 | 15.64 | 0.000 | 1.885 (0/5) |
| DecisionTree[legacy_level] | 0.2076 | 0.3243 | 10.00 | 4.024 | 0.494 | 14.91 | 0.000 | 2.349 (0/5) |

Best learned model on hold-out: **Ridge[candle]** (MAE/Pers. 1.032). No learned model beats persistence on hold-out MAE.
