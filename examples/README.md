# Example outputs — one wildfire case study

These selected aggregated outputs were produced by the user's successful QGIS 4.2.2 run on 2026-11-02 with CALIBRATION=1. No raw training/test points, original geospatial raster datasets, fitted model files, or local file paths are included.

- `CV_Results.csv`: model comparison under spatial cross-validation.
- `Independent_Test_Risk_Levels_{raw,calibrated}.csv`: event frequencies by five risk levels.
- `Independent_Test_Reliability_{raw,calibrated}.csv`: binned probability reliability.
- `SHAP_Global_Importance.csv`, `SHAP_Grouped_Classwise_Importance.csv`: explanatory importance summaries.
- PNG figures: SHAP and calibrated reliability visualizations.

**Caveats:** 7 train/test raster cells overlap; temporal independence was not verified. Slope units require checking. These results do not establish general performance or causality.
