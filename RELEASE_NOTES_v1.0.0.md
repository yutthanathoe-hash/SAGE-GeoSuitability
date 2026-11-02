# SAGE-GeoSuitability v1.0.0 — Initial Public Release

First public QGIS Processing release, derived from the V1.3.3.7 development series.

### Highlights
- Raster-only predictor inputs (single/multiband); polygon predictor mapping removed for simpler workflows.
- Point/polygon labelled training and optional independent test; AOI polygon retained.
- Spatial CV, model comparison, SHAP additivity audit and grouped importance.
- Binary spatial out-of-fold sigmoid probability calibration and risk-level assessment.
- Generic `Risk_Score.tif` and `Risk_Level.tif` outputs.

### Validation and limitations
Successfully executed in QGIS 4.2.2 / Python 3.12 on one wildfire case study. This is **not** universal scientific certification. Default validation policy allows shared raster cells; spatial and temporal independence are not established by that policy. SHAP is explanatory, not causal. Confirm slope units and event-label semantics. See README for details.

### Install
Copy `sage_geosuitability/` into the QGIS Python plugins directory and enable it. See README.md.
