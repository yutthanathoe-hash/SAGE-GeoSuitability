# SAGE-GeoSuitability V1.0.0

QGIS Processing plugin for **scientific spatial machine-learning risk assessment**. Supports raster predictor layers (including multiband), labelled point/polygon training, optional independent test, spatial cross-validation, model comparison, SHAP diagnostics, raster integrity checks, and binary probability calibration.

## Changes from V1.3.3.7 (development)

- Public release version reset to **V1.0.0**; Processing algorithm id: `sage_geosuitability_v1`.
- Removed **polygon predictor layers** and **polygon mapping CSV** inputs and their slow rasterization/aggregation pipeline. **Training/test polygon samples and study-area AOI polygons remain supported.**
- Kept the existing scientific processing pipeline, default independent-test validation policy (`2`, allow shared raster cells without temporal verification), and SHAP/calibration outputs.
- The package does not require the old polygon predictor helper modules.

## Install

1. Install QGIS (tested by the project on QGIS 4.2.2, Python 3.12).
2. Install required Python packages in **QGIS's Python environment**, not an unrelated system Python: `numpy scipy scikit-learn joblib` (plus `xgboost` and `shap` if those features are used). GDAL and PyQGIS are normally provided by QGIS.
3. Copy the **`sage_geosuitability`** directory into the QGIS Python plugin directory; restart QGIS and enable the plugin in Plugins > Manage and Install Plugins.
4. Open Processing Toolbox > SAGE GeoAI > SAGE-GeoSuitability V1.0.0.
5. For upgrades, remove the **old** `sage_geosuitability` folder first to avoid retaining unused polygon helper files. Back up the old version first.

## Scientific cautions

- `VALIDATION_POLICY=2` allows shared train/test raster cells; **spatial/temporal independence is not established**.
- Never reuse the same labelled source dataset for training and independent test.
- `CALIBRATION=1` performs binary spatial out-of-fold sigmoid calibration; check the calibration report before interpreting risk scores as probabilities.
- SHAP attribution is not causal inference; verify output space and additivity in `SHAP_Audit.json`.
- Check raster integrity warnings (e.g., slope in degrees vs percent) before publication.
- Binary risk score semantics assume event label `1`; multiclass risk maps require additional class-specific review.

## GitHub release checklist

- Create a new GitHub repository (or a clean V1 branch) and commit this source folder and documentation.
- Tag `v1.0.0`; upload the distributable ZIP as a GitHub Release asset.
- Run a fresh QGIS 4.2.2 smoke test and confirm outputs before calling this release fully validated.
- Review and approve the proposed MIT `LICENSE` before public distribution.
- Optionally link GitHub to Zenodo to mint a version-specific DOI.

Author: Yutthana Thoenglom · Maejo University Phrae Campus.

## Release scope and evidence

This is an **initial public software release**, not a claim of universal scientific validation. A successful QGIS 4.2.2 run was reported on 2026-11-02 with seven raster predictors, 5 spatial folds × 3 repeats, an optional independent test, SHAP samples=300 and spatial OOF sigmoid calibration. The example outputs are from one wildfire case study only. They do not validate performance for flood, landslide, or other applications.

See [`examples/README.md`](examples/README.md) for the selected aggregated results. Do not treat example scores as guarantees of generalization.

## License

Proposed MIT license; copyright © 2026 Yutthana Thoenglom. **Repository owner should approve the license before publication.**
