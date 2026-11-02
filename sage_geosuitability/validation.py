"""Scientific diagnostics for SAGE-GeoSuitability 1.0.0.
Independent test data are used strictly for evaluation, never for fitting.
"""
import csv
import json
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reliability(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError('Reliability requires nonempty aligned labels and scores')
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Scores must be finite and in [0, 1]')
    rows = []
    ece = 0.
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        mask = (p >= lo) & ((p < hi) if k < bins - 1 else (p <= hi))
        n = int(mask.sum())
        conf = float(p[mask].mean()) if n else None
        observed = float(y[mask].mean()) if n else None
        rows.append({'bin': k + 1, 'score_min': lo, 'score_max': hi, 'n': n,
                     'mean_score': conf, 'observed_event_rate': observed})
        if n: ece += n / len(y) * abs(conf - observed)
    eps = 1e-15
    summary = {'n': len(y), 'prevalence': float(y.mean()),
               'brier_score': float(brier_score_loss(y, p)),
               'log_loss': float(log_loss(y, np.clip(p, eps, 1-eps), labels=[0, 1])),
               'ece_10_bins': float(ece), 'ece_is_descriptive': True}
    if len(np.unique(y)) == 2:
        summary['roc_auc'] = float(roc_auc_score(y, p))
    else:
        summary['roc_auc'] = None
    return rows, summary


def risk_levels(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    lv = np.minimum(5, np.floor(np.minimum(p, 1.0) * 5).astype(int) + 1)
    rows = []
    for k in range(1, 6):
        m = lv == k
        n = int(m.sum())
        events = int(y[m].sum())
        rows.append({'risk_level': k, 'score_min': (k-1)/5, 'score_max': k/5,
                     'n_test': n, 'event_n': events, 'non_event_n': n-events,
                     'observed_event_rate': float(events/n) if n else None,
                     'mean_score': float(p[m].mean()) if n else None})
    return rows


def fit_oof_sigmoid(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) != 2: raise ValueError('Binary training labels required')
    if not np.all(np.isfinite(p)): raise ValueError('Incomplete OOF scores')
    z = np.log(np.clip(p, 1e-6, 1-1e-6) / (1-np.clip(p, 1e-6, 1-1e-6)))
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(z.reshape(-1,1), y)
    return calibrator


def apply_sigmoid(calibrator, p):
    p = np.asarray(p, dtype=float)
    z = np.log(np.clip(p, 1e-6, 1-1e-6) / (1-np.clip(p, 1e-6, 1-1e-6)))
    return calibrator.predict_proba(z.reshape(-1,1))[:, 1]


def _base_to_shape(base, values):
    n, _, outputs = values.shape
    b = np.asarray(base, dtype=float)
    if b.ndim == 0: return np.full((n, outputs), float(b))
    if b.ndim == 1:
        if len(b) == outputs: return np.broadcast_to(b[None, :], (n, outputs))
        if outputs == 1 and len(b) == n: return b.reshape(-1, 1)
    if b.shape == (n, outputs): return b
    raise ValueError('Unexpected expected_value shape: %s' % (b.shape,))


def shap_scientific(model, transformed, explainer, explanation, classification, labels, feature_names, original_names):
    """SHAP additivity in model RAW output space; binary 2D values are positive class.
    Never compare a raw-margin explanation against predict_proba.
    """
    raw = np.asarray(explanation.values, dtype=float)
    n, d = transformed.shape
    if raw.ndim == 2:
        if raw.shape != (n,d): raise ValueError('Unexpected SHAP feature dimensions')
        values = raw[:, :, None]
        output_labels = [labels[1] if classification and len(labels)==2 else (labels[0] if classification else 'regression')]
    elif raw.ndim == 3 and raw.shape[:2] == (n,d):
        values = raw
        output_labels = labels if len(labels)==raw.shape[2] else ['output_'+str(i) for i in range(raw.shape[2])]
    else: raise ValueError('Unexpected SHAP output shape %s' % (raw.shape,))
    if len(original_names) != d: raise ValueError('Missing transformed-feature provenance')
    base = _base_to_shape(explanation.base_values, values)
    reconstructed = base + values.sum(axis=1)
    # The SHAP TreeExplainer default is raw output; for binary XGBoost this is log-odds.
    output_space = getattr(explainer, 'model_output', 'raw')
    if not isinstance(output_space, str): output_space = str(output_space)
    target = None
    target_name = None
    if not classification:
        target = np.asarray(model.predict(transformed), dtype=float).reshape(-1,1)
        target_name = 'predict'
    elif output_space == 'probability':
        target = np.asarray(model.predict_proba(transformed), dtype=float)
        target_name = 'predict_proba'
    elif output_space == 'raw':
        if hasattr(model, 'get_booster'):
            import xgboost as xgb
            target = np.asarray(model.get_booster().predict(xgb.DMatrix(transformed), output_margin=True), dtype=float)
            target = target.reshape(-1, 1) if target.ndim == 1 else target
            target_name = 'xgboost_raw_margin'
        elif hasattr(model, 'decision_function'):
            target = np.asarray(model.decision_function(transformed), dtype=float)
            target = target.reshape(-1,1) if target.ndim == 1 else target
            target_name = 'decision_function'
        elif values.shape[2] > 1 and hasattr(model, 'predict_proba'):
            # sklearn tree ensemble classifiers' raw TreeExplainer outputs are class probabilities.
            target = np.asarray(model.predict_proba(transformed), dtype=float)
            target_name = 'predict_proba_sklearn_tree'
    audit = {'explainer': type(explainer).__name__, 'model_output': output_space,
             'target_output': target_name, 'shap_shape': list(raw.shape),
             'output_labels': output_labels, 'additivity_check': 'NOT_VERIFIED',
             'additivity_tolerance': 1e-4, 'limitations': ['SHAP is associative, not causal',
             'Final-model SHAP does not demonstrate fold-wise stability']}
    if target is not None and target.shape == reconstructed.shape:
        diff = np.abs(target - reconstructed)
        audit['additivity_max_abs_error'] = float(np.max(diff))
        audit['additivity_mean_abs_error'] = float(np.mean(diff))
        audit['additivity_check'] = 'PASS' if np.max(diff) <= 1e-4 else 'FAIL'
    else:
        audit['reason'] = 'No trustworthy comparable model output, or incompatible shape: reconstructed %s, target %s' % (reconstructed.shape, None if target is None else target.shape)
    class_rows, group_rows, global_rows, sample_rows = [], [], [], []
    for k, label in enumerate(output_labels):
        for j, fname in enumerate(feature_names):
            v = values[:,j,k]
            class_rows.append({'class_label': label, 'feature': fname, 'original_predictor': original_names[j],
                               'mean_abs_shap': float(np.abs(v).mean()), 'mean_shap': float(v.mean())})
        for orig in dict.fromkeys(original_names):
            idx = [j for j, name in enumerate(original_names) if name == orig]
            grouped = values[:,idx,k].sum(axis=1)  # signed sum preserves additivity
            group_rows.append({'class_label': label, 'predictor': orig, 'n_encoded_features': len(idx),
                               'mean_abs_shap': float(np.abs(grouped).mean()), 'mean_shap': float(grouped.mean())})
        for i in range(n):
            for j, fname in enumerate(feature_names):
                sample_rows.append({'sample_index': i, 'class_label': label, 'feature': fname,
                                    'original_predictor': original_names[j], 'feature_value': float(transformed[i,j]),
                                    'shap_value': float(values[i,j,k])})
    for orig in dict.fromkeys(original_names):
        matches = [r for r in group_rows if r['predictor'] == orig]
        global_rows.append({'predictor': orig, 'mean_abs_shap': float(np.mean([r['mean_abs_shap'] for r in matches]))})
    return audit, class_rows, group_rows, global_rows, sample_rows, values
