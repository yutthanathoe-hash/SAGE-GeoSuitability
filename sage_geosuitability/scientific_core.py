"""SAGE GeoSuitability V1.2 scientific diagnostics. No QGIS dependency."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
 confusion_matrix, roc_auc_score, average_precision_score, log_loss, brier_score_loss,
 mean_absolute_error, mean_squared_error, r2_score, explained_variance_score)

class FoldFeatureSelector(BaseEstimator, TransformerMixin):
    """Fit ONLY on fold-training rows. Keep categorical columns; drop constant and correlated numeric columns."""
    def __init__(self, categorical=(), correlation=0.95, variance=1e-12):
        self.categorical=categorical; self.correlation=correlation; self.variance=variance
    def fit(self,X,y=None):
        X=np.asarray(X,dtype=float); self.n_features_in_=X.shape[1]
        cats=set(self.categorical); kept=[]; dropped=[]
        for j in range(X.shape[1]):
            if j not in cats and np.nanvar(X[:,j])<=self.variance: dropped.append((j,'near_zero_variance'));continue
            if j not in cats and any(k not in cats and abs(np.corrcoef(X[:,j],X[:,k])[0,1])>=self.correlation for k in kept):
                dropped.append((j,'correlation'));continue
            kept.append(j)
        if not kept: raise ValueError('No predictors remain after fold selection')
        self.selected_=np.array(kept,dtype=int); self.dropped_=dropped
        return self
    def transform(self,X): return np.asarray(X)[:,self.selected_]

def correlation_vif(X,names,cat_idx=()):
    X=np.asarray(X,dtype=float); cat=set(cat_idx); indices=[i for i in range(X.shape[1]) if i not in cat and np.std(X[:,i])>1e-12]
    rows=[]; corr=np.corrcoef(X[:,indices],rowvar=False) if len(indices)>1 else np.eye(len(indices))
    corr=np.atleast_2d(corr)
    for a,i in enumerate(indices):
        other=[j for j in indices if j!=i]; vif=1.0
        if other:
            z=X[:,other]; z=np.column_stack([np.ones(len(X)),z]); target=X[:,i]
            beta=np.linalg.lstsq(z,target,rcond=None)[0]; resid=target-z@beta
            r2=1-np.sum(resid**2)/max(np.sum((target-target.mean())**2),1e-20)
            vif=float(1/max(1-r2,1e-12))
        rows.append({'predictor':names[i],'vif':vif,'max_abs_correlation':float(max([abs(corr[a,b]) for b in range(len(indices)) if b!=a],default=0))})
    return rows,indices,corr

def metrics(y,pred,classification,proba=None,nclasses=None):
    y=np.asarray(y);pred=np.asarray(pred)
    if not classification:
        return {'mae':float(mean_absolute_error(y,pred)),'rmse':float(np.sqrt(mean_squared_error(y,pred))),
          'r2':float(r2_score(y,pred)) if len(y)>1 else None,
          'explained_variance':float(explained_variance_score(y,pred)) if len(y)>1 else None}
    labels=np.arange(nclasses); pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=labels,zero_division=0)
    result={'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
       'macro_f1':float(np.mean(f1)),'weighted_f1':float(np.average(f1,weights=support)) if sum(support) else None,
       'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),
       'per_class':[{'class_id':int(i),'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'support':int(support[i])} for i in labels]}
    if proba is not None:
        try: result['log_loss']=float(log_loss(y,proba,labels=labels))
        except ValueError: result['log_loss']=None
        if nclasses==2 and len(np.unique(y))==2:
            result['roc_auc']=float(roc_auc_score(y,proba[:,1]));result['pr_auc']=float(average_precision_score(y,proba[:,1]));result['brier']=float(brier_score_loss(y,proba[:,1]))
        elif nclasses>2 and len(np.unique(y))==nclasses:
            result['roc_auc_ovr_macro']=float(roc_auc_score(y,proba,multi_class='ovr',average='macro',labels=labels))
    return result

def ood_fit(X,quantile=0.99):
    X=np.asarray(X,float);mu=X.mean(0);sd=np.maximum(X.std(0),1e-9)
    distances=np.max(np.abs((X-mu)/sd),axis=1)
    return mu,sd,float(np.quantile(distances,quantile))

def shap_explain(pipeline,X,feature_names,classes=None,max_samples=150,seed=42):
    """Optional SHAP; model-specific explanation of transformed features; returns audit not causal claims."""
    import shap
    rng=np.random.default_rng(seed); X=np.asarray(X)
    ids=rng.choice(len(X),size=min(max_samples,len(X)),replace=False)
    transformed=pipeline.named_steps['pre'].transform(pipeline.named_steps['select'].transform(X[ids]))
    if hasattr(transformed,'toarray'): transformed=transformed.toarray()
    model=pipeline.named_steps['model']
    if not hasattr(model,'feature_importances_'): raise ValueError('SHAP TreeExplainer requires supported tree model')
    explanation=shap.TreeExplainer(model)(transformed)
    vals=np.asarray(explanation.values)
    return vals,ids,transformed
