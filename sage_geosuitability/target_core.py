"""SAGE GeoSuitability V1.3 target utilities; standalone, no QGIS dependency."""
import numpy as np

BLOCKED={'id','fid','objectid','sample_id','x','y','x_utm','y_utm','longitude','latitude','year','date','time','geometry','area','perimeter'}
PREFERRED=('target','fire_frequency','burn_count','event_count','event_rate','suitability','class','label','fire','agb_mg_ha','yield')

def choose_target_field(fields, selected='', preferred=None):
    fields=list(fields)
    if selected:
        if selected not in fields: raise ValueError('Selected target field not found: '+selected)
        return selected
    if preferred and preferred in fields: return preferred
    candidates=[f for f in fields if f.lower() not in BLOCKED and not f.lower().endswith('_id')]
    ranked=[f for name in PREFERRED for f in candidates if f.lower()==name]
    if len(ranked)==1: return ranked[0]
    if len(ranked)>1: raise ValueError('Multiple plausible target fields: '+', '.join(ranked)+'; select manually')
    if len(candidates)==1: return candidates[0]
    raise ValueError('Target cannot be selected safely; select field manually from: '+', '.join(candidates))

def infer_target_type(values):
    values=[str(v).strip() for v in values if v is not None and str(v).strip()]
    if not values: raise ValueError('No target values')
    try: a=np.asarray(values,dtype=float)
    except ValueError: return 'classification'
    if not np.all(np.isfinite(a)): raise ValueError('Nonfinite target values')
    # Numeric labels including 0/1 and 1..10 can be ordinal, count or regression.
    return 'ambiguous'

def validate_test_labels(training_labels,test_values):
    return sorted(set(map(str,test_values))-set(map(str,training_labels)))

def build_n_year_target(annual_values,min_valid_years=1,output='count'):
    """0/1/None annual event indicators, shape (samples, years).
    Missing values are not zero; records with insufficient coverage become NaN.
    """
    a=np.asarray(annual_values,dtype=float)
    if a.ndim!=2 or a.shape[1]<1: raise ValueError('Expected samples x years matrix')
    good=np.isfinite(a)
    if np.any(~np.isin(a[good],[0,1])): raise ValueError('Annual events must be 0, 1 or missing')
    n=good.sum(axis=1)
    if min_valid_years<1 or min_valid_years>a.shape[1]: raise ValueError('Invalid min_valid_years')
    counts=np.where(good,a,0).sum(axis=1)
    result=np.full(len(a),np.nan)
    valid=n>=min_valid_years
    if output=='count': result[valid]=counts[valid]
    elif output=='rate': result[valid]=counts[valid]/n[valid]
    else: raise ValueError('output must be count or rate')
    return {'target':result,'event_count':counts,'valid_years':n,'event_rate':np.divide(counts,n,out=np.full(len(a),np.nan),where=n>0)}
