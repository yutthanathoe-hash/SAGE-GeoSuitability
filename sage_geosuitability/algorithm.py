# -*- coding: utf-8 -*-
"""SAGE-GeoSuitability V1.0.0 | Scientific Spatial Risk Assessment | QGIS Processing Script.
Independent implementation informed by FireGeoAI / ForestEncroachAI; originals untouched.
"""
import os, csv, json, math, sys
if "__file__" in globals():
    _script_dir=os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path: sys.path.insert(0,_script_dir)
import numpy as np
from osgeo import gdal, osr
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, QgsProcessingException,
    QgsProcessingParameterMultipleLayers, QgsProcessingParameterVectorLayer,
    QgsProcessingParameterField, QgsProcessingParameterEnum, QgsProcessingParameterNumber,
    QgsProcessingParameterString, QgsProcessingParameterFolderDestination,
    QgsCoordinateReferenceSystem, QgsCoordinateTransform, QgsProject, QgsGeometry, QgsPointXY)
from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
try:
    from .validation import write_csv, reliability, risk_levels, fit_oof_sigmoid, apply_sigmoid, shap_scientific
except ImportError:
    from validation import write_csv, reliability, risk_levels, fit_oof_sigmoid, apply_sigmoid, shap_scientific
try:
    from .target_core import choose_target_field, infer_target_type, validate_test_labels, build_n_year_target
except ImportError:
    from target_core import choose_target_field, infer_target_type, validate_test_labels, build_n_year_target
import joblib
try:
    from .scientific_core import FoldFeatureSelector, correlation_vif, metrics, ood_fit, shap_explain
except ImportError:
    from scientific_core import FoldFeatureSelector, correlation_vif, metrics, ood_fit, shap_explain

class SelectedPreprocessor(BaseEstimator, TransformerMixin):
    """Infer selected categorical columns by column provenance passed from selector."""
    def __init__(self,categorical=()): self.categorical=categorical
    def fit(self,X,y=None):
        # Selector injects original indices through the pipeline's set_params below.
        indices=getattr(self,'selected_original_',tuple(range(X.shape[1])))
        cats=[i for i,j in enumerate(indices) if j in self.categorical]
        nums=[i for i,j in enumerate(indices) if j not in self.categorical]
        trans=[]
        if cats: trans.append(('cat',OneHotEncoder(handle_unknown='ignore'),cats))
        if nums: trans.append(('num',StandardScaler(),nums))
        self.transformer_=ColumnTransformer(trans).fit(X,y);return self
    def transform(self,X):return self.transformer_.transform(X)

class ScientificPipeline(Pipeline):
    def fit(self,X,y=None,**kwargs):
        # The feature selector is fit on this call's X only (fold-train or final train).
        selector=self.named_steps['select'].fit(X,y)
        self.named_steps['pre'].selected_original_=tuple(selector.selected_)
        return super().fit(X,y,**kwargs)

class SAGEGeoSuitability(QgsProcessingAlgorithm):
    def name(self): return 'sage_geosuitability_v1'
    def displayName(self): return 'SAGE-GeoSuitability V1.0.0 — Scientific Spatial Risk Assessment'
    def group(self): return 'SAGE GeoAI'
    def groupId(self): return 'sage_geoai'
    def createInstance(self): return SAGEGeoSuitability()
    def shortHelpString(self): return ('Predictors + labelled training determine the prediction target. '
        'Optional independent test is never used in training or model selection. '
        'Projected CRS in metres required. Supports binary/multiclass classification and regression. '
        'Target type AUTO requires confirmation for numeric ambiguous targets.')
    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMultipleLayers('PREDICTORS','Predictor raster layers (multiband; first raster is analysis grid)',layerType=QgsProcessing.TypeRaster))
        self.addParameter(QgsProcessingParameterVectorLayer('TRAIN','Training samples (point / polygon)',[QgsProcessing.TypeVectorPoint,QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterField('TARGET','Training target field (blank = auto-suggest)',parentLayerParameterName='TRAIN',optional=True))
        self.addParameter(QgsProcessingParameterVectorLayer('TEST','Independent test samples (optional)',[QgsProcessing.TypeVectorPoint,QgsProcessing.TypeVectorPolygon],optional=True))
        self.addParameter(QgsProcessingParameterField('TEST_TARGET','Independent test target field (required only with test)',parentLayerParameterName='TEST',optional=True))
        self.addParameter(QgsProcessingParameterVectorLayer('AOI','Study area polygon (optional)',[QgsProcessing.TypeVectorPolygon],optional=True))
        self.addParameter(QgsProcessingParameterEnum('TASK','Target type',options=['Auto (unambiguous only)','Classification','Regression','Ordinal classification','Count regression'],defaultValue=1))
        self.addParameter(QgsProcessingParameterString('YEAR_FIELDS','N-year binary event fields, comma separated (optional; overrides training target)',optional=True))
        self.addParameter(QgsProcessingParameterEnum('YEAR_TARGET','N-year target statistic',options=['Event count','Event rate'],defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber('MIN_VALID_YEARS','Minimum valid annual observations',type=QgsProcessingParameterNumber.Integer,defaultValue=1,minValue=1))
        self.addParameter(QgsProcessingParameterEnum('MODEL','Model',options=['Compare all','Random Forest','Extra Trees','Gradient Boosting','Logistic / Ridge','XGBoost (if installed)'],defaultValue=0))
        self.addParameter(QgsProcessingParameterString('CATEGORICAL','Categorical predictor names, comma separated (optional)',optional=True))
        self.addParameter(QgsProcessingParameterNumber('FOLDS','Spatial CV folds',type=QgsProcessingParameterNumber.Integer,defaultValue=5,minValue=2,maxValue=10))
        self.addParameter(QgsProcessingParameterNumber('REPEATS','Repeated spatial CV runs (regression folds may repeat)',type=QgsProcessingParameterNumber.Integer,defaultValue=3,minValue=1,maxValue=10))
        self.addParameter(QgsProcessingParameterNumber('BOOTSTRAP','Bootstrap replicates for CV interval',type=QgsProcessingParameterNumber.Integer,defaultValue=500,minValue=100,maxValue=5000))
        self.addParameter(QgsProcessingParameterNumber('BLOCK_KM','Spatial block size (km)',type=QgsProcessingParameterNumber.Double,defaultValue=5,minValue=0.1))
        self.addParameter(QgsProcessingParameterNumber('SEED','Random seed',type=QgsProcessingParameterNumber.Integer,defaultValue=42,minValue=0))
        self.addParameter(QgsProcessingParameterNumber('CORRELATION','Fold-only maximum absolute correlation',type=QgsProcessingParameterNumber.Double,defaultValue=0.95,minValue=0.5,maxValue=1.0))
        self.addParameter(QgsProcessingParameterNumber('TEST_BUFFER_M','Minimum train-test distance in metres (0 disables)',type=QgsProcessingParameterNumber.Double,defaultValue=0,minValue=0))
        self.addParameter(QgsProcessingParameterEnum('CALIBRATION','Probability calibration',options=['None','Spatial out-of-fold sigmoid (binary only)'],defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber('OOD_QUANTILE','OOD training distance quantile',type=QgsProcessingParameterNumber.Double,defaultValue=0.99,minValue=0.8,maxValue=0.999))
        self.addParameter(QgsProcessingParameterNumber('SHAP_SAMPLES','SHAP samples (0 disables)',type=QgsProcessingParameterNumber.Integer,defaultValue=300,minValue=0,maxValue=500))
        self.addParameter(QgsProcessingParameterEnum('VALIDATION_POLICY','Independent test validation policy',options=['Strict spatial holdout (shared raster cells forbidden)','Temporal holdout (shared cells allowed across years)','Independent test — allow shared raster cells (unverified temporal)'],defaultValue=2))
        self.addParameter(QgsProcessingParameterString('TRAIN_TIME','Training time field',optional=True))
        self.addParameter(QgsProcessingParameterString('TEST_TIME','Test time field',optional=True))
        self.addParameter(QgsProcessingParameterFolderDestination('OUTPUT','Output folder'))

    @staticmethod
    def _raster(path, a, gt, prj, dtype=gdal.GDT_Float32, nodata=-9999):
        ds=gdal.GetDriverByName('GTiff').Create(path,a.shape[1],a.shape[0],1,dtype,options=['COMPRESS=DEFLATE','TILED=YES','BIGTIFF=IF_SAFER'])
        ds.SetGeoTransform(gt); ds.SetProjection(prj); b=ds.GetRasterBand(1); b.SetNoDataValue(float(nodata))
        b.WriteArray(np.where(np.isfinite(a),a,nodata)); ds=None

    @staticmethod
    def _samples(layer, field, ref_crs, gt, valid, arrays):
        if layer is None: return np.empty((0,len(arrays))), np.array([]), np.empty((0,2)), set()
        tr=QgsCoordinateTransform(layer.crs(),ref_crs,QgsProject.instance()) if layer.crs()!=ref_crs else None
        invgt=gdal.InvGeoTransform(gt)
        if invgt is None: raise QgsProcessingException('Non-invertible raster geotransform')
        rows,cols=valid.shape; coords={}; seen=set()
        for ft in layer.getFeatures():
            value=ft[field]
            if value is None or str(value).strip()=='' or str(value)=='NULL': continue
            geom=QgsGeometry(ft.geometry())
            if geom.isEmpty(): continue
            if tr: geom.transform(tr)
            if geom.type()==0:
                p=geom.asPoint(); candidates=[(p.x(),p.y())]
            else:
                bb=geom.boundingBox(); p0=gdal.ApplyGeoTransform(invgt,bb.xMinimum(),bb.yMaximum()); p1=gdal.ApplyGeoTransform(invgt,bb.xMaximum(),bb.yMinimum())
                c0,c1=sorted([int(math.floor(p0[0])),int(math.ceil(p1[0]))]); r0,r1=sorted([int(math.floor(p0[1])),int(math.ceil(p1[1]))]); candidates=[]
                for r in range(max(0,r0),min(rows,r1+1)):
                    for c in range(max(0,c0),min(cols,c1+1)):
                        x,y=gdal.ApplyGeoTransform(gt,c+0.5,r+0.5)
                        if geom.contains(QgsGeometry.fromPointXY(QgsPointXY(x,y))): candidates.append((x,y))
            for x,y in candidates:
                c,r=gdal.ApplyGeoTransform(invgt,x,y); c=int(math.floor(c)); r=int(math.floor(r))
                if 0<=r<rows and 0<=c<cols and valid[r,c]:
                    key=(r,c)
                    if key in coords and str(coords[key][0])!=str(value): raise QgsProcessingException('Conflicting labels in same raster cell')
                    coords[key]=(value,x,y); seen.add(key)
        keys=sorted(coords); X=np.array([[a[r,c] for a in arrays] for r,c in keys],dtype=float).reshape((-1,len(arrays)))
        return X,np.array([coords[k][0] for k in keys]),np.array([[coords[k][1],coords[k][2]] for k in keys]).reshape((-1,2)),seen

    def processAlgorithm(self, p, context, feedback):
        layers=self.parameterAsLayerList(p,'PREDICTORS',context)
        if not layers: raise QgsProcessingException('At least one predictor raster is required')
        out=self.parameterAsString(p,'OUTPUT',context); os.makedirs(out,exist_ok=True)
        ref=gdal.Open(layers[0].source().split('|')[0]);
        if ref is None: raise QgsProcessingException('Cannot open reference raster')
        gt=ref.GetGeoTransform(); prj=ref.GetProjection(); rows,cols=ref.RasterYSize,ref.RasterXSize
        srs=osr.SpatialReference(); srs.ImportFromWkt(prj)
        if not srs.IsProjected() or abs(srs.GetLinearUnits()-1)>0.01: raise QgsProcessingException('Reference CRS must use projected metre units')
        if gt[2]!=0 or gt[4]!=0: raise QgsProcessingException('Rotated reference rasters not supported in V1.0')
        # Fail fast on target and mapping errors before costly raster alignment.
        train_pre=self.parameterAsVectorLayer(p,'TRAIN',context)
        if train_pre is None: raise QgsProcessingException('Training samples are required')
        try:
            field_pre=choose_target_field(train_pre.fields().names(),self.parameterAsString(p,'TARGET',context))
        except ValueError as exc:
            raise QgsProcessingException('Training target error: '+str(exc))
        task_pre=self.parameterAsEnum(p,'TASK',context)
        if task_pre==0:
            raise QgsProcessingException('Target type Auto is ambiguous for numeric labels. Select Classification for fire_code 0/1 before running.')
        feedback.pushInfo('Preflight: target='+field_pre+', task='+['Auto','Classification','Regression','Ordinal classification','Count regression'][task_pre])
        # Preflight for point training: report conflicting labels by reference raster cell.
        # Does not delete, merge or change any scientific training labels.
        if train_pre.geometryType()==0:
            tr_pre=QgsCoordinateTransform(train_pre.crs(),QgsCoordinateReferenceSystem.fromWkt(prj),QgsProject.instance()) if train_pre.crs()!=QgsCoordinateReferenceSystem.fromWkt(prj) else None
            inv_pre=gdal.InvGeoTransform(gt)
            if inv_pre is None: raise QgsProcessingException('Non-invertible reference raster geotransform')
            cell_labels={}; conflicts=[]
            for feature in train_pre.getFeatures():
                value=feature[field_pre]
                if value is None or str(value).strip() in ('','NULL'): continue
                geom=QgsGeometry(feature.geometry())
                if geom.isEmpty(): continue
                if tr_pre: geom.transform(tr_pre)
                pt=geom.asPoint()
                c,r=gdal.ApplyGeoTransform(inv_pre,pt.x(),pt.y())
                col,row=int(math.floor(c)),int(math.floor(r))
                if not (0<=row<rows and 0<=col<cols): continue
                key=(row,col); entry=(str(value),str(feature.id()),pt.x(),pt.y())
                if key in cell_labels:
                    for previous in cell_labels[key]:
                        if previous[0]!=entry[0]:
                            conflicts.append((row,col,previous,entry))
                cell_labels.setdefault(key,[]).append(entry)
            if conflicts:
                report=os.path.join(out,'preflight_conflicting_training_cells.csv')
                with open(report,'w',newline='',encoding='utf-8-sig') as fh:
                    writer=csv.writer(fh)
                    writer.writerow(['row','column','first_feature_id','first_label','first_x','first_y','second_feature_id','second_label','second_x','second_y'])
                    for row,col,a,b in conflicts:
                        writer.writerow([row,col,a[1],a[0],a[2],a[3],b[1],b[0],b[2],b[3]])
                raise QgsProcessingException('Preflight: '+str(len(conflicts))+' conflicting label pairs in reference raster cells. Review '+report+'; labels were not modified.')
            feedback.pushInfo('Preflight: point training cell labels have no conflicts on reference raster grid')
        ref_crs=QgsCoordinateReferenceSystem.fromWkt(prj); arrays=[]; names=[]; valid=np.ones((rows,cols),bool)
        cat_input={v.strip() for v in (self.parameterAsString(p,'CATEGORICAL',context) or '').split(',') if v.strip()}
        # Known thematic forest-type raster: never bilinearly resample a class code.
        for lyr in layers:
            stem=os.path.splitext(os.path.basename(lyr.source().split('|')[0]))[0]
            if stem.lower() in ('forest_type','foresttype','forest_types'):
                ds_check=gdal.Open(lyr.source().split('|')[0])
                for bi in range(1,ds_check.RasterCount+1):
                    cat_input.add(stem+'_B%02d'%bi)
                feedback.pushInfo('Categorical forest-type raster auto-detected: nearest-neighbour alignment enabled')
        raster_audit=[]
        for i,lyr in enumerate(layers):
            src=lyr.source().split('|')[0]
            ds=gdal.Open(src)
            if ds is None: raise QgsProcessingException('Cannot open '+src)
            if i>0:
                # Warp in-memory, preserve reference pixel alignment and all bands
                mem=gdal.GetDriverByName('MEM').Create('',cols,rows,ds.RasterCount,gdal.GDT_Float32)
                mem.SetGeoTransform(gt); mem.SetProjection(prj)
                for b in range(1,mem.RasterCount+1): mem.GetRasterBand(b).SetNoDataValue(-9999)
                # Warp each band separately so categorical and continuous predictors keep correct resampling.
                for band_id in range(1,ds.RasterCount+1):
                    band_name=os.path.splitext(os.path.basename(src))[0]+'_B%02d'%band_id
                    resample=gdal.GRA_NearestNeighbour if band_name in cat_input else gdal.GRA_Bilinear
                    src_band=gdal.GetDriverByName('MEM').Create('',ds.RasterXSize,ds.RasterYSize,1,gdal.GDT_Float32)
                    src_band.SetGeoTransform(ds.GetGeoTransform()); src_band.SetProjection(ds.GetProjection())
                    original=ds.GetRasterBand(band_id)
                    src_band.GetRasterBand(1).WriteArray(original.ReadAsArray())
                    if original.GetNoDataValue() is not None: src_band.GetRasterBand(1).SetNoDataValue(original.GetNoDataValue())
                    dst_band=gdal.GetDriverByName('MEM').Create('',cols,rows,1,gdal.GDT_Float32)
                    dst_band.SetGeoTransform(gt); dst_band.SetProjection(prj)
                    dst_band.GetRasterBand(1).SetNoDataValue(-9999)
                    err=gdal.ReprojectImage(src_band,dst_band,ds.GetProjection(),prj,resample)
                    if err!=0: raise QgsProcessingException('Raster alignment failed: '+src)
                    mem.GetRasterBand(band_id).WriteArray(dst_band.GetRasterBand(1).ReadAsArray())
                ds=mem
            for b in range(1,ds.RasterCount+1):
                band=ds.GetRasterBand(b); a=band.ReadAsArray().astype(float); nd=band.GetNoDataValue(); ok=np.isfinite(a)
                if nd is not None: ok &= a!=nd
                a[~ok]=np.nan; valid &= ok
                name=os.path.splitext(os.path.basename(src))[0]+'_B%02d'%b
                finite=a[np.isfinite(a)]
                categorical=name in cat_input
                fractional=float(np.mean(np.abs(finite-np.rint(finite))>1e-4)) if categorical and finite.size else None
                flags=[]
                if not finite.size: flags.append('NO_VALID_PIXELS')
                if categorical and fractional is not None and fractional>0.001: flags.append('FRACTIONAL_CATEGORY_CODES_IN_SOURCE_OR_ALIGNED_GRID')
                if finite.size and np.min(finite)==np.max(finite): flags.append('CONSTANT_RASTER')
                if 'slope' in name.lower() and finite.size and np.max(finite)>90: flags.append('SLOPE_GT_90_CHECK_PERCENT_VS_DEGREES')
                if any(v in name.lower() for v in ('ndvi','ndwi','nbr','evi')) and finite.size and (np.min(finite)<-1.01 or np.max(finite)>1.01): flags.append('INDEX_OUTSIDE_MINUS1_PLUS1_VERIFY_SCALING')
                raster_audit.append({'predictor':name,'source':src,'categorical':categorical,'resampling':'native/reference' if i==0 else ('nearest' if categorical else 'bilinear'),'valid_pixel_pct':round(100*finite.size/a.size,3),'min':float(np.min(finite)) if finite.size else None,'max':float(np.max(finite)) if finite.size else None,'unique_values':int(np.unique(finite).size) if finite.size else 0,'fractional_category_fraction':fractional,'flags':flags})
                if name in names: name+='_'+str(i)
                names.append(name); arrays.append(a)
        # Scientific raster integrity audit (warnings, no silent alteration of original files).
        with open(os.path.join(out,'Raster_Integrity_Audit.json'),'w',encoding='utf-8') as fh:
            json.dump({'version':'1.0.0','reference_grid':{'rows':rows,'cols':cols,'geotransform':list(gt),'crs_wkt':prj},'rasters':raster_audit},fh,ensure_ascii=False,indent=2)
        with open(os.path.join(out,'Raster_Integrity_Audit.csv'),'w',newline='',encoding='utf-8-sig') as fh:
            columns=['predictor','source','categorical','resampling','valid_pixel_pct','min','max','unique_values','fractional_category_fraction','flags']
            wr=csv.DictWriter(fh,fieldnames=columns);wr.writeheader()
            for entry in raster_audit: wr.writerow({**entry,'flags':';'.join(entry['flags'])})
        for entry in raster_audit:
            if entry['flags']: feedback.pushWarning('Raster QA '+entry['predictor']+': '+', '.join(entry['flags']))
        missing=cat_input-set(names)
        if missing: raise QgsProcessingException('Unknown categorical predictor names: '+', '.join(sorted(missing)))
        aoi=self.parameterAsVectorLayer(p,'AOI',context)
        if aoi:
            mask_ds=gdal.GetDriverByName('MEM').Create('',cols,rows,1,gdal.GDT_Byte); mask_ds.SetGeoTransform(gt); mask_ds.SetProjection(prj)
            tr=QgsCoordinateTransform(aoi.crs(),ref_crs,QgsProject.instance()) if aoi.crs()!=ref_crs else None
            from osgeo import ogr
            drv=ogr.GetDriverByName('Memory'); vds=drv.CreateDataSource(''); vly=vds.CreateLayer('mask',srs,ogr.wkbPolygon)
            for ft in aoi.getFeatures():
                g=QgsGeometry(ft.geometry());
                if tr: g.transform(tr)
                og=ogr.CreateGeometryFromWkb(bytes(g.asWkb())); feat=ogr.Feature(vly.GetLayerDefn()); feat.SetGeometry(og); vly.CreateFeature(feat)
            gdal.RasterizeLayer(mask_ds,[1],vly,burn_values=[1]); valid &= mask_ds.GetRasterBand(1).ReadAsArray()>0
        train=self.parameterAsVectorLayer(p,'TRAIN',context); field=self.parameterAsString(p,'TARGET',context)
        year_fields=[v.strip() for v in (self.parameterAsString(p,'YEAR_FIELDS',context) or '').split(',') if v.strip()]
        if year_fields:
            raise QgsProcessingException('N-year field aggregation is provided by target_core.build_n_year_target and preprocessing utility; direct QGIS polygon aggregation is not yet wired in V1.3. Supply a prepared event_count or event_rate target field.')
        field=choose_target_field(train.fields().names(),field)
        feedback.pushInfo('Training target selected: '+field)
        test=self.parameterAsVectorLayer(p,'TEST',context); test_field=self.parameterAsString(p,'TEST_TARGET',context)
        if test is not None:
            test_field=choose_target_field(test.fields().names(),test_field,preferred=field)
            feedback.pushInfo('Independent test target selected: '+test_field)
        X,yraw,xy,train_cells=self._samples(train,field,ref_crs,gt,valid,arrays)
        if len(yraw)<20: raise QgsProcessingException('At least 20 valid training raster cells required')
        task=self.parameterAsEnum(p,'TASK',context)
        if task==0:
            suggested=infer_target_type(yraw)
            if suggested=='ambiguous': raise QgsProcessingException('Ambiguous numeric target: explicitly select Classification, Ordinal, Regression or Count')
            task=1 if suggested=='classification' else 2
        classification=(task in (1,3)); encoder=None
        if task==4:
            try: count_values=np.asarray(yraw,dtype=float)
            except (TypeError,ValueError): raise QgsProcessingException('Count target must be nonnegative integers')
            if np.any(~np.isfinite(count_values)) or np.any(count_values<0) or np.any(count_values!=np.floor(count_values)):
                raise QgsProcessingException('Count target must be nonnegative integers')
        if classification:
            encoder=LabelEncoder(); y=encoder.fit_transform([str(v) for v in yraw]); nclass=len(encoder.classes_)
            if nclass<2: raise QgsProcessingException('Classification needs at least two classes')
            if min(np.bincount(y))<2: raise QgsProcessingException('At least two training cells per class required')
        else:
            try: y=np.array([float(v) for v in yraw],dtype=float)
            except Exception: raise QgsProcessingException('Regression target must be numeric')
            if not np.all(np.isfinite(y)): raise QgsProcessingException('Nonfinite regression target')
        Xt,yt_raw,xytest,test_cells=self._samples(test,test_field,ref_crs,gt,valid,arrays) if test else (None,None,None,set())
        if test and not len(yt_raw): raise QgsProcessingException('Test layer has no valid samples on predictor grid')
        if test and (train.id()==test.id() or (train.source() and test.source() and os.path.normcase(os.path.abspath(train.source().split('|')[0]))==os.path.normcase(os.path.abspath(test.source().split('|')[0])))):
            raise QgsProcessingException('Independent TEST must not be the same dataset as TRAIN. Leave TEST blank or supply a distinct test dataset.')
        shared_cells=train_cells.intersection(test_cells) if test else set()
        policy=self.parameterAsEnum(p,'VALIDATION_POLICY',context)
        audit={'validation_policy':{0:'strict_spatial_holdout',1:'temporal_holdout',2:'independent_test_allow_shared_cells'}.get(policy,'invalid'),
               'train_test_shared_raster_cells':len(shared_cells),'train_n':len(yraw),
               'test_n':len(yt_raw) if test else 0,
               'temporal_independence_verified':False}
        if policy not in (0,1,2):
            raise QgsProcessingException('Unknown independent validation policy: {}'.format(policy))
        # Temporal holdout must be proven from per-feature year fields before shared cells are accepted.
        # The year audit does not claim that predictor rasters are temporally valid.
        train_time=(self.parameterAsString(p,'TRAIN_TIME',context) or '').strip()
        test_time=(self.parameterAsString(p,'TEST_TIME',context) or '').strip()
        if policy==1 and not test:
            raise QgsProcessingException('Temporal holdout policy requires an independent TEST layer')
        if policy==1 and not (train_time and test_time):
            raise QgsProcessingException('Temporal holdout requires TRAIN_TIME and TEST_TIME year/date fields. No unverified overlap bypass.')
        if policy==2 and (train_time or test_time):
            feedback.pushWarning('Year fields supplied under unverified independent-test policy: years will be audited when both fields are present; policy remains unverified.')
        if bool(train_time)!=bool(test_time):
            raise QgsProcessingException('Specify both TRAIN_TIME and TEST_TIME, or neither (strict spatial policy only)')
        if train_time and test_time:
            from datetime import datetime
            def parse_year(value):
                if value is None: raise ValueError('missing year')
                raw=str(value).strip()
                if not raw or raw.upper() in ('NULL','NONE'): raise ValueError('missing year')
                # Accept 2025, 2568, 2025.0, 2025-03-14, 2568-03-14, ISO datetimes.
                m=re.match(r'^(\d{4})(?:\.0+)?(?:$|[-/ T])',raw)
                if not m: raise ValueError('expected YYYY, Thai Buddhist YYYY, or ISO date')
                yr=int(m.group(1)); yr=yr-543 if yr>=2400 else yr
                if not 1900<=yr<=2200: raise ValueError('year out of supported range')
                return yr
            def read_years(layer,field,side):
                if field not in layer.fields().names():
                    raise QgsProcessingException('{} time field not found: {}'.format(side,field))
                vals=[]; errors=[]
                for f in layer.getFeatures():
                    try: vals.append(parse_year(f[field]))
                    except ValueError as e:
                        errors.append('{}: {}'.format(f.id(),e))
                if errors:
                    raise QgsProcessingException('{} time field has {} invalid/missing values (first: {}). All features require a valid year.'.format(side,len(errors),errors[0]))
                if not vals: raise QgsProcessingException('{} time field contains no records'.format(side))
                return vals
            ty=read_years(train,train_time,'TRAIN'); vy=read_years(test,test_time,'TEST')
            audit['train_year_range']=[min(ty),max(ty)]
            audit['test_year_range']=[min(vy),max(vy)]
            audit['train_year_counts']={str(z):ty.count(z) for z in sorted(set(ty))}
            audit['test_year_counts']={str(z):vy.count(z) for z in sorted(set(vy))}
            if policy==1 and max(ty)>=min(vy):
                raise QgsProcessingException('Temporal holdout requires all TRAIN years earlier than all TEST years (train max={}, test min={}).'.format(max(ty),min(vy)))
            audit['temporal_independence_verified']=True
            feedback.pushInfo('Temporal year audit passed: TRAIN {}-{} < TEST {}-{}'.format(min(ty),max(ty),min(vy),max(vy)))
        elif policy==1:
            raise QgsProcessingException('Temporal holdout requires valid year fields')
        if policy==0 and shared_cells:
            raise QgsProcessingException('Train/test raster-cell overlap detected ({} shared cells). Use Temporal holdout only with verified year fields.'.format(len(shared_cells)))
        if test:
            from scipy.spatial import cKDTree
            dist=cKDTree(xy).query(xytest,k=1)[0]
            audit['min_train_test_distance_m']=float(dist.min())
            buffer_m=self.parameterAsDouble(p,'TEST_BUFFER_M',context)
            audit['train_test_distance_below_buffer_count']=int(np.sum(dist<buffer_m)) if buffer_m>0 else 0
            if audit['train_test_distance_below_buffer_count']:
                raise QgsProcessingException('Independent test too close to training samples for configured TEST_BUFFER_M. Set 0 for temporal same-area holdout.')
        audit['shared_cells_action']=('ALLOWED_VERIFIED_TEMPORAL_HOLDOUT' if policy==1 else 'ALLOWED_UNVERIFIED_SHARED_CELLS' if policy==2 else 'BLOCKED_STRICT') if shared_cells else 'NONE'
        audit['spatial_independence_verified']=bool(not shared_cells and (not test or audit.get('min_train_test_distance_m',0)>0))
        audit['test_used_for_training_or_selection']=False
        audit['limitations']=['Shared cells are not spatially independent; if no time fields were supplied, temporal independence is unverified','Independent test predictions may reuse predictor values from training cells','A separate predictor-date/provenance audit is required to establish pre-event predictor validity']
        if shared_cells and policy==1:
            feedback.pushWarning('{} shared raster cells retained: verified later-year independent test. This is TEMPORAL validation, not spatial independence.'.format(len(shared_cells)))
        if shared_cells and policy==2:
            feedback.pushWarning('{} shared raster cells retained without year verification. Test is a separate labelled dataset, but spatial and temporal independence are NOT established.'.format(len(shared_cells)))
        if policy==2 and not shared_cells:
            feedback.pushWarning('Independent-test unverified policy selected. No shared cells detected, but temporal independence is not established.')
        # Save an auditable cell-level overlap table, without dropping any sample.
        with open(os.path.join(out,'Train_Test_Cell_Overlap.csv'),'w',newline='',encoding='utf-8-sig') as fh:
            w=csv.writer(fh); w.writerow(['row','col','cell_id','validation_policy','action'])
            for r,c in sorted(shared_cells):
                w.writerow([r,c,int(r)*cols+int(c),audit['validation_policy'],audit['shared_cells_action']])
        with open(os.path.join(out,'Temporal_Validation_Audit.json'),'w',encoding='utf-8') as fh:
            json.dump(audit,fh,ensure_ascii=False,indent=2)
        if test:
            if classification:
                unknown=validate_test_labels(encoder.classes_,yt_raw)
                if unknown: raise QgsProcessingException('Test has unseen target classes: '+str(unknown))
                yt=encoder.transform([str(v) for v in yt_raw])
            else:
                try: yt=np.array([float(v) for v in yt_raw])
                except Exception: raise QgsProcessingException('Test target must be numeric')
        cat_idx=[i for i,n in enumerate(names) if n in cat_input]; num_idx=[i for i,n in enumerate(names) if n not in cat_input]
        transformers=[]
        if cat_idx: transformers.append(('cat',OneHotEncoder(handle_unknown='ignore'),cat_idx))
        if num_idx: transformers.append(('num',StandardScaler(),num_idx))
        pre=ColumnTransformer(transformers)
        def make_pipe(est):
            # Selection and preprocessing are refitted INSIDE each CV training fold.
            # The preprocessor must be reconstructed for the selected columns.
            from sklearn.preprocessing import FunctionTransformer
            from sklearn.base import BaseEstimator, TransformerMixin
            class _PreprocessSelected(BaseEstimator,TransformerMixin):
                def fit(self,X,y=None):
                    self.selected_=self.selector.selected_.tolist()
                    cat=[i for i,j in enumerate(self.selected_) if j in cat_idx]
                    num=[i for i,j in enumerate(self.selected_) if j not in cat_idx]
                    steps=[]
                    if cat: steps.append(('cat',OneHotEncoder(handle_unknown='ignore'),cat))
                    if num: steps.append(('num',StandardScaler(),num))
                    self.transformer_=ColumnTransformer(steps).fit(X,y);return self
                def transform(self,X):return self.transformer_.transform(X)
            # Use dynamic preprocessing based on fitted selector via a combined estimator below.
            return ScientificPipeline([('select',FoldFeatureSelector(categorical=tuple(cat_idx),correlation=self.parameterAsDouble(p,'CORRELATION',context))),
                ('pre',SelectedPreprocessor(tuple(cat_idx))),('model',clone(est))])
        seed=self.parameterAsInt(p,'SEED',context)
        if classification:
            models={'RandomForest':RandomForestClassifier(n_estimators=250,random_state=seed,n_jobs=-1,class_weight='balanced'), 'ExtraTrees':ExtraTreesClassifier(n_estimators=250,random_state=seed,n_jobs=-1,class_weight='balanced'),'GradientBoosting':GradientBoostingClassifier(random_state=seed),'LogisticRegression':LogisticRegression(max_iter=1500,class_weight='balanced')}
        else:
            models={'RandomForest':RandomForestRegressor(n_estimators=250,random_state=seed,n_jobs=-1),'ExtraTrees':ExtraTreesRegressor(n_estimators=250,random_state=seed,n_jobs=-1),'GradientBoosting':GradientBoostingRegressor(random_state=seed),'Ridge':Ridge()}
        choice=self.parameterAsEnum(p,'MODEL',context)
        try:
            from xgboost import XGBClassifier, XGBRegressor
            if classification:
                models['XGBoost']=XGBClassifier(n_estimators=250,max_depth=5,learning_rate=0.05,tree_method='hist',eval_metric='logloss',random_state=seed,n_jobs=-1)
            else:
                models['XGBoost']=XGBRegressor(n_estimators=250,max_depth=5,learning_rate=0.05,tree_method='hist',random_state=seed,n_jobs=-1)
        except ImportError:
            if choice==5: raise QgsProcessingException('XGBoost selected but xgboost is not installed')
            feedback.pushInfo('XGBoost not installed; continuing without it')
        if choice:
            keys=['RandomForest','ExtraTrees','GradientBoosting','LogisticRegression' if classification else 'Ridge','XGBoost']
            key=keys[choice-1]
            models={key:models[key]}

        block=self.parameterAsDouble(p,'BLOCK_KM',context)*1000
        bx=np.floor((xy[:,0]-xy[:,0].min())/block).astype(int); by=np.floor((xy[:,1]-xy[:,1].min())/block).astype(int)
        groups=by*(bx.max()+1)+bx; n_groups=len(np.unique(groups)); folds=self.parameterAsInt(p,'FOLDS',context)
        if n_groups<folds: raise QgsProcessingException('Insufficient spatial blocks: reduce folds or block size; no random CV fallback')
        repeats=self.parameterAsInt(p,'REPEATS',context)
        rng=np.random.default_rng(seed)
        results=[]; oof_records=[]
        for repeat in range(repeats):
            if classification:
                splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed+repeat*1009)
                splits=list(splitter.split(X,y,groups))
            else:
                # GroupKFold provides spatial separation; shuffling groups varies repeats.
                unique_groups=np.unique(groups); shuffled=unique_groups.copy(); rng.shuffle(shuffled)
                mapping={g:i for i,g in enumerate(shuffled)}
                shuffled_groups=np.array([mapping[g] for g in groups])
                splits=list(GroupKFold(n_splits=folds).split(X,y,shuffled_groups))
            for name,est in models.items():
                for fold,(tr,va) in enumerate(splits,1):
                    if classification and len(np.unique(y[tr]))!=len(np.unique(y)):
                        raise QgsProcessingException('Spatial fold missing training class; adjust folds or block size')
                    pipe_cv=make_pipe(est)
                    pipe_cv.fit(X[tr],y[tr]); pred=pipe_cv.predict(X[va])
                    if classification:
                        score=f1_score(y[va],pred,average='macro',zero_division=0); metric='macro_f1'
                    else:
                        score=-math.sqrt(mean_squared_error(y[va],pred)); metric='negative_rmse'
                    results.append({'model':name,'repeat':repeat+1,'fold':fold,'metric':metric,'value':float(score)})
                    if feedback.isCanceled(): raise QgsProcessingException('Canceled')
            feedback.pushInfo('Completed spatial CV repeat %d/%d'%(repeat+1,repeats))
        mean_scores={n:np.mean([r['value'] for r in results if r['model']==n]) for n in models}
        best=max(mean_scores,key=mean_scores.get)
        pipe=make_pipe(models[best]); pipe.fit(X,y)
        with open(os.path.join(out,'CV_Results.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['model','repeat','fold','metric','value']); w.writeheader(); w.writerows(results)
        joblib.dump({'pipeline':pipe,'features':names,'task':('ordinal' if task==3 else 'count' if task==4 else 'classification' if classification else 'regression'),'classes':encoder.classes_.tolist() if classification else None},os.path.join(out,'Best_Model.joblib'))
        report={'version':'1.0.0','task':('ordinal' if task==3 else 'count' if task==4 else 'classification' if classification else 'regression'),'best_model':best,'cv_mean_scores':{k:float(v) for k,v in mean_scores.items()},'train_n':len(y),'test_n':len(yt) if test else 0,'test_used_for_training_or_selection':False,'test_overlap_check':'policy-dependent raster-cell overlap; year verification required only for temporal holdout; predictor temporal provenance not automatically verified','validation_policy':audit['validation_policy'],'shared_raster_cells':len(shared_cells),'temporal_independence_verified':audit['temporal_independence_verified'],'predictors':names,'categorical':sorted(cat_input),'class_labels':encoder.classes_.tolist() if classification else None,'training_target_field':field,'test_target_field':test_field if test else None,'target_field_selection':'auto_suggestion_or_user_selected','target_type_setting':task,'limitations':['Spatial block CV; temporal independence is unverified for allow-shared-cells policy; same-cell test is not spatially independent','SHAP additivity and probability calibration are explicitly audited; OOD distance remains uncalibrated','Training polygons may yield spatially autocorrelated samples']}
        report['leakage_audit']=audit
        report['target_semantics_note']='Ordinal target uses multiclass classifier; count target uses general regression and is NOT a bounded binomial model. Interpret predictions accordingly.'
        vif_rows,vif_indices,corr=correlation_vif(X,names,cat_idx)
        with open(os.path.join(out,'Correlation_Matrix.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['predictor']+[names[i] for i in vif_indices]);
            for i,row in enumerate(corr):w.writerow([names[vif_indices[i]]]+[float(v) for v in row])
        with open(os.path.join(out,'VIF.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['predictor','vif','max_abs_correlation']);w.writeheader();w.writerows(vif_rows)
        report['final_selected_features']=[names[i] for i in pipe.named_steps['select'].selected_]
        report['final_dropped_features']=[{'predictor':names[i],'reason':reason} for i,reason in pipe.named_steps['select'].dropped_]
        # Calibration fits exclusively on spatial out-of-fold training predictions.
        # The independent test is never used to fit the sigmoid or choose a model.
        calibration_requested=self.parameterAsEnum(p,'CALIBRATION',context)==1
        calibrator=None
        report['calibration_status']='DISABLED'
        report['calibration_method']='None'
        if calibration_requested:
            if not classification or len(encoder.classes_)!=2:
                raise QgsProcessingException('Spatial OOF sigmoid calibration requires binary classification')
            try:
                cal_splits=list(StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed).split(X,y,groups))
                oof=np.full(len(y),np.nan)
                for tr,va in cal_splits:
                    fold_model=make_pipe(models[best]); fold_model.fit(X[tr],y[tr])
                    oof[va]=fold_model.predict_proba(X[va])[:,1]
                calibrator=fit_oof_sigmoid((y==1).astype(int),oof)
                joblib.dump({'calibrator':calibrator,'model_name':best,'training_only':True,'source':'spatial_oof_first_repeat'},os.path.join(out,'Spatial_OOF_Calibrator.joblib'))
                report['calibration_status']='FITTED_SPATIAL_OOF_SIGMOID'
                report['calibration_method']='Logistic regression of OOF logit scores (first spatial CV repeat)'
                report['calibration_training_warning']='Calibration fitted to OOF predictions; apparent OOF calibration performance is optimistic. Use independent test for evaluation.'
                report['calibration_oof_raw_reliability']=reliability((y==1).astype(int),oof)[1]
                report['calibration_oof_apparent_reliability']=reliability((y==1).astype(int),apply_sigmoid(calibrator,oof))[1]
            except Exception as exc:
                raise QgsProcessingException('Spatial OOF calibration failed: '+str(exc))
        # Independent test is used only after model selection and final training.
        if test:
            pred=pipe.predict(Xt)
            if classification:
                report['independent_test']=metrics(yt,pred,True,pipe.predict_proba(Xt) if hasattr(pipe,'predict_proba') else None,len(encoder.classes_))
            else:
                report['independent_test']=metrics(yt,pred,False)
        else: report['independent_test']='NOT PROVIDED'
        # Independent-test reliability and observed event rate per risk level.
        # Always report raw scores, and calibrated scores separately if requested.
        if test and classification and len(encoder.classes_)==2:
            event_idx=next((i for i,label in enumerate(encoder.classes_) if str(label).strip()=='1'),None)
            if event_idx is not None:
                y_event=(yt==event_idx).astype(int)
                raw_test=pipe.predict_proba(Xt)[:,event_idx]
                report['independent_test_risk_assessment']={}
                for score_name,scores in [('raw',raw_test)]+([('calibrated',apply_sigmoid(calibrator,raw_test))] if calibrator is not None else []):
                    lv_rows=risk_levels(y_event,scores)
                    rel_rows,rel_summary=reliability(y_event,scores)
                    write_csv(os.path.join(out,'Independent_Test_Risk_Levels_'+score_name+'.csv'),lv_rows,
                              ['risk_level','score_min','score_max','n_test','event_n','non_event_n','observed_event_rate','mean_score'])
                    write_csv(os.path.join(out,'Independent_Test_Reliability_'+score_name+'.csv'),rel_rows,
                              ['bin','score_min','score_max','n','mean_score','observed_event_rate'])
                    report['independent_test_risk_assessment'][score_name]={'risk_level_table':lv_rows,'reliability':rel_summary}
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fig,ax=plt.subplots(figsize=(6,5))
                        valid_bins=[r for r in rel_rows if r['n']>0]
                        ax.plot([0,1],[0,1],'--',label='Perfect calibration')
                        ax.plot([r['mean_score'] for r in valid_bins],[r['observed_event_rate'] for r in valid_bins],marker='o',label=score_name)
                        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted score',ylabel='Observed event rate',title='Independent test reliability ('+score_name+')')
                        ax.legend();fig.tight_layout()
                        fig.savefig(os.path.join(out,'Independent_Test_Reliability_'+score_name+'.png'),dpi=160)
                        plt.close(fig)
                    except Exception as exc:
                        feedback.pushWarning('Reliability plot unavailable: '+str(exc))
                report['independent_test_risk_assessment']['interpretation']='Descriptive independent-test event rates; shared raster cells and unverified temporal independence limit generalization. Calibration metrics do not establish causality.'
            else:
                report['independent_test_risk_assessment']='SKIPPED: binary event class label 1 not found'
        else:
            report['independent_test_risk_assessment']='NOT APPLICABLE: requires binary classification, separate independent test and event label 1'
        # Training-domain diagnostic: maximum standardized distance to training distribution.
        # This is an OOD heuristic, not a probability or calibrated confidence interval.
        mu,sd,ood_threshold=ood_fit(X[:,pipe.named_steps['select'].selected_],self.parameterAsDouble(p,'OOD_QUANTILE',context)); report['ood_threshold']=ood_threshold
        # Continuous feature permutation importance on held-out spatial groups only.
        # Keep independent TEST completely outside model selection and importance.
        importance=[]
        try:
            cv_split=list(GroupKFold(n_splits=folds).split(X,y,groups))
            tr,va=cv_split[0]
            guard=make_pipe(models[best]); guard.fit(X[tr],y[tr])
            scoring='f1_macro' if classification else 'neg_root_mean_squared_error'
            perm=permutation_importance(guard,X[va],y[va],n_repeats=5,random_state=seed,scoring=scoring,n_jobs=1)
            importance=[{'predictor':n,'mean_importance':float(perm.importances_mean[i]),'std_importance':float(perm.importances_std[i])} for i,n in enumerate(names)]
            with open(os.path.join(out,'Spatial_Permutation_Importance.csv'),'w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['predictor','mean_importance','std_importance']);w.writeheader();w.writerows(importance)
        except Exception as exc:
            feedback.pushWarning('Spatial permutation importance unavailable: '+str(exc))
        report['spatial_permutation_importance']=importance
        # Bootstrap fold-score intervals (descriptive; correlated CV folds limit interpretation).
        b_n=self.parameterAsInt(p,'BOOTSTRAP',context)
        ci={}
        for model_name in models:
            scores=np.array([r['value'] for r in results if r['model']==model_name],dtype=float)
            means=np.mean(rng.choice(scores,size=(b_n,len(scores)),replace=True),axis=1)
            ci[model_name]={'mean':float(scores.mean()),'bootstrap_95pct':[float(v) for v in np.quantile(means,[.025,.975])]}
        report['cv_bootstrap_descriptive_intervals']=ci
        report['cv_repeats']=repeats
        # Predictor QA report; never remove features based on test data.
        qc=[]
        for i,n in enumerate(names):
            vals=X[:,i]
            qc.append({'predictor':n,'type':'categorical' if n in cat_input else 'continuous','train_min':float(vals.min()),'train_max':float(vals.max()),'train_variance':float(vals.var()),'unique_values':int(len(np.unique(vals)))})
        with open(os.path.join(out,'Predictor_QC.csv'),'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['predictor','type','train_min','train_max','train_variance','unique_values']);w.writeheader();w.writerows(qc)
        report['predictor_qc']=qc
        report['raster_integrity_audit']='Raster_Integrity_Audit.json'
        report['raster_integrity_warnings']={x['predictor']:x['flags'] for x in raster_audit if x['flags']}
        report['class_encoding']={str(i):str(label) for i,label in enumerate(encoder.classes_)} if classification else None
        if classification:
            with open(os.path.join(out,'Class_Mapping.csv'),'w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['raster_class_code','original_target_label']);w.writerows(enumerate(encoder.classes_))
        prediction=np.full((rows,cols),np.nan,dtype=np.float32); confidence=np.full((rows,cols),np.nan,dtype=np.float32)
        ood=np.full((rows,cols),np.nan,dtype=np.float32)
        # Risk score refers to explicit event class 1, not the maximum class confidence.
        risk_index=next((i for i,label in enumerate(encoder.classes_) if str(label).strip()=='1'),None) if classification and len(encoder.classes_)==2 else None
        risk_score=np.full((rows,cols),np.nan,dtype=np.float32) if risk_index is not None else None
        inds=np.flatnonzero(valid.ravel()); batch=50000
        for start in range(0,len(inds),batch):
            ids=inds[start:start+batch]; rr=ids//cols; cc=ids%cols
            XX=np.column_stack([a[rr,cc] for a in arrays]); pp=pipe.predict(XX)
            prediction.ravel()[ids]=pp.astype(np.float32)
            ood.ravel()[ids]=np.max(np.abs((XX[:,pipe.named_steps['select'].selected_]-mu)/sd),axis=1).astype(np.float32)
            if classification and hasattr(pipe,'predict_proba'):
                probs=pipe.predict_proba(XX)
                confidence.ravel()[ids]=(np.maximum(apply_sigmoid(calibrator,probs[:,1]),1-apply_sigmoid(calibrator,probs[:,1])) if calibrator is not None else probs.max(axis=1)).astype(np.float32)
                if risk_score is not None:
                    event_prob=probs[:,risk_index]
                    if calibrator is not None:
                        if risk_index!=1: raise QgsProcessingException('Calibration event class index mismatch')
                        event_prob=apply_sigmoid(calibrator,event_prob)
                    risk_score.ravel()[ids]=event_prob.astype(np.float32)
            if feedback.isCanceled(): raise QgsProcessingException('Canceled')
        ood_flag=np.where(np.isfinite(ood),(ood>ood_threshold).astype(float),np.nan)
        self._raster(os.path.join(out,'OOD_Flag.tif'),ood_flag,gt,prj)
        report['ood_threshold_validation']='In-sample training quantile only; independent OOD validation not established'
        report['uncertainty_definition']=('1 - max binary class probability after spatial-OOF sigmoid calibration' if calibrator is not None else '1 - max predicted class probability (uncalibrated); regression not supported')
        if classification:
            self._raster(os.path.join(out,'Uncertainty_Heuristic.tif'),np.where(np.isfinite(confidence),1-confidence,np.nan),gt,prj)
        shap_n=self.parameterAsInt(p,'SHAP_SAMPLES',context)
        report['shap_status']='DISABLED'
        if shap_n:
            try:
                import shap
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from scipy import sparse
                rng_shap=np.random.default_rng(seed)
                ids=np.sort(rng_shap.choice(len(X),size=min(shap_n,len(X)),replace=False))
                selector=pipe.named_steps['select']; pre=pipe.named_steps['pre']; model=pipe.named_steps['model']
                selected_names=[names[int(j)] for j in selector.selected_]
                transformed=pre.transform(selector.transform(X[ids]))
                if sparse.issparse(transformed): transformed=transformed.toarray()
                transformed=np.asarray(transformed,dtype=float)
                ct=pre.transformer_
                feature_names=list(ct.get_feature_names_out(selected_names))
                if len(feature_names)!=transformed.shape[1]:
                    raise ValueError('Transformed feature names do not match model feature count')
                # Use ColumnTransformer output_indices_ and fitted OneHotEncoder categories
                # to map each encoded column back to its original input predictor.
                provenance=[None]*transformed.shape[1]
                for trans_name, transformer, original_cols in ct.transformers_:
                    if trans_name=='remainder' or transformer=='drop': continue
                    sl=ct.output_indices_[trans_name]
                    idxs=list(range(sl.start,sl.stop))
                    orig_names=[selected_names[int(j)] if isinstance(j,(int,np.integer)) else str(j) for j in original_cols]
                    if trans_name=='cat':
                        if not hasattr(transformer,'categories_'): raise ValueError('OneHotEncoder provenance unavailable')
                        offsets=[len(cat) for cat in transformer.categories_]
                        if sum(offsets)!=len(idxs): raise ValueError('Unexpected OneHotEncoder output shape')
                        cursor=0
                        for orig,width in zip(orig_names,offsets):
                            for j in idxs[cursor:cursor+width]:provenance[j]=orig
                            cursor+=width
                    elif len(idxs)==len(orig_names):
                        for j,orig in zip(idxs,orig_names):provenance[j]=orig
                    else: raise ValueError('Unsupported transformer provenance')
                if any(x is None for x in provenance): raise ValueError('Incomplete encoded-feature provenance')
                if hasattr(model,'feature_importances_'):
                    explainer=shap.TreeExplainer(model,model_output='raw')
                elif hasattr(model,'coef_'):
                    bg=pre.transform(selector.transform(X))
                    if sparse.issparse(bg): bg=bg.toarray()
                    bg=np.asarray(bg,dtype=float)
                    bg=bg[rng_shap.choice(len(bg),size=min(200,len(bg)),replace=False)]
                    explainer=shap.LinearExplainer(model,bg)
                else: raise ValueError('Estimator does not support the scientific SHAP explainers')
                explanation=explainer(transformed)
                labels=[str(x) for x in encoder.classes_] if classification else ['regression']
                audit_shap,class_rows,group_rows,global_rows,sample_rows,values=shap_scientific(
                    model,transformed,explainer,explanation,classification,labels,feature_names,provenance)
                audit_shap['n_samples']=len(ids)
                audit_shap['n_transformed_features']=len(feature_names)
                audit_shap['sample_indices_in_training']=ids.tolist()
                audit_shap['classwise_output_semantics']='For binary 2D raw-margin explanations, the reported class is the positive encoded class (index 1); negative class SHAP cannot be inferred in raw-margin space.'
                np.savez_compressed(os.path.join(out,'SHAP_Values.npz'),values=values,sample_indices=ids,
                                    feature_names=np.asarray(feature_names),original_predictors=np.asarray(provenance))
                for r in sample_rows:r['sample_index']=int(ids[r['sample_index']])
                write_csv(os.path.join(out,'SHAP_Global_Importance.csv'),global_rows,['predictor','mean_abs_shap'])
                write_csv(os.path.join(out,'SHAP_Classwise_Importance.csv'),class_rows,
                          ['class_label','feature','original_predictor','mean_abs_shap','mean_shap'])
                write_csv(os.path.join(out,'SHAP_Grouped_Classwise_Importance.csv'),group_rows,
                          ['class_label','predictor','n_encoded_features','mean_abs_shap','mean_shap'])
                write_csv(os.path.join(out,'SHAP_Samples.csv'),sample_rows,
                          ['sample_index','class_label','feature','original_predictor','feature_value','shap_value'])
                ranked=sorted(global_rows,key=lambda r:r['mean_abs_shap'],reverse=True)[:20]
                fig,ax=plt.subplots(figsize=(9,max(4,len(ranked)*.36+1)))
                ax.barh([r['predictor'] for r in ranked[::-1]],[r['mean_abs_shap'] for r in ranked[::-1]])
                ax.set_xlabel('Mean |SHAP| (raw model output space)')
                ax.set_title('Grouped SHAP Importance')
                fig.tight_layout();fig.savefig(os.path.join(out,'SHAP_Summary.png'),dpi=160);plt.close(fig)
                with open(os.path.join(out,'SHAP_Audit.json'),'w',encoding='utf-8') as f:
                    json.dump(audit_shap,f,indent=2,ensure_ascii=False)
                report['shap_status']='EXPORTED_ADDIVITY_'+audit_shap['additivity_check']
                report['shap_audit']=audit_shap
                if audit_shap['additivity_check']!='PASS':
                    feedback.pushWarning('SHAP additivity not verified: '+audit_shap['additivity_check'])
            except Exception as exc:
                report['shap_status']='UNAVAILABLE: '+str(exc)
                feedback.pushWarning(report['shap_status'])
        ood_path=os.path.join(out,'OOD_Standardized_Distance.tif');self._raster(ood_path,ood,gt,prj)
        pred_path=os.path.join(out,'Prediction.tif'); self._raster(pred_path,prediction,gt,prj)
        result={'PREDICTION':pred_path,'OOD_DIAGNOSTIC':ood_path}
        if classification:
            cp=os.path.join(out,'Class_Confidence.tif'); self._raster(cp,confidence,gt,prj); result['CONFIDENCE']=cp
        if risk_score is not None:
            score_path=os.path.join(out,'Risk_Score.tif')
            self._raster(score_path,risk_score,gt,prj)
            # Descriptive score bands only, NOT observed fire probabilities or validated thresholds.
            levels=np.where(np.isfinite(risk_score),np.minimum(5,np.floor(np.minimum(risk_score,0.999999)*5).astype(np.int16)+1),np.nan)
            levels_path=os.path.join(out,'Risk_Level.tif')
            self._raster(levels_path,levels,gt,prj)
            result['RISK_SCORE']=score_path
            result['RISK_LEVEL']=levels_path
            report['risk_levels']={'class_label':'1','score_definition':('Spatial-OOF sigmoid-calibrated score for event class 1' if calibrator is not None else 'Uncalibrated classifier score for event class 1'),'levels':{'1':'0.0–<0.2 very low','2':'0.2–<0.4 low','3':'0.4–<0.6 moderate','4':'0.6–<0.8 high','5':'0.8–1.0 very high'},'warning':'Descriptive fixed score bands, not empirically validated hazard thresholds. Check independent test event rates and calibration diagnostics.'}
        else:
            report['risk_levels']='Not exported: requires binary classification with explicit original event class label 1; multiclass risk target selection is not yet supported.'
        report['ood_definition']='Maximum absolute standardized deviation from selected training feature means; training quantile threshold is descriptive, not externally validated'
        report_path=os.path.join(out,'Research_Report.json')
        with open(report_path,'w',encoding='utf-8') as f: json.dump(report,f,indent=2,ensure_ascii=False)
        result['PREDICTOR_QC']=os.path.join(out,'Predictor_QC.csv'); result['REPORT']=report_path; result['MODEL']=os.path.join(out,'Best_Model.joblib'); result['CV']=os.path.join(out,'CV_Results.csv')
        return result
