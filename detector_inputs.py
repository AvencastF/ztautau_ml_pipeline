"""Missingness-aware MAIN inputs for the additive tau_analysis Parquet schema.

No truth labels, selections, detector calibrations or training are inferred here.
"""
from __future__ import annotations
import hashlib,json,math
import numpy as np

MAIN='sDST_MAIN_Particles'
PID={
 'sDST_ELID_ElectronID':('tag','conversion_tag'),
 'sDST_MUID_MuonID':('raw_word','refit_chi2','hit_pattern'),
 'sDST_RPROCO_CombinedTags':('pion','kaon','proton','heavy','electron','quality_flag'),
 'sDST_MTPC_dEdxExtended':('dedx80','sigma80','dedx65','sigma65','dedx_integrated','npads','nwires','nsaturated','nempty','padrow_pattern'),
 'sDST_HAID_HadronID':('kaon_rich','proton_rich','pion_rich','kaon_dedx','proton_dedx','x_kaon','x_proton','rich_quality','gas_angle','gas_angle_error','gas_nphotons','gas_expected_photons','gas_flag','liquid_angle','liquid_angle_error','liquid_nphotons','liquid_expected_photons','liquid_flag'),
}
CALO=('sDST_EMNC_Showers','sDST_HCNC_Showers','xsDST_HCAL_Showers','sDST_SSTC_Showers')
STATUS=('linked','no_association','unavailable','malformed','ambiguous','empty')
MOMENTUM=('Part_energy','Part_pt','Part_eta','Part_phi')
FEATURES=(*MOMENTUM,'Part_charge','Part_mass',*(f'{c}.{n}' for c,ns in PID.items() for n in ns),'track.chi2','track.ndf','track.count','MAIN.TrackLength','MAIN.DetectorMask',*(f'{c}.{n}' for c in CALO for n in ('count','energy_sum','position.x_mean','position.y_mean','position.z_mean')))
CHANNELS=('stored_valid','measurement_known','measurement_valid','normalization_supported')
STATUS_SOURCES=(*PID,'track',*CALO)
BASE_NAMES=tuple(FEATURES)+tuple(f'{n}__{m}' for n in FEATURES for m in CHANNELS)+tuple(f'{c}__status_{s}' for c in STATUS_SOURCES for s in STATUS)
RAW_NAMES=(*MOMENTUM,*(n for n in BASE_NAMES if n not in (*MOMENTUM,'Part_charge')),'Part_charge')
FINGERPRINT=hashlib.sha256(json.dumps({'version':1,'features':RAW_NAMES,'gate':'stored_and_not_measurement_false','normalization':'masked_unweighted_train_only_charge_passthrough_zero_count_unsupported'},separators=(',',':')).encode()).hexdigest()

def datum(value=None,measurement=None):
    valid=isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    return {'value':value if valid else None,'stored_valid':valid,'measurement_valid':measurement}

def scalar(d):
    value=d.get('value'); valid=d.get('stored_valid') is True and isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    semantic=d.get('measurement_valid')
    if semantic is not None and type(semantic) is not bool: raise ValueError('Non-scalar semantic validity')
    return float(value) if valid else 0.,valid,semantic is not None,semantic is True

def semantic(ds):
    states=[d.get('measurement_valid') for d in ds]
    return False if any(v is False for v in states) else True if all(v is True for v in states) else None

def linked_objects(record):
    found={c:[] for c in CALO};bad=set();seen=set();unknown='unavailable' if 'clusters' not in record.get('associations',{}) else None
    def walk(r):
        nonlocal unknown
        for name,association in r.get('associations',{}).items():
            if name=='clusters' and association.get('status') in ('malformed','unavailable'):
                unknown='malformed' if association.get('status')=='malformed' else unknown or 'unavailable'
            for link in association.get('records') or []:
                c=link.get('collection');key=(c,link.get('index'))
                if name=='clusters' and link.get('status')=='malformed':unknown='malformed'
                if c in found:
                    if link.get('status')!='linked' or not link.get('payload'):bad.add(c);continue
                    if key not in seen:found[c].append(link['payload']);seen.add(key)
                if link.get('payload'):walk(link['payload'])
    walk(record)
    return found,bad,unknown

def particle_features(p,collections):
    fields=p['main']['fields']; values={n:{} for n in FEATURES};statuses={}
    for name,source in [('Part_energy','energy'),('Part_charge','charge'),('Part_mass','mass')]:values[name]=fields.get(source,{})
    xyz=[scalar(fields.get('momentum.'+c,{})) for c in 'xyz']
    if all(v[1] for v in xyz):
        x,y,z=[v[0] for v in xyz];pt=math.hypot(x,y)
        meaning=semantic([fields.get('momentum.'+c,{}) for c in 'xyz'])
        values['Part_pt']=datum(pt,meaning);values['Part_phi']=datum(math.atan2(y,x),meaning) if pt>0 else datum()
        values['Part_eta']=datum(math.asinh(z/pt),meaning) if pt>0 else datum()
    for c,names in PID.items():
        group=p.get('detector_rows',{}).get(c,{});records=group.get('records') or [];status=group.get('status','unavailable')
        if len(records)>1 and status!='malformed':status='ambiguous'
        statuses[c]=status
        if len(records)==1 and status=='linked':
            for name in names:values[c+'.'+name]=records[0].get('named_parameters',{}).get(name,{})
    track=p['main'].get('associations',{}).get('tracks',{});records=track.get('records');status=track.get('status','unavailable')
    if status=='linked' and any(r.get('status')!='linked' or not r.get('payload') for r in records or []):status='malformed'
    if status=='linked' and len(records or [])>1:status='ambiguous'
    statuses['track']=status
    if status in ('linked','no_association','ambiguous') and records is not None:values['track.count']=datum(len(records))
    if status=='linked' and len(records or [])==1:
        for name in ('chi2','ndf'):values['track.'+name]=(records[0].get('payload') or {}).get('fields',{}).get(name,{})
    for name in ('TrackLength','DetectorMask'):values['MAIN.'+name]=p.get('parallel',{}).get(MAIN+'_'+name,{})
    calos,bad,unknown=linked_objects(p['main'])
    for c,records in calos.items():
        # With no linked shower, collection-level availability is not established.
        inventory=collections.get(c,{})
        statuses[c]=('malformed' if c in bad else unknown if unknown else 'linked' if records else 'unavailable' if not inventory.get('present') else 'malformed' if inventory.get('count') is None else 'empty' if inventory['count']==0 else 'no_association')
        if statuses[c] in ('no_association','empty'):values[c+'.count']=datum(0)
        if statuses[c]!='linked':continue
        values[c+'.count']=datum(len(records))
        for name in ('energy','position.x','position.y','position.z'):
            ds=[r.get('fields',{}).get(name,{}) for r in records]
            if all(scalar(d)[1] for d in ds):
                nums=[scalar(d)[0] for d in ds]
                key='energy_sum' if name=='energy' else name+'_mean'
                values[c+'.'+key]=datum(math.fsum(nums) if name=='energy' else math.fsum(nums)/len(nums),semantic(ds))
    raw=np.array([scalar(values[n])[0] for n in FEATURES],dtype=np.float64)
    masks=np.array([scalar(values[n])[1:] for n in FEATURES],dtype=bool)
    usable=masks[:,0]&(~masks[:,1]|masks[:,2])
    flags=np.c_[masks.astype(np.float64),np.zeros(len(FEATURES))].ravel()
    status_values=[]
    for c in STATUS_SOURCES:
        s=statuses.get(c,'unavailable')
        if s not in STATUS:raise ValueError('Unknown association status '+str(s))
        status_values.extend(s==v for v in STATUS)
    return raw,masks,usable,np.r_[flags,status_values]

def extract(rows,max_particles):
    raw=np.zeros((len(rows),max_particles,len(FEATURES)));masks=np.zeros(raw.shape+(3,),bool);usable=np.zeros(raw.shape,bool)
    flags=np.zeros((len(rows),max_particles,len(RAW_NAMES)-len(FEATURES)));present=np.zeros((len(rows),max_particles),bool);indices=np.full(present.shape,-1,np.int64)
    for e,row in enumerate(rows):
        payload=json.loads(row['particle_detector'])
        if payload.get('version')!=1 or payload.get('status')!='available':raise ValueError('Unsupported/unavailable detector payload')
        ps=payload['particles']; ids=[p['main_index'] for p in ps]
        if sorted(ids)!=list(range(len(ps))) or any(p['main']['index']!=p['main_index'] for p in ps):raise ValueError('MAIN alignment mismatch')
        if len(ps)>max_particles:raise ValueError('Particle capacity exceeded; no truncation permitted')
        for p in ps:
            i=p['main_index'];raw[e,i],masks[e,i],usable[e,i],flags[e,i]=particle_features(p,payload.get('collections',{}));present[e,i]=True;indices[e,i]=i
    return {'raw_values':raw,'feature_masks':masks,'usable':usable,'flags':flags,'x_mask':present,'main_indices':indices}

def fit_normalization(extracted,*,role,provenance):
    if role!='train':raise ValueError('Normalization fit is train-only')
    x=extracted['raw_values'];mask=extracted['usable']&extracted['x_mask'][...,None];means=[];stds=[];counts=[]
    for i in range(len(FEATURES)):
        v=x[...,i][mask[...,i]];counts.append(len(v));means.append(float(v.mean()) if len(v) else 0.);s=float(v.std()) if len(v) else 1.;stds.append(s if s>0 else 1.)
    means[FEATURES.index('Part_charge')]=0.;stds[FEATURES.index('Part_charge')]=1.
    return {'version':1,'fingerprint':FINGERPRINT,'feature_names':list(RAW_NAMES),'mean':means,'std':stds,'count':counts,'fit_role':role,'provenance':provenance,'weighted':False}

def apply_normalization(extracted,normalization):
    if normalization['fingerprint']!=FINGERPRINT or normalization['feature_names']!=list(RAW_NAMES) or normalization['fit_role']!='train':raise ValueError('Normalization contract mismatch')
    mean=np.array(normalization['mean']);std=np.array(normalization['std'])
    if mean.shape!=(len(FEATURES),) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():raise ValueError('Invalid normalization')
    supported=np.asarray(normalization['count'])>0
    if supported.shape!=mean.shape:raise ValueError('Invalid support counts')
    values=np.where(extracted['usable']&supported,(extracted['raw_values']-mean)/std,0.)
    flags=extracted['flags'].copy(); flags[...,:len(FEATURES)*len(CHANNELS)].reshape(*flags.shape[:2],len(FEATURES),len(CHANNELS))[...,3]=supported
    x=np.concatenate((values,flags),axis=-1)[...,[BASE_NAMES.index(n) for n in RAW_NAMES]]*extracted['x_mask'][...,None]
    with np.errstate(over='ignore'):x=x.astype(np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite tensor; do not clip silently')
    return x

def grouped_config():
    momentum=list(MOMENTUM);charge=['Part_charge'];detector=[n for n in RAW_NAMES if n not in (*momentum,*charge)]
    return {'Inputs':{'Part':{'Momentum':{'index':[0,1,2,3],'output_dim':4,'input':momentum},'ObjectTag':{'index':[4,5],'output_dim':2,'input':detector},'Charge':{'index':[6],'output_dim':1,'input':charge}},'Global':{'Fields':[]}},'Normalization':{'Sequential':{n:'none' for n in RAW_NAMES}}}


def load_tensor_batch(table,shapes,unflatten,drop_column_prefix=None):
    """Use EveNet's unflatten for tensors; restore explicit zero-width globals.

    Only model tensor columns are loaded. Source/truth sidecars cannot leak into x.
    Pass this as process_event_batch's unflatten callable via functools.partial.
    """
    data={};empty={}
    columns=table.column_names if hasattr(table,'column_names') else table.keys()
    n=len(table) if hasattr(table,'column_names') else len(next(iter(table.values())))
    for key,shape in shapes.items():
        if shape and math.prod(shape)==0:
            empty[key]=np.empty((n,*shape),np.float32);continue
        names=[n for n in columns if n==key or n.startswith(key+':')]
        for name in names:data[name]=table[name].to_numpy() if hasattr(table[name],'to_numpy') else table[name]
    result=unflatten(data,{k:v for k,v in shapes.items() if k not in empty})
    result.update(empty)
    return result
