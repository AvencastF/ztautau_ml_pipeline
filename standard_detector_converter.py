"""Standard numerical EveNet shards; reconstructed source stays in companions."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from detector_inputs import extract,apply_normalization,grouped_config,RAW_NAMES,FINGERPRINT
from detector_classification import targets,contract,DEFAULT_CONFIG,read_yaml,training_config,backend_normalization
from detector_event_inputs import *

class Moments:
    def __init__(self,n):self.count=np.zeros(n,np.int64);self.mean=np.zeros(n);self.m2=np.zeros(n)
    def add(self,x,mask):
        for i in range(len(self.count)):
            v=x[...,i][mask[...,i]];n=len(v)
            if not n:continue
            mu=v.mean();delta=mu-self.mean[i];total=self.count[i]+n
            self.m2[i]+=np.square(v-mu).sum()+delta**2*self.count[i]*n/total
            self.mean[i]+=delta*n/total;self.count[i]=total
    def result(self):
        std=np.sqrt(np.divide(self.m2,self.count,out=np.zeros_like(self.mean),where=self.count>0));std[std==0]=1
        return {'mean':self.mean.tolist(),'std':std.tolist(),'count':self.count.tolist()}

def input_config(neutrino=True):
    c=grouped_config();c['Inputs']['Global']['Fields']=list(GLOBAL_NAMES)
    c['Normalization']['Global']={n:'none' for n in GLOBAL_NAMES}
    if neutrino:c['Normalization']['Invisible']={'theta':'normalize','phi':'normalize'}
    return c

def arrays(rows,normal,capacity,config):
    e=extract(rows,capacity);labels,reasons=targets(rows,config);neutrino=normal['provenance']['contract'].get('tasks','classification_neutrino')=='classification_neutrino'
    invisible,mask,target_reasons=angular_targets(rows,labels['classification'],contract(config)['classes'],config) if neutrino else (np.zeros((len(rows),2,2),np.float32),np.zeros((len(rows),2),bool),['task_disabled']*len(rows))
    # Unsupported train-fit dimensions cannot become silently unscaled targets.
    if neutrino and not all(np.asarray(normal['invisible']['count'])>0):
        target_reasons=['normalization_unsupported' if m.any() else reason for m,reason in zip(mask,target_reasons)];mask[:]=False;invisible[:]=0
    cond,cm=conditions(rows,normal['global']);num=e['x_mask'].sum(1).astype(np.float32)
    a={'x':apply_normalization(e,normal),'x_mask':e['x_mask'],'conditions':cond,'conditions_mask':cm,'num_vectors':num,'num_sequential_vectors':num,
       'classification':labels['classification'],'event_weight':np.array([r['weight'] for r in rows],np.float32),'central_weight':np.array([r['weight'] for r in rows],np.float32),
       'x_invisible':invisible,'x_invisible_mask':mask,'num_invisible_raw':np.full(len(rows),2,np.int64),'num_invisible_valid':mask.sum(1)}
    if not neutrino:
        for key in ('x_invisible','x_invisible_mask','num_invisible_raw','num_invisible_valid'):a.pop(key)
    return a,labels['classification_eligible'],reasons,target_reasons

def main():
    from build_evenet_input_from_parquet import batches,fit_tables
    root=Path(__file__).parent;sys.path.insert(0,str(root/'EveNet-Align'))
    from evenet.dataset.preprocess import flatten_dict
    from generate_event_info_yaml import parse_feature_config,parse_evenet_config,build_event_info_payload
    import torch
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['fit','apply']);p.add_argument('--input',nargs='+',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--normalization',required=True,type=Path)
    p.add_argument('--role',required=True,choices=['train','validation','test','data','smoke']);p.add_argument('--split-id',required=True);p.add_argument('--region',required=True);p.add_argument('--max-particles',required=True,type=int);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--max-events',type=int);p.add_argument('--analysis-config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--validation-features',type=Path)
    p.add_argument('--tasks',choices=['classification','classification_neutrino'],default='classification_neutrino')
    a=p.parse_args();neutrino=a.tasks=='classification_neutrino';cfg=read_yaml(a.analysis_config);cc=contract(cfg)
    if a.output.exists():p.error('Fresh output required')
    if min(a.max_particles,a.batch_size)<=0 or (a.max_events is not None and a.max_events<=0):p.error('Positive bounds required')
    spec={'region':a.region,'max_particles':a.max_particles,'feature_fingerprint':FINGERPRINT,'global_names':list(GLOBAL_NAMES),'target_version':TARGET_VERSION if neutrino else None,'tasks':a.tasks}
    provenance={'input_files':[{'path':str(f.resolve()),'bytes':f.stat().st_size,'rows':pq.ParquetFile(f).metadata.num_rows,'schema_sha256':hashlib.sha256(pq.ParquetFile(f).schema_arrow.serialize().to_pybytes()).hexdigest()} for f in a.input],'role':a.role,'split_id':a.split_id,'max_events':a.max_events,'batch_size':a.batch_size,'contract':spec}
    def tables():return batches(a.input,a.region,a.batch_size,a.max_events)
    def eligible_tables():
        for t in tables():
            labels,_=targets(t.to_pylist(),cfg);t=t.filter(pa.array(labels['classification_eligible']))
            if len(t):yield t
    if a.mode=='fit':
        if a.role!='train' or a.normalization.exists():p.error('Fresh train-only normalization required')
        # Reject data even if it would otherwise be filtered from fitting.
        for t in tables():
            if any(r.get('process')=='data' or r.get('is_data') is True or r.get('sample_is_data') is True for r in t.to_pylist()):p.error('Data forbidden in train-fit sources')
        n=fit_tables(eligible_tables(),a.max_particles,provenance,cfg);g=Moments(4);iv=Moments(2)
        for t in eligible_tables():
            rows=t.to_pylist();v,m=reconstructed_conditions(rows);g.add(v,m);
            if neutrino:
                labels,_=targets(rows,cfg);v,m,_=angular_targets(rows,labels['classification'],cc['classes'],cfg);iv.add(v,np.repeat(m[...,None],2,axis=-1))
        n.update(global_=g.result());n['global']=n.pop('global_');
        if neutrino:n['invisible']=iv.result();n['target_version']=TARGET_VERSION
        a.normalization.parent.mkdir(parents=True,exist_ok=True);a.normalization.write_text(json.dumps(n,indent=2)+'\n')
    else:
        n=json.loads(a.normalization.read_text())
        if n['provenance']['contract']!=spec or n['classification_contract']!=cc:p.error('Frozen normalization contract mismatch')
    a.output.mkdir(parents=True);companion=a.output.parent/(a.output.name+'-source');companion.mkdir(exist_ok=False)
    reason_counts={};target_counts={};shards=[];source_rows=0;shapes={}
    for source_index,source in enumerate(tables()):
        rows=source.to_pylist();data,eligible,reasons,why=arrays(rows,n,a.max_particles,cfg)
        name=f'part-{source_index:05d}.parquet'
        # Original Arrow types and metadata preserved, plus explicit rejection provenance.
        source=source.append_column('converter_classification_reason',pa.array(reasons)).append_column('converter_angular_target_reason',pa.array(why));pq.write_table(source,companion/name,compression='zstd');source_rows+=len(source)
        for r in reasons:reason_counts[r]=reason_counts.get(r,0)+1
        for r in why:target_counts[r]=target_counts.get(r,0)+1
        keep=eligible if a.role in ('train','validation') else np.ones(len(rows),bool)
        if not keep.any():continue
        data={k:v[keep] for k,v in data.items()};data['EXTRA/source_shard']=np.full(keep.sum(),source_index,np.int64);data['EXTRA/source_row']=np.flatnonzero(keep);table,shapes=flatten_dict(data);pq.write_table(table,a.output/name,compression='zstd');shards.append({'path':name,'rows':len(table),'source_path':name,'source_row_indices':np.flatnonzero(keep).tolist()})
    if not shards:raise ValueError('No eligible numerical rows; companions preserved')
    inputs=input_config(neutrino);fc=parse_feature_config(inputs);schema=parse_evenet_config(read_yaml(root/'config/evenet_schema.yaml'),inputs,fc);event=build_event_info_payload(cc['classes'],fc,schema)
    event['GENERATIONS']={'Conditions':[],'GlobalTargets':[],'Events':[],'Neutrinos':{'theta':'normalize','phi':'normalize'} if neutrino else {}}
    backend=backend_normalization(n,cc);backend['input_mean']['Conditions']=torch.zeros(len(GLOBAL_NAMES));backend['input_std']['Conditions']=torch.ones(len(GLOBAL_NAMES))
    if neutrino:backend['invisible_mean']={'Source':torch.tensor(n['invisible']['mean'],dtype=torch.float32)};backend['invisible_std']={'Source':torch.tensor(n['invisible']['std'],dtype=torch.float32)}
    torch.save(backend,a.output/'normalization.pt');train=training_config(a.output,cc,a.validation_features)
    defaults=read_yaml(root/'config/train_pretrain.yaml')['options']['Training']['Components']
    if neutrino:
        for k in ('TruthGeneration','InvisibleInputProjector'):train['options']['Training']['Components'][k]=defaults[k]
    train['options']['Training']['strategy']={'name':'supervised'}
    train['options']['Training']['ProgressiveTraining']['stages']=[{'name':'supervised','epoch_ratio':1.,'transition_ratio':0.,'loss_weights':{'classification':[1.,1.],**({'generation-truth':[1.,1.]} if neutrino else {})}}]
    for name,value in [('event_info.yaml',event),('analysis_detector.yaml',inputs),('train_supervised.yaml',train)]: (a.output/name).write_text(yaml.safe_dump(value,sort_keys=False))
    (a.output/'shape_metadata.json').write_text(json.dumps(shapes,indent=2)+'\n')
    manifest={**provenance,'fingerprint':FINGERPRINT,'sequential_features':list(RAW_NAMES),'classification_contract':cc,'normalization':str(a.normalization.resolve()),'normalization_sha256':hashlib.sha256(a.normalization.read_bytes()).hexdigest(),'source_companions':str(companion.resolve()),'source_rows':source_rows,'rows':sum(s['rows'] for s in shards),'shards':shards,'classification_reasons':reason_counts,'angular_target_reasons':target_counts}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()

def validate_training_inputs(config_path):
    """Top-level preflight; stock runtime remains unchanged after this check."""
    sys.path.insert(0,str(Path(__file__).parent/'EveNet-Align'))
    cfg=read_yaml(Path(config_path));train=Path(cfg['platform']['data_parquet_dir']);val_path=cfg['platform'].get('data_parquet_val_dir')
    if not val_path:raise ValueError('Explicit pre-separated validation directory required')
    valid=Path(val_path)
    if train.resolve()==valid.resolve():raise ValueError('Training and validation must differ')
    manifests=[json.loads((d/'manifest.json').read_text()) for d in (train,valid)]
    cc=contract(read_yaml(DEFAULT_CONFIG));normal_path=Path(manifests[0]['normalization']);normal_bytes=normal_path.read_bytes();normal=json.loads(normal_bytes)
    fingerprint=hashlib.sha256(normal_bytes).hexdigest();shapes=[json.loads((d/'shape_metadata.json').read_text()) for d in (train,valid)]
    if shapes[0]!=shapes[1]:raise ValueError('Train/validation tensor shapes differ')
    if manifests[0]['split_id']==manifests[1]['split_id']:raise ValueError('Distinct split designations required')
    expected_roles=('train','validation')
    identity_sets=[]
    for directory,m,role in zip((train,valid),manifests,expected_roles):
        if m['role']!=role or m['rows']<=0:raise ValueError('Supervised train/validation role required')
        if m['fingerprint']!=FINGERPRINT or m['sequential_features']!=list(RAW_NAMES) or m['classification_contract']!=cc:raise ValueError('Feature/class contract mismatch')
        if m['contract']!=normal['provenance']['contract'] or m['normalization_sha256']!=fingerprint:raise ValueError('Frozen normalization contract mismatch')
        if shapes[0]['x']!=[m['contract']['max_particles'],len(RAW_NAMES)] or shapes[0]['conditions']!=[len(GLOBAL_NAMES)]:raise ValueError('Shape/capacity mismatch')
        if {p.name for p in directory.glob('*.parquet')}!={s['path'] for s in m['shards']}:raise ValueError('Parquet file set differs from manifest')
        identities=set();count=0
        for shard in m['shards']:
            path=directory/shard['path'];labels=pq.read_table(path,columns=['classification']).column(0).to_pylist()
            if any(type(x)!=int or not 0<=x<len(cc['classes']) for x in labels):raise ValueError('Ineligible supervised classification')
            source=pq.read_table(Path(m['source_companions'])/shard['source_path'],columns=['identity','process'])
            indices=shard['source_row_indices']
            if len(labels)!=len(indices) or len(set(indices))!=len(indices):raise ValueError('Invalid source row mapping')
            for i in indices:
                row=source.slice(i,1).to_pylist()[0]
                if row['process']=='data':raise ValueError('Data cannot enter supervised training')
                key=json.dumps(decode(row['identity']),sort_keys=True,separators=(',',':'))
                if key in identities:raise ValueError('Duplicate identity within split')
                identities.add(key)
            count+=len(labels)
        if count!=m['rows']:raise ValueError('Shard row count mismatch')
        identity_sets.append(identities)
    if identity_sets[0]&identity_sets[1]:raise ValueError('Train/validation identity overlap')
    event=read_yaml(Path(cfg['event_info']['default']))
    for names,actual in ((RAW_NAMES,event['INPUTS']['SEQUENTIAL']['Source']),(GLOBAL_NAMES,event['INPUTS']['GLOBAL']['Conditions'])):
        if list(actual)!=list(names) or any(v!='none' for v in actual.values()):raise ValueError('Backend feature transform/order mismatch')
    if event['CLASSLABEL']['EVENT']['signal']!=[cc['classes']]:raise ValueError('Configured class order mismatch')
    neutrino=normal['provenance']['contract'].get('tasks','classification_neutrino')=='classification_neutrino'
    if event['GENERATIONS']['Neutrinos']!=({'theta':'normalize','phi':'normalize'} if neutrino else {}):raise ValueError('Angular target feature contract mismatch')
    if cfg['options']['Training'].get('strategy',{}).get('name')!='supervised':raise ValueError('Supervised strategy required')
    components=cfg['options']['Training']['Components']
    if not components['Classification']['include'] or bool(components['TruthGeneration']['include'])!=neutrino or any(components[k]['include'] for k in ('Regression','Assignment','Segmentation','GlobalGeneration','ReconGeneration')):raise ValueError('Only supervised classification and TruthGeneration supported')
    import torch
    actual=torch.load(cfg['options']['Dataset']['normalization_file'],map_location='cpu',weights_only=True)
    expected=backend_normalization(normal,cc);expected['input_mean']['Conditions']=torch.zeros(len(GLOBAL_NAMES));expected['input_std']['Conditions']=torch.ones(len(GLOBAL_NAMES))
    if neutrino:expected['invisible_mean']={'Source':torch.tensor(normal['invisible']['mean'],dtype=torch.float32)};expected['invisible_std']={'Source':torch.tensor(normal['invisible']['std'],dtype=torch.float32)}
    def same(a,b):
        if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        return a==b
    if not same(actual,expected):raise ValueError('Configured engine normalization mismatch')
    return {'train_rows':len(identity_sets[0]),'validation_rows':len(identity_sets[1]),'disjoint':True,'normalization_sha256':fingerprint}
