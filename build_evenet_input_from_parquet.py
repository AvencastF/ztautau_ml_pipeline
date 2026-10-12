#!/usr/bin/env python3
"""Fit/apply detector feature shards; task targets require explicit integration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from detector_classification import targets,contract as classification_contract,DEFAULT_CONFIG,read_yaml
from detector_inputs import extract,apply_normalization,grouped_config,RAW_NAMES,FEATURES,FINGERPRINT


def batches(paths,region,batch_size,max_events):
    count=0
    for path in paths:
        source=pq.ParquetFile(path);meta=source.schema_arrow.metadata or {}
        if b'regions' not in meta:raise ValueError('Saved regions metadata required')
        regions=json.loads(meta[b'regions']);assert isinstance(regions,list) and len(set(regions))==len(regions)
        if region!='all' and region not in regions:raise ValueError('Unknown saved region '+region)
        for batch in source.iter_batches(batch_size=batch_size):
            table=pa.Table.from_batches([batch],schema=source.schema_arrow)
            if region!='all':table=table.filter(pc.not_equal(pc.bit_wise_and(table['region_mask'],pa.scalar(1<<regions.index(region),pa.uint64())),0))
            if max_events is not None:table=table.slice(0,max_events-count)
            if len(table):yield table;count+=len(table)
            if max_events is not None and count>=max_events:return


def tensor_table(source,x,extracted,classification=None):
    if any(n=='x' or n.startswith('x:') for n in source.column_names):raise ValueError('Already converted input')
    n=len(source);num=extracted['x_mask'].sum(axis=1).astype(np.float32)
    arrays={'x':x,'x_mask':extracted['x_mask'],'main_indices':extracted['main_indices'],'raw_detector_values':extracted['raw_values'],'detector_feature_masks':extracted['feature_masks'],'num_sequential_vectors':num,'num_vectors':num,'conditions':np.empty((n,0),np.float32),'conditions_mask':np.zeros((n,1),bool),'event_weight':source['weight'].to_numpy(),'central_weight':source['weight'].to_numpy()}
    if classification is not None:arrays.update(classification)
    if any(name.split(':')[0] in arrays for name in source.column_names):raise ValueError('Tensor/source column collision')
    shapes={}; columns=list(source.columns); names=list(source.column_names);seen=set(names)
    for key,arr in arrays.items():
        shapes[key]=list(arr.shape[1:]);flat=arr.reshape(n,int(np.prod(arr.shape[1:])))
        if flat.shape[1]==0:
            if key in seen:raise ValueError('Tensor/source column collision')
            # Empty globals roundtrip via zero-width shape metadata, no fake token.
            columns.append(pa.array([[]]*n,type=pa.list_(pa.float32())));names.append(key)
        for i in range(flat.shape[1]):
            suffix=':'+':'.join(map(str,np.unravel_index(i,arr.shape[1:]))) if arr.ndim>1 else ''
            if key+suffix in seen:raise ValueError('Tensor/source column collision')
            columns.append(pa.array(flat[:,i]));names.append(key+suffix)
    return pa.Table.from_arrays(columns,names=names).replace_schema_metadata(source.schema.metadata),shapes


def fit_tables(tables,capacity,provenance,analysis_config=None):
    count=np.zeros(len(FEATURES),np.int64);mean=np.zeros(len(FEATURES));m2=np.zeros(len(FEATURES))
    identity_digest=hashlib.sha256();fit_rows=0
    class_counts=np.zeros(len(classification_contract(analysis_config)['classes']),np.int64) if analysis_config is not None else None
    for table in tables:
        if 'process' not in table.column_names:raise ValueError('Fit requires explicit saved process column')
        if any(v=='data' for v in table['process'].to_pylist()) or any(any(table[k].to_pylist()) for k in ('is_data','sample_is_data') if k in table.column_names):raise ValueError('Data cannot fit normalization')
        for r in table.select(['identity','process']).to_pylist():identity_digest.update(json.dumps(r,sort_keys=True).encode()+b'\n');fit_rows+=1
        if analysis_config is not None:
            labels,_=targets(table.to_pylist(),analysis_config);class_counts+=np.bincount(labels['classification'][labels['classification_eligible']],minlength=len(class_counts))
        e=extract(table.to_pylist(),capacity)
        for i in range(len(FEATURES)):
            v=e['raw_values'][...,i][e['usable'][...,i]];n=len(v)
            if not n:continue
            mu=float(v.mean());ss=float(np.square(v-mu).sum());total=count[i]+n;delta=mu-mean[i]
            m2[i]+=ss+delta*delta*count[i]*n/total;mean[i]+=delta*n/total;count[i]=total
    if not count.any():raise ValueError('No usable training features')
    std=np.sqrt(np.divide(m2,count,out=np.zeros_like(mean),where=count>0));std[std==0]=1
    mean[FEATURES.index('Part_charge')]=0;std[FEATURES.index('Part_charge')]=1
    provenance={**provenance,'fit_identity_sha256':identity_digest.hexdigest(),'fit_rows':fit_rows}
    return {'version':1,'fingerprint':FINGERPRINT,'feature_names':list(RAW_NAMES),'raw_value_names':list(FEATURES),'mean':mean.tolist(),'std':std.tolist(),'count':count.tolist(),'fit_role':'train','provenance':provenance,'weighted':False,**({'classification_counts':class_counts.tolist(),'classification_contract':classification_contract(analysis_config)} if class_counts is not None else {})}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','apply']);p.add_argument('--input',nargs='+',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--normalization',required=True,type=Path);p.add_argument('--role',required=True,choices=['train','validation','test','data','smoke']);p.add_argument('--split-id',required=True,help='Owner-defined pre-existing split designation; no automatic splitting');p.add_argument('--region',required=True,help='Saved region name, or explicit all');p.add_argument('--max-particles',required=True,type=int);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--max-events',type=int)
    p.add_argument('--analysis-config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--validation-features',type=Path,help='Separate validation apply output for generated training config')
    a=p.parse_args();analysis_config=read_yaml(a.analysis_config);class_contract=classification_contract(analysis_config)
    if a.max_particles<=0 or a.batch_size<=0 or (a.max_events is not None and a.max_events<=0):p.error('Bounds must be positive')
    if a.output.exists():p.error('Fresh output directory required')
    contract={'region':a.region,'max_particles':a.max_particles,'feature_fingerprint':FINGERPRINT}
    provenance={'input_files':[{'path':str(f.resolve()),'bytes':f.stat().st_size,'rows':pq.ParquetFile(f).metadata.num_rows,'schema_sha256':hashlib.sha256(pq.ParquetFile(f).schema_arrow.serialize().to_pybytes()).hexdigest()} for f in a.input],'role':a.role,'split_id':a.split_id,'region':a.region,'max_events':a.max_events,'max_particles':a.max_particles,'contract':contract}
    if a.mode=='fit':
        if a.role!='train':p.error('Fit requires explicit train split inputs; no validation/data/smoke refits')
        if a.normalization.exists():p.error('Normalization artifact already exists')
        n=fit_tables(batches(a.input,a.region,a.batch_size,a.max_events),a.max_particles,provenance,analysis_config)
        a.normalization.parent.mkdir(parents=True,exist_ok=True);a.normalization.write_text(json.dumps(n,indent=2)+'\n')
    else:
        n=json.loads(a.normalization.read_text())
        if n['classification_contract']!=class_contract:p.error('Classification contract mismatch')
        if n['provenance']['contract']!=contract:p.error('Selection/capacity normalization contract mismatch')
    a.output.mkdir(parents=True);count=0;shards=[];shapes={};reason_counts={} 
    for source in batches(a.input,a.region,a.batch_size,a.max_events):
        e=extract(source.to_pylist(),a.max_particles);x=apply_normalization(e,n);labels,reasons=targets(source.to_pylist(),analysis_config)
        for reason in reasons:reason_counts[reason]=reason_counts.get(reason,0)+1
        table,shapes=tensor_table(source,x,e,labels);table=table.append_column('classification_reason',pa.array(reasons));name=f'part-{len(shards):05d}.parquet';pq.write_table(table,a.output/name,compression='zstd');shards.append({'path':name,'rows':len(table)});count+=len(table)
    (a.output/'analysis_detector.yaml').write_text(yaml.safe_dump(grouped_config(),sort_keys=False));(a.output/'shape_metadata.json').write_text(json.dumps(shapes,indent=2)+'\n')
    manifest={**provenance,'fingerprint':FINGERPRINT,'sequential_features':list(RAW_NAMES),'raw_value_names':list(FEATURES),'normalization':str(a.normalization.resolve()),'normalization_sha256':hashlib.sha256(a.normalization.read_bytes()).hexdigest(),'rows':count,'shards':shards,'classification_contract':class_contract,'classification_reason_counts':reason_counts,'task_targets':'Classification only; unknown/unmapped/data are ineligible with target -1'}
    import sys,torch
    sys.path.insert(0,str(Path(__file__).parent/'EveNet-Align'))
    from detector_classification import backend_normalization,training_config
    from generate_event_info_yaml import parse_feature_config,parse_evenet_config,build_event_info_payload
    schema=parse_evenet_config(read_yaml(Path(__file__).parent/'config/evenet_schema.yaml'),grouped_config(),parse_feature_config(grouped_config()))
    event_info=build_event_info_payload(class_contract['classes'],parse_feature_config(grouped_config()),schema)
    event_info['GENERATIONS']={'Conditions':[],'GlobalTargets':[],'Events':[],'Neutrinos':{}}
    (a.output/'train_classification.yaml').write_text(yaml.safe_dump(training_config(a.output,class_contract,a.validation_features),sort_keys=False))
    torch.save(backend_normalization(n,class_contract),a.output/'normalization.pt')
    (a.output/'event_info.yaml').write_text(yaml.safe_dump(event_info,sort_keys=False))
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'rows':count,'features':len(RAW_NAMES),'output':str(a.output)}))

if __name__=='__main__':main()
