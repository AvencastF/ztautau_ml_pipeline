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
    # Canonical CLI uses ordinary numerical shards and unchanged backend.
    from standard_detector_converter import main as standard_main
    standard_main()

if __name__=='__main__':main()
