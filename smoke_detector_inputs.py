#!/usr/bin/env python3
"""Bounded feature-reader/7-slot encoder smoke; does not construct task targets."""
import argparse,functools,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from detector_inputs import load_tensor_batch,RAW_NAMES,FINGERPRINT,grouped_config
from generate_event_info_yaml import parse_feature_config
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'EveNet-Align'))
from evenet.dataset.preprocess import process_event_batch,unflatten_dict

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--rows',type=int,default=2);a=parser.parse_args()
    if a.rows<=0:parser.error('Positive row bound required')
    manifest=json.loads((a.directory/'manifest.json').read_text());shapes=json.loads((a.directory/'shape_metadata.json').read_text())
    if manifest['fingerprint']!=FINGERPRINT or manifest['sequential_features']!=list(RAW_NAMES):raise ValueError('Feature contract mismatch')
    table=next(pq.ParquetFile(a.directory/manifest['shards'][0]['path']).iter_batches(batch_size=a.rows))
    columns={name:table.column(i).to_numpy(zero_copy_only=False) for i,name in enumerate(table.schema.names) if name.split(':')[0] in shapes}
    tensors=process_event_batch(columns,shapes,functools.partial(load_tensor_batch,unflatten=unflatten_dict))
    spec=importlib.util.spec_from_file_location('detector_grouped',ROOT/'EveNet-Align/evenet/network/body/grouped_sequential_embedding.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    config=parse_feature_config(grouped_config());encoder=module.GroupedSequentialEmbedding(list(RAW_NAMES),config.grouped_sequential_config['SEQUENTIAL'])
    projected=encoder(torch.tensor(tensors['x']),torch.tensor(tensors['x_mask'])[...,None])
    assert projected.shape[-1]==7 and torch.isfinite(projected).all()
    assert tensors['conditions'].shape==(len(table),0) and tensors['conditions_mask'].shape==(len(table),1)
    print(json.dumps({'rows':len(table),'input_shape':list(tensors['x'].shape),'projected_shape':list(projected.shape),'conditions_shape':list(tensors['conditions'].shape),'finite':True,'scope':'Feature reader and grouped encoder only; task targets not constructed'}))
if __name__=='__main__':main()
