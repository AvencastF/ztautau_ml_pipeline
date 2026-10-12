#!/usr/bin/env python3
"""Streaming owner-approved inclusive80/20 identity split, before fitting."""
import argparse,hashlib,json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from build_evenet_input_from_parquet import batches
from detector_classification import ALIASES,targets,read_yaml,DEFAULT_CONFIG
from detector_event_inputs import decode

def validation_identity(identity,seed=42):
    value=decode(identity)
    if not isinstance(value,list) or not value or any(v is None for v in value):raise ValueError('Unavailable saved event identity')
    encoded=json.dumps({'seed':seed,'identity':value},sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return hashlib.sha256(encoded).digest(),int.from_bytes(hashlib.sha256(encoded).digest()[:8],'big')%5==0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--region',default='inclusive_1p1p');p.add_argument('--seed',type=int,default=42);p.add_argument('--capacity-limit',type=int,default=128);a=p.parse_args()
    if a.output.exists():p.error('Fresh split output required')
    if a.batch_size<=0 or a.capacity_limit<=0:p.error('Positive bounds required')
    if a.region!='inclusive_1p1p' or a.seed!=42:p.error('Prepared owner policy is inclusive_1p1p, seed42')
    a.output.mkdir(parents=True);[(a.output/r).mkdir() for r in ('train','validation')]
    writers={};counts={'train':0,'validation':0,'ineligible':0};seen=set();maximum=0;process_counts={};config=read_yaml(DEFAULT_CONFIG)
    try:
        for source in a.input:
            process=source.stem
            if process not in ALIASES:raise ValueError('Only configured mapped MC inputs supported: '+process)
            for table in batches([source],a.region,a.batch_size,None):
                rows=table.to_pylist();labels,_=targets(rows,config);keep={'train':[],'validation':[]}
                for i,row in enumerate(rows):
                    if row['process']!=process or row.get('is_data') is True or row.get('sample_is_data') is True:raise ValueError('Input process/data provenance mismatch')
                    digest,val=validation_identity(row['identity'],a.seed)
                    if digest in seen:raise ValueError('Duplicate input identity')
                    seen.add(digest)
                    if not labels['classification_eligible'][i]:counts['ineligible']+=1;continue
                    payload=decode(row['particle_detector']);n=len(payload['particles']);maximum=max(maximum,n)
                    role='validation' if val else 'train';keep[role].append(i);counts[role]+=1;process_counts.setdefault(process,{'train':0,'validation':0})[role]+=1
                for role,indices in keep.items():
                    if not indices:continue
                    key=(process,role)
                    if key not in writers:writers[key]=pq.ParquetWriter(a.output/role/(process+'.parquet'),table.schema,compression='zstd')
                    writers[key].write_table(table.take(pa.array(indices,pa.int64())))
    finally:
        for w in writers.values():w.close()
    report={'version':1,'region':a.region,'seed':a.seed,'split':'sha256(canonical JSON {seed,identity}) first8bytes bigendian modulo5 ==0 validation; remaining train','counts':counts,'process_counts':process_counts,'max_main_particles':maximum,'capacity_limit':a.capacity_limit,'within_capacity':maximum<=a.capacity_limit,'input_files':[{'path':str(f.resolve()),'bytes':f.stat().st_size,'rows':pq.ParquetFile(f).metadata.num_rows} for f in a.input]}
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    if not counts['train'] or not counts['validation']:raise ValueError('Both splits must be nonempty')
    if maximum>a.capacity_limit:raise ValueError(f'Observed MAIN maximum{maximum} exceeds capacity limit{a.capacity_limit}; split preserved, review capacity before conversion')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
