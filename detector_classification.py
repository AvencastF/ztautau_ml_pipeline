"""Audited charge-ordered classification targets, separate from detector inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
from common import build_classification_lookup,read_yaml

ALIASES={'Z_tautau':'Ztautau','Z_ee':'Zll','Z_mumu':'Zll','Z_qq':'Zqq'}
MODES={'pi':1,'rho':2,'e':3,'mu':4,'other_1prong':5,'other_3prong':5,'other_prong':5}
DEFAULT_CONFIG=Path(__file__).parent/'config/analysis.yaml'

def contract(config):
    lookup=build_classification_lookup(config)
    spec={'classes':list(lookup.class_labels),'aliases':ALIASES,'modes':MODES,'subcategories':config['Subcategories'],'ordering':'10*plus+minus'}
    return {**spec,'fingerprint':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()}

def targets(rows,config):
    lookup=build_classification_lookup(config);indices=[];reasons=[]
    for row in rows:
        process=row.get('process');sample=ALIASES.get(process)
        label=None;reason='unmapped_process'
        if process=='data' or row.get('is_data') is True or row.get('sample_is_data') is True:reason='data'
        elif sample=='Ztautau':
            try:
                truth=json.loads(row['truth_decay']) if isinstance(row.get('truth_decay'),str) else row.get('truth_decay')
                plus=MODES.get(truth['plus']['mode']);minus=MODES.get(truth['minus']['mode'])
                if plus is not None and minus is not None:
                    label=lookup.sample_event_category_to_label[sample].get(10*plus+minus);reason='mapped' if label else 'unmapped_category'
                else:reason='unknown_truth'
            except (KeyError,TypeError,ValueError):reason='unavailable_truth'
        elif sample in lookup.sample_default_label:label=lookup.sample_default_label[sample];reason='mapped'
        indices.append(lookup.label_to_index[label] if label is not None else -1);reasons.append(reason)
    indices=np.asarray(indices,np.int64)
    return {'classification':indices,'classification_eligible':indices>=0},reasons

def backend_normalization(normalization,class_contract):
    import torch
    from evenet.dataset.postprocess import compute_classification_balance
    from detector_inputs import RAW_NAMES
    counts=np.asarray(normalization['classification_counts'])
    if not counts.any():raise ValueError('No eligible training classifications')
    return {'input_mean':{'Source':torch.zeros(len(RAW_NAMES)),'Conditions':torch.empty(0)},'input_std':{'Source':torch.ones(len(RAW_NAMES)),'Conditions':torch.empty(0)},'class_balance':torch.tensor(compute_classification_balance(counts)),'class_counts':torch.tensor(counts),'classification_contract':class_contract,'detector_fingerprint':normalization['fingerprint']}


def training_config(directory,class_contract=None,validation=None):
    root=Path(__file__).parent;align=root/'EveNet-Align';directory=Path(directory).resolve()
    result={'platform':{'data_parquet_dir':str(directory),'number_of_workers':1,'resources_per_worker':{'CPU':1,'GPU':0},'batch_size':16,'prefetch_batches':1,'use_gpu':False},
            'logger':{'wandb':{'enabled':False},'local':{'save_dir':str(directory/'logs'),'name':'detector-classification','version':'run'}},
            'options':{'default':str(align/'share/options/options.yaml'),'Training':{'model_checkpoint_save_path':None,'model_checkpoint_load_path':None,'pretrain_model_load_path':None,'Components':{'GroupedSequentialEmbedding':read_yaml(root/'config/train_pretrain.yaml')['options']['Training']['Components']['GroupedSequentialEmbedding'],'Classification':{'include':True,'include_cross_term':False},'Regression':{'include':False},'Assignment':{'include':False},'Segmentation':{'include':False},'GlobalGeneration':{'include':False},'ReconGeneration':{'include':False},'TruthGeneration':{'include':False}},'ProgressiveTraining':{'stages':[{'name':'classification','epoch_ratio':1.0,'transition_ratio':0.0,'loss_weights':{'classification':[1.0,1.0]}}]}},'Dataset':{'normalization_file':str(directory/'normalization.pt'),'detector_classification_contract':class_contract or contract(read_yaml(DEFAULT_CONFIG))}},
            'network':{'default':str(align/'share/network/network-20M.yaml')},'event_info':{'default':str(directory/'event_info.yaml')},'resonance':{'default':str(root/'config/resonance.yaml')}}

    if validation is not None:result['platform']['data_parquet_val_dir']=str(Path(validation).resolve())
    return result

def validate_manifest(directory,class_labels,expected_contract=None):
    from detector_inputs import FINGERPRINT,RAW_NAMES
    import torch
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['fingerprint']!=FINGERPRINT or manifest['sequential_features']!=list(RAW_NAMES):raise ValueError('Detector feature contract mismatch')
    if expected_contract is not None and manifest['classification_contract']!=expected_contract:raise ValueError('Classification mapping contract mismatch')
    if manifest['classification_contract']['aliases']!=ALIASES or manifest['classification_contract']['modes']!=MODES or manifest['classification_contract']['ordering']!='10*plus+minus':raise ValueError('Unsupported classification mapping')
    if manifest['classification_contract']['classes']!=list(class_labels):raise ValueError('Classification class order mismatch')
    normal_path=Path(manifest['normalization']);normal_bytes=normal_path.read_bytes();normal=json.loads(normal_bytes)
    if hashlib.sha256(normal_bytes).hexdigest()!=manifest['normalization_sha256'] or normal['classification_contract']!=manifest['classification_contract']:raise ValueError('Classification normalization contract mismatch')
    if normal['provenance']['contract']!=manifest['contract']:raise ValueError('Selection/capacity contract mismatch')
    shapes=json.loads((directory/'shape_metadata.json').read_text())
    expected={'x':[manifest['max_particles'],len(RAW_NAMES)],'x_mask':[manifest['max_particles']],'conditions':[0],'classification':[],'classification_eligible':[]}
    if any(shapes.get(k)!=v for k,v in expected.items()):raise ValueError('Tensor shape contract mismatch')
    backend=torch.load(directory/'normalization.pt',map_location='cpu',weights_only=True)
    if backend['classification_contract']!=manifest['classification_contract'] or backend['detector_fingerprint']!=FINGERPRINT:raise ValueError('Backend normalization contract mismatch')
    expected_backend=backend_normalization(normal,manifest['classification_contract'])
    if not torch.equal(backend['class_counts'],expected_backend['class_counts']) or not torch.equal(backend['class_balance'],expected_backend['class_balance']) or any(not torch.equal(backend[k][n],v) for k in ('input_mean','input_std') for n,v in expected_backend[k].items()):raise ValueError('Backend normalization values mismatch')
    return manifest

def eligible_row(row,class_count):
    value=row['classification'];eligible=row['classification_eligible']
    if not isinstance(eligible,(bool,np.bool_)) or isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not -1<=value<class_count or bool(eligible)!=(value>=0):raise ValueError('Malformed classification eligibility/index')
    return bool(eligible)
