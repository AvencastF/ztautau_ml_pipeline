import copy,json,unittest,sys,tempfile,shutil
from unittest.mock import patch
import numpy as np
from detector_classification import *

class Tests(unittest.TestCase):
    def test_charge_order_and_unknown(self):
        config=read_yaml(DEFAULT_CONFIG);classes=contract(config)['classes'];rows=[]
        for plus,minus in [('pi','rho'),('rho','pi'),('other_3prong','e'),('unknown','pi')]:rows.append({'process':'Z_tautau','truth_decay':json.dumps({'plus':{'mode':plus},'minus':{'mode':minus}})})
        labels,reasons=targets(rows,config)
        self.assertEqual(labels['classification'].tolist(),[classes.index('Ztautau_pirho'),classes.index('Ztautau_rhopi'),classes.index('Ztautau_others'),-1]);self.assertEqual(reasons[-1],'unknown_truth')
        self.assertEqual(len(classes),19)
    def test_data_background_and_malformed(self):
        config=read_yaml(DEFAULT_CONFIG);rows=[{'process':p} for p in ('data','Z_ee','Z_mumu','Z_qq','Bhabha','yy_ll','Z_tautau')]
        labels,reasons=targets(rows,config);self.assertEqual(labels['classification'].tolist(),[-1,17,17,18,-1,-1,-1]);self.assertEqual(labels['classification_eligible'].dtype,np.dtype(bool))
        for i,row in enumerate(rows):self.assertEqual(eligible_row({k:v[i] for k,v in labels.items()},19),labels['classification_eligible'][i])
        for value,flag in [(19,True),(-2,False),(-1,True),(1,False),(1.5,True),(True,True),(1,1)]:
            with self.assertRaises(ValueError):eligible_row({'classification':value,'classification_eligible':flag},19)
        # No ineligible row can enter the registered supervised filter.
        self.assertFalse(any(eligible_row({'classification':-1,'classification_eligible':False},19) for _ in range(2)))
    def test_registered_contract_rejections(self):
        root=Path(__file__).parent;fixture=root.parent/'artifacts/ml-detector-converter-20261012/classification-train-out'
        if not fixture.is_dir():self.skipTest('External bounded classification fixture absent')
        sys.path.insert(0,str(root/'EveNet-Align'))
        from evenet.control.global_config import global_config
        from scripts.shared import make_process_fn
        classes=contract(read_yaml(DEFAULT_CONFIG))['classes']
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)
            for name in ('manifest.json','shape_metadata.json','normalization.pt'):shutil.copyfile(fixture/name,target/name)
            validate_manifest(target,classes)
            with self.assertRaises(ValueError):validate_manifest(target,list(reversed(classes)))
            shapes=json.loads((target/'shape_metadata.json').read_text());bad=copy.deepcopy(shapes);bad['x'][0]+=1;(target/'shape_metadata.json').write_text(json.dumps(bad))
            with self.assertRaises(ValueError):validate_manifest(target,classes)
            (target/'shape_metadata.json').write_text(json.dumps(shapes))
            manifest=json.loads((target/'manifest.json').read_text());manifest['max_particles']+=1;(target/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):validate_manifest(target,classes)
            shutil.copyfile(fixture/'manifest.json',target/'manifest.json')
            global_config.load_yaml(fixture/'train_classification.yaml')
            # Configured engine artifact is the original directory, not this validated copy.
            with self.assertRaises(ValueError):make_process_fn(target)
            global_config.options.Dataset.normalization_file=str(target/'normalization.pt');make_process_fn(target)
            features=global_config.event_info.input_features['Source'];original=features
            for key in ('normalize','log_scale','uniform'):
                global_config.event_info.input_features['Source']=(features[0]._replace(**{key:True}),*features[1:])
                with self.assertRaises(ValueError):make_process_fn(target)
            global_config.event_info.input_features['Source']=original
            global_config.event_info.input_features['Conditions']=(features[0],)
            with self.assertRaises(ValueError):make_process_fn(target)

    def test_loader_supervision_and_prediction_guards(self):
        root=Path(__file__).parent;art=root.parent/'artifacts/ml-detector-converter-20261012';train=art/'classification-train-out';data=art/'classification-data-out'
        if not train.is_dir():self.skipTest('External bounded classification fixture absent')
        sys.path.insert(0,str(root/'EveNet-Align'))
        from evenet.control.global_config import global_config
        from scripts.shared import make_process_fn,register_dataset,prepare_datasets
        global_config.load_yaml(train/'train_classification.yaml');process=make_process_fn(train)
        with self.assertRaises(ValueError):prepare_datasets(train,process,global_config.platform,base_val_dir=None)
        with self.assertRaises(ValueError):prepare_datasets(train,process,global_config.platform,base_val_dir=data)
        class TinyDataset:
            # Tests registered control flow without starting a Ray cluster.
            def __init__(self,rows):self.rows=rows
            def filter(self,fn):return TinyDataset([r for r in self.rows if fn(r)])
            def count(self):return len(self.rows)
            def map_batches(self,*args,**kwargs):return self
        ignored=TinyDataset([{'classification':-1,'classification_eligible':False}]*2)
        with patch('scripts.shared.ray.data.read_parquet',return_value=ignored):
            with self.assertRaises(ValueError):register_dataset(['fixture'],process,global_config.platform)
            retained,count=register_dataset(['fixture'],process,global_config.platform,supervised=False);self.assertEqual(count,2)
        normal=TinyDataset([{'classification':16,'classification_eligible':True},{'classification':-1,'classification_eligible':False}])
        with patch('scripts.shared.ray.data.read_parquet',return_value=normal):
            retained,count=register_dataset(['fixture'],process,global_config.platform);self.assertEqual(count,1)
        # Existing unrelated datasets do not activate detector eligibility filtering.
        def legacy(*args,**kwargs):pass
        with patch('scripts.shared.ray.data.read_parquet',return_value=ignored):self.assertEqual(register_dataset(['fixture'],legacy,global_config.platform)[1],2)

    def test_disabled_segmentation_import_and_mask_dtype(self):
        root=Path(__file__).parent;fixture=root.parent/'artifacts/ml-detector-converter-20261012/classification-train-out'
        if not fixture.is_dir():self.skipTest('External bounded classification fixture absent')
        sys.path.insert(0,str(root/'EveNet-Align'))
        import torch
        torch.set_num_threads(2)
        from scripts.engine import EveNetEngine
        from evenet.network.evenet_model import EveNetModel
        from evenet.dataset.types import FeatureInfo
        from evenet.control.global_config import global_config
        self.assertNotIn('evenet.network.metrics.segmentation',sys.modules)
        global_config.load_yaml(fixture/'train_classification.yaml');normal=torch.load(fixture/'normalization.pt',weights_only=True)
        for features in [(),(FeatureInfo('fixture_global',False,False,False),)]:
            global_config.event_info.input_features['Conditions']=features
            normal['input_mean']['Conditions']=torch.zeros(len(features));normal['input_std']['Conditions']=torch.ones(len(features))
            model=EveNetModel(global_config,device=torch.device('cpu'),classification=True,normalization_dict=normal)
            self.assertEqual(model.global_normalizer.norm_mask.dtype,torch.bool);self.assertEqual(model.global_normalizer.norm_mask.numel(),len(features))

if __name__=='__main__':unittest.main()
