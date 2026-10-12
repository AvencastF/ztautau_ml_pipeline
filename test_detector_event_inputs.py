import copy,json,os,unittest
from pathlib import Path
import numpy as np
from detector_event_inputs import *
from detector_classification import targets,contract,read_yaml,DEFAULT_CONFIG
from standard_detector_converter import Moments

class Tests(unittest.TestCase):
    def test_globals_missing_zero_and_unsupported(self):
        rows=[{'axis':[1.,0.,0.],'value.isolation':0.,'valid.isolation':True},{'axis':[None,None,None],'value.isolation':0.,'valid.isolation':False}]
        n=fit_conditions(rows);x,m=conditions(rows,n)
        self.assertTrue(m[0,0]);self.assertFalse(m[1,0]);self.assertEqual(x[0,7],1);self.assertEqual(x[1,7],0)
        n['count']=[0]*4;x,m=conditions(rows,n);self.assertFalse(m.any());self.assertFalse(x.any())
        self.assertFalse(reconstructed_conditions([{'axis':[2.,0.,0.]}])[1].any())
    def test_stable_moments_valid_zero(self):
        m=Moments(1);m.add(np.array([[1e12],[1e12+2]]),np.ones((2,1),bool));m.add(np.array([[1e12+4]]),np.ones((1,1),bool))
        self.assertAlmostEqual(m.result()['std'][0],np.sqrt(8/3));m=Moments(1);m.add(np.array([[0.],[99.]]),np.array([[True],[False]]));self.assertEqual(m.result()['count'],[1]);self.assertEqual(m.result()['mean'],[0.])
    def test_malformed_direction(self):
        for p in ([1,2], [0,0,0],[0,0,1],[None,1,2]):
            with self.assertRaises((ValueError,TypeError)):angles(p)
    def test_independent_angular_formula_and_masked_loss(self):
        from detector_inputs import datum
        def particle(i,p,q):return {'main_index':i,'main':{'fields':{**{'momentum.'+c:datum(v,True) for c,v in zip('xyz',p)},'charge':datum(q,True)}}}
        parts=[particle(0,[-1,.1,0],1),particle(1,[1,0,0],-1),particle(2,[-1,0,0],0),particle(3,[0,1,0],0)]
        prefix='sDST_LUJ_GenParticles/sDST_LUJ_GenParticles.'
        r={'particle_detector':{'particles':parts},'leading_indices':[1,0],'truth_decay':{'plus':{'tau_index':0},'minus':{'tau_index':1}},'generator_particles':{prefix+'PDG':[-15,15],prefix+'momentum.x':[-1,1],prefix+'momentum.y':[-.1,0],prefix+'momentum.z':[0,0]},'photon_associations':{'working_point':'existing_input_resolved_hpc_v1','valid':{'selected_photon_count':True},'values':{'selected_photon_count':2},'records':[{'main_index':2,'resolved_hpc_pass':True,'momentum_gev':[-1,0,0]},{'main_index':3,'resolved_hpc_pass':True,'momentum_gev':[0,1,0]}]}}
        c=read_yaml(DEFAULT_CONFIG);classes=contract(c)['classes'];v,m,_=angular_targets([r],[0],classes,c)
        expected=(math.atan2(-.1,-1)-math.atan2(.1,-2)+math.pi)%(2*math.pi)-math.pi
        np.testing.assert_allclose(v[0],[[0,expected],[0,0]],atol=1e-7);self.assertTrue(m.all())
        r['photon_associations']['records']=[];r['photon_associations']['values']['selected_photon_count']=0
        v,m,_=angular_targets([r],[0],classes,c);np.testing.assert_allclose(v[0,0,1],2*math.atan(.1),atol=1e-7)
        import sys,torch
        sys.path.insert(0,str(Path(__file__).parent/'EveNet-Align'))
        from evenet.network.loss.generation import loss
        pred=torch.zeros((2,2,2),requires_grad=True);target=torch.ones_like(pred);mask=torch.tensor([[[True],[True]],[[False],[False]]])
        a=loss(pred,target,mask,feature_dim=2);target[1]=999
        self.assertEqual(a.item(),loss(pred,target,mask,feature_dim=2).item());a.backward();self.assertTrue((pred.grad[1]==0).all())
        self.assertEqual(loss(pred,target,torch.zeros_like(mask),feature_dim=2).item(),0)
    def test_standard_preflight_contract(self):
        fixture=os.environ.get('STANDARD_DETECTOR_FIXTURE')
        if not fixture:self.skipTest('Set STANDARD_DETECTOR_FIXTURE for generated bounded shards')
        import tempfile,yaml,shutil
        from standard_detector_converter import validate_training_inputs
        c=read_yaml(Path(fixture));report=validate_training_inputs(fixture);self.assertTrue(report['disjoint'])
        from train_detector_classification import validate_cli_arguments
        with self.assertRaises(ValueError):validate_cli_arguments(['config.yaml','--load_all'])
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'config.yaml'
            copied=Path(d)/'train';shutil.copytree(Path(c['platform']['data_parquet_dir']),copied);(copied/'unexpected.parquet').touch();bad=copy.deepcopy(c);bad['platform']['data_parquet_dir']=str(copied);f.write_text(yaml.safe_dump(bad))
            with self.assertRaises(ValueError):validate_training_inputs(f)
            for mutate in (lambda c:c['platform'].pop('data_parquet_val_dir'),lambda c:c['platform'].update(data_parquet_val_dir=c['platform']['data_parquet_dir']),lambda c:c['options']['Training'].update(strategy={'name':'dgpo'})):
                bad=copy.deepcopy(c);mutate(bad);f.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):validate_training_inputs(f)
            e=read_yaml(Path(c['event_info']['default']));e['CLASSLABEL']['EVENT']['signal'][0].reverse();ep=Path(d)/'event.yaml';ep.write_text(yaml.safe_dump(e,sort_keys=False));bad=copy.deepcopy(c);bad['event_info']['default']=str(ep);f.write_text(yaml.safe_dump(bad))
            with self.assertRaises(ValueError):validate_training_inputs(f)
            e=read_yaml(Path(c['event_info']['default']));e['INPUTS']['SEQUENTIAL']['Source']['Part_energy']='normalize';ep.write_text(yaml.safe_dump(e,sort_keys=False))
            with self.assertRaises(ValueError):validate_training_inputs(f)
    def test_classification_only_without_angular_payload(self):
        from unittest.mock import patch
        from standard_detector_converter import arrays
        from detector_inputs import FEATURES,RAW_NAMES
        row={'identity':['Z_ee',1994,1,1],'process':'Z_ee','weight':-2.,'axis':[1.,0.,0.],'particle_detector':json.dumps({'version':1,'status':'available','particles':[],'collections':{}})}
        n={'fit_role':'train','fingerprint':__import__('detector_inputs').FINGERPRINT,'feature_names':list(RAW_NAMES),'raw_value_names':list(FEATURES),'mean':[0.]*len(FEATURES),'std':[1.]*len(FEATURES),'count':[1]*len(FEATURES),'provenance':{'contract':{'tasks':'classification'}},'global':{'mean':[0.]*4,'std':[1.]*4,'count':[1]*4}}
        with patch('standard_detector_converter.angular_targets',side_effect=AssertionError('must not read angular truth')):
            data,eligible,_,why=arrays([row],n,2,read_yaml(DEFAULT_CONFIG))
        self.assertTrue(eligible[0]);self.assertNotIn('x_invisible',data);self.assertEqual(data['event_weight'][0],-2.);self.assertEqual(why,['task_disabled'])
    def test_deterministic_identity_split(self):
        from split_detector_parquet import validation_identity
        identity=['Z_tautau',1994,1,2];self.assertEqual(validation_identity(identity),validation_identity(json.dumps(identity)))
        train=set();validation=set()
        for i in range(100):
            digest,isval=validation_identity(['Z_tautau',1994,1,i]);(validation if isval else train).add(digest)
        self.assertEqual(len(train|validation),100);self.assertFalse(train&validation);self.assertTrue(10<len(validation)<35)
        with self.assertRaises(ValueError):validation_identity(None)
    def test_actual_alignment_and_target_masks(self):
        path=os.environ.get('DETECTOR_SOURCE_FIXTURE')
        if not path:self.skipTest('Set DETECTOR_SOURCE_FIXTURE for genuine bounded source')
        import pyarrow.parquet as pq
        rows=pq.read_table(path).to_pylist();c=read_yaml(DEFAULT_CONFIG);labels,_=targets(rows,c);classes=contract(c)['classes'];v,m,_=angular_targets(rows,labels['classification'],classes,c)
        self.assertEqual(m.sum(),6);self.assertTrue(np.isfinite(v).all());self.assertTrue((v[~m]==0).all())
        r=copy.deepcopy(rows[3]);p=decode(r['particle_detector']);p['particles'].reverse();r['particle_detector']=p
        vv,mm,_=angular_targets([r],labels['classification'][3:4],classes,c);np.testing.assert_equal(vv,v[3:4]);np.testing.assert_equal(mm,m[3:4])
        r['leading_indices']=list(reversed(decode(r['leading_indices'])));vv,mm,_=angular_targets([r],labels['classification'][3:4],classes,c);np.testing.assert_equal(vv,v[3:4])
        p['particles'].append(p['particles'][0]);_,mm,_=angular_targets([r],labels['classification'][3:4],classes,c);self.assertFalse(mm.any())
        r=copy.deepcopy(rows[3]);ph=decode(r['photon_associations']);ph['valid']['selected_photon_count']=False;r['photon_associations']=ph
        vv,mm,_=angular_targets([r],labels['classification'][3:4],classes,c);self.assertFalse(mm.any());self.assertFalse(vv.any())
        vv,mm,_=angular_targets([{'process':'Z_ee'},{'process':'data'}],np.array([17,-1]),classes,c);self.assertFalse(mm.any());self.assertFalse(vv.any())

if __name__=='__main__':unittest.main()
