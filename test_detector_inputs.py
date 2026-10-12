import copy,functools,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from detector_inputs import *
from build_evenet_input_from_parquet import tensor_table,fit_tables,batches
from generate_event_info_yaml import parse_feature_config

ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'EveNet-Align'))
from evenet.dataset.preprocess import unflatten_dict,process_event_batch

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
Grouped=module('detector_test_grouped',ROOT/'EveNet-Align/evenet/network/body/grouped_sequential_embedding.py').GroupedSequentialEmbedding
Normalizer=module('detector_test_normalizer',ROOT/'EveNet-Align/evenet/network/body/normalizer.py').Normalizer

def fixture(value=0,semantic=True):
    main={'index':0,'fields':{k:datum(v) for k,v in {'energy':2,'momentum.x':0,'momentum.y':0,'momentum.z':0,'charge':-1,'mass':0}.items()},'associations':{'clusters':{'status':'no_association','records':[]},'tracks':{'status':'no_association','records':[]}}}
    p={'main_index':0,'main':main,'detector_rows':{'sDST_ELID_ElectronID':{'status':'linked','records':[{'index':0,'named_parameters':{'tag':datum(value,semantic)}}]}},'parallel':{}}
    payload={'version':1,'status':'available','particles':[p],'collections':{c:{'present':True,'count':0} for c in CALO}}
    return {'identity':'fixture','process':'Z_tautau','weight':-2.,'region_mask':0,'particle_detector':json.dumps(payload)}

class Tests(unittest.TestCase):
    def test_zero_missing_semantics_padding(self):
        zero=fixture();missing=fixture(None);unknown=fixture(3,None);failed=fixture(-1,False)
        e=extract([zero,missing,unknown,failed],2);i=FEATURES.index('sDST_ELID_ElectronID.tag')
        self.assertEqual(e['raw_values'][:,0,i].tolist(),[0,0,3,-1]);self.assertEqual(e['usable'][:,0,i].tolist(),[True,False,True,False])
        self.assertEqual(e['feature_masks'][2,0,i].tolist(),[True,False,False]);self.assertTrue(e['x_mask'][:,0].all());self.assertFalse(e['x_mask'][:,1].any())
        n=fit_normalization(e,role='train',provenance={});self.assertEqual(n['count'][i],2);self.assertEqual(n['mean'][i],1.5)
        x=apply_normalization(e,n);self.assertTrue(np.isfinite(x).all());self.assertEqual(x[0,0,RAW_NAMES.index('Part_charge')],-1)
        eta=RAW_NAMES.index('Part_eta');self.assertTrue((x[:,:,eta]==0).all())
        self.assertEqual(x[0,0,RAW_NAMES.index('Part_eta__normalization_supported')],0)
        self.assertTrue((x[:,1]==0).all())
        with self.assertRaises(ValueError):fit_normalization(e,role='validation',provenance={})
        bad=copy.deepcopy(n);bad['feature_names'].reverse()
        with self.assertRaises(ValueError):apply_normalization(e,bad)
        bad=copy.deepcopy(n);bad['count'][i]=0
        self.assertEqual(apply_normalization(e,bad)[2,0,RAW_NAMES.index('sDST_ELID_ElectronID.tag')],0)
    def test_associations(self):
        row=fixture();p=json.loads(row['particle_detector']);particle=p['particles'][0]
        vals=particle_features(particle,p['collections']);self.assertEqual(vals[0][FEATURES.index(CALO[0]+'.count')],0);self.assertTrue(vals[1][FEATURES.index(CALO[0]+'.count'),0])
        p['collections'][CALO[0]]['present']=False;self.assertFalse(particle_features(particle,p['collections'])[1][FEATURES.index(CALO[0]+'.count'),0])
        particle['main']['associations']['clusters']={'status':'malformed','records':[]};self.assertFalse(particle_features(particle,p['collections'])[1][FEATURES.index(CALO[1]+'.count'),0])
        particle['main']['associations']['tracks']={'status':'linked','records':[{'status':'linked','payload':{'fields':{'chi2':datum(99)}}}]*2}
        raw,m,_,_=particle_features(particle,p['collections']);self.assertFalse(m[FEATURES.index('track.chi2'),0]);self.assertEqual(raw[FEATURES.index('track.count')],2)
        particle['detector_rows']['sDST_ELID_ElectronID']['records']*=2
        self.assertFalse(particle_features(particle,p['collections'])[1][FEATURES.index('sDST_ELID_ElectronID.tag'),0])
    def test_derived_validity_and_data_fit(self):
        row=fixture();payload=json.loads(row['particle_detector']);p=payload['particles'][0]
        p['main']['fields']['momentum.x']=datum(3,False)
        raw,m,usable,_=particle_features(p,payload['collections']);self.assertFalse(usable[FEATURES.index('Part_pt')]);self.assertTrue(m[FEATURES.index('Part_pt'),1])
        p['main']['fields']['momentum.x']=datum(3,True)
        self.assertFalse(particle_features(p,payload['collections'])[1][FEATURES.index('Part_pt'),1])
        c=CALO[0];r={'fields':{'energy':datum(5,False)}}
        p['main']['associations']['clusters']={'status':'linked','records':[{'collection':c,'index':0,'status':'linked','payload':r}]}
        self.assertFalse(particle_features(p,payload['collections'])[2][FEATURES.index(c+'.energy_sum')])
        r['fields']['energy']=datum(5,None)
        self.assertTrue(particle_features(p,payload['collections'])[2][FEATURES.index(c+'.energy_sum')])
        row['process']='data'
        with self.assertRaises(ValueError):fit_tables([pa.Table.from_pylist([row])],2,{})

    def test_genuine_source_schema(self):
        path=Path(os.environ.get('DETECTOR_INPUT_FIXTURE',str(ROOT.parent/'artifacts/particle-detector-parquet-20261012/Z_tautau/skim/Z_tautau.parquet')))
        if not path.is_file():self.skipTest('Genuine513-column fixture absent; set DETECTOR_INPUT_FIXTURE explicitly')
        source=pq.read_table(path);self.assertEqual(len(source.column_names),513)
        capacity=max(len(json.loads(v)['particles']) for v in source['particle_detector'].to_pylist())
        e=extract(source.to_pylist(),capacity);n=fit_normalization(e,role='train',provenance={'scope':'engineering fixture'})
        table,shapes=tensor_table(source,apply_normalization(e,n),e)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'converted.parquet';pq.write_table(table,out);loaded=pq.read_table(out)
        self.assertTrue(loaded.select(source.column_names).equals(source,check_metadata=True))
        self.assertEqual(list(loaded.schema)[:513],list(source.schema))

    def test_alignment_empty_overflow_truth(self):
        row=fixture();p=json.loads(row['particle_detector']);p['particles'][0]['main_index']=3;row['particle_detector']=json.dumps(p)
        with self.assertRaises(ValueError):extract([row],2)
        row=fixture();p=json.loads(row['particle_detector']);p['particles']=[];row['particle_detector']=json.dumps(p);e=extract([row],2);self.assertFalse(e['x_mask'].any())
        with self.assertRaises(ValueError):extract([fixture()],0)
        a=extract([fixture()],2);r=fixture();r['truth_decay']='changed truth';r['generator_particles']='[{"PDG":13}]';b=extract([r],2)
        np.testing.assert_array_equal(a['raw_values'],b['raw_values'])
    def test_stable_fit_and_real_loader_projection(self):
        rows=[fixture(1e12+v) for v in (0,1,2,3)];sources=[pa.Table.from_pylist(rows[:2]),pa.Table.from_pylist(rows[2:])]
        n=fit_tables(sources,2,{});i=FEATURES.index('sDST_ELID_ElectronID.tag');self.assertAlmostEqual(n['std'][i],np.std([0,1,2,3]))
        e=extract(rows,2);x=apply_normalization(e,n);source=pa.Table.from_pylist(rows);table,shapes=tensor_table(source,x,e)
        self.assertTrue(table.select(source.column_names).equals(source,check_metadata=True))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'tiny.parquet';pq.write_table(table,path);loaded=pq.read_table(path);t=process_event_batch({n:loaded[n].to_numpy() for n in loaded.column_names if n not in source.column_names},shapes,functools.partial(load_tensor_batch,unflatten=unflatten_dict))
        np.testing.assert_array_equal(t['x'],x);self.assertEqual(t['conditions'].shape,(4,0));self.assertEqual(t['conditions_mask'].shape,(4,1));self.assertFalse(t['conditions_mask'].any());self.assertEqual(t['event_weight'].tolist(),[-2]*4)
        parsed=parse_feature_config(grouped_config());self.assertEqual(parsed.raw_sequential_fields,tuple(RAW_NAMES));self.assertEqual(len(parsed.projected_sequential_feature_names),7)
        norm=Normalizer(torch.zeros(len(RAW_NAMES)),torch.ones(len(RAW_NAMES)),torch.zeros(len(RAW_NAMES),dtype=torch.bool));xt=torch.tensor(t['x']);mask=torch.tensor(t['x_mask'])[...,None]
        normalized=norm(xt,mask);self.assertTrue(torch.equal(xt,normalized));model=Grouped(list(RAW_NAMES),parsed.grouped_sequential_config["SEQUENTIAL"]);projected=model(normalized,mask)
        self.assertEqual(tuple(projected.shape),(4,2,7));self.assertTrue(torch.isfinite(projected).all());self.assertTrue((projected[:,1]==0).all());self.assertTrue(torch.equal(projected[:,0,6],torch.full((4,),-1.)))

if __name__=='__main__':unittest.main()
