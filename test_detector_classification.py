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

if __name__=='__main__':unittest.main()
