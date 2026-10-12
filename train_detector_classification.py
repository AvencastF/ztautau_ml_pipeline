#!/usr/bin/env python3
"""Preflight standard detector shards, then run unchanged supervised Align."""
import os,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parent
def validate_cli_arguments(args):
    if '--load_all' in args:raise ValueError('--load_all bypasses separate validation; unsupported for frozen detector splits')

if __name__=='__main__':
    validate_cli_arguments(sys.argv[1:])
    if len(sys.argv)<2:raise SystemExit('Usage: train_detector_classification.py CONFIG.yaml [ordinary Align arguments]')
    from standard_detector_converter import validate_training_inputs
    print(validate_training_inputs(Path(sys.argv[1])))
    os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
    sys.path.insert(0,str(root/'EveNet-Align'))
    runpy.run_path(str(root/'EveNet-Align/scripts/train.py'),run_name='__main__')
