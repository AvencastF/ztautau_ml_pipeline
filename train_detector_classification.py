#!/usr/bin/env python3
"""Run the normal Align training entrance with the detector adapter importable."""
import os,runpy,sys
from pathlib import Path
root=Path(__file__).resolve().parent
os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
sys.path.insert(0,str(root/'EveNet-Align'))
runpy.run_path(str(root/'EveNet-Align/scripts/train.py'),run_name='__main__')
