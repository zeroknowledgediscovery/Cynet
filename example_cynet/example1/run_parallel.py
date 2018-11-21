import cynet.cynet as cn
import pandas as pd
import numpy as np
import subprocess
import os
import sys
from tqdm import tqdm, tqdm_pandas
import uuid
import glob
from joblib import Parallel , delayed
import yaml
import seaborn as sns
import pylab as plt
import viscynet.viscynet as vis
plt.style.use('dark_background')
import warnings

DEBUG_=False

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']
print(VARNAME)

cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH, FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=4, gamma=True)
cn.flexroc_only_parallel('models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=4)

VARNAMES=['Personnel','Infrastructure','Casualties']

cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)

model_path=settings_dict['FILEPATH']+'*model.json'
MAX_DIST=settings_dict['MAX_DIST']
MIN_DIST=settings_dict['MIN_DIST']
MAX_GAMMA=settings_dict['MAX_GAMMA']
MIN_GAMMA=settings_dict['MIN_GAMMA']
COLORMAP=settings_dict['COLORMAP']

vis.render_network_parallel(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,COLORMAP,horizon,model_nums[0], newmodel_name='newmodel.json')
