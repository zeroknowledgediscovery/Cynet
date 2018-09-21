import cynet.cynet as cn
from cynet.cynet import simulateModel
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
plt.style.use('dark_background')
import warnings

DEBUG_=False

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

DATA_PATH='/home/ishanu/Dropbox/ZED/Research/cytest/data/split/@TR'
VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']


cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH)

VARNAMES=['Personnel','Infrastructure','Casualties']

cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)
