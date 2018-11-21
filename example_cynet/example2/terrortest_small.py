

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
LOGFILE= 'terror.csv'
STOREFILE='terror.p'

EPS=5 #ONLY AND EXAMPLE. USE A LARGER EPS.
DATES = []
for year in range(1995, 2014):
    period_start = str(year) + '-01-01'
    period_end = str(year + 4) + '-01-01'
    period_end_extended = str(year + 5) + '-01-01'
    DATES.append((period_start, period_end, period_end_extended))

grid={'Latitude':np.around(np.linspace(-4,49,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-16,84,EPS),decimals=5),
      'Eps':EPS}


tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])



S0=cn.spatioTemporal(log_file=LOGFILE,
                     log_store=STOREFILE,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     value_limits=[0,10000],
                     #types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],
                     #value_limits=None,
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2017',
                     freq='D',
                     EVENT='nkill',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)

S0.fit(csvPREF='NKILL')

tiles = S0.getGrid()


S00=cn.spatioTemporal(log_store=STOREFILE,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     types=[['Bombing/Explosion','Facility/Infrastructure Attack']],
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2017',
                     freq='D',
                     EVENT='attacktype1_txt',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)

S00.fit(csvPREF='')

S1=cn.spatioTemporal(log_store=STOREFILE,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     types=[['Armed Assault', 'Hostage Taking (Barricade Incident)','Hijacking','Assassination','Hostage Taking (Kidnapping) ']],
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2017',
                     freq='D',
                     EVENT='attacktype1_txt',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)
S1.fit(csvPREF='')


CSVfile=['NKILL.csv','Bombing_Explosion-Facility_Infrastructure_Attack.csv', 'Armed_Assault-Hostage_Taking_Barricade_Incident-Hijacking-Assassination-Hostage_Taking_Kidnapping_.csv']

for period in DATES:
    begin = period[0]
    end = period[1]
    extended_end = period[2]
    name = 'triplet/' + 'TERROR-'+'_' + begin + '_' + end
    cn.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    cn.splitTS(CSVfile, BEG = begin, END = extended_end, dirname = './split', prefix = begin + '_' + extended_end)


stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

TS_PATH=settings_dict['TS_PATH']
NAME_PATH=settings_dict['NAME_PATH']
LOG_PATH=settings_dict['LOG_PATH']
FILEPATH=settings_dict['FILEPATH']
END=settings_dict['END']
BEG=settings_dict['BEG']
NUM=settings_dict['NUM']
PARTITION=settings_dict['PARTITION']
XgenESeSS=settings_dict['XgenESeSS']
RUN_LOCAL=settings_dict['RUN_LOCAL']

XG = cn.xgModels(TS_PATH,NAME_PATH, LOG_PATH,FILEPATH, BEG, END, NUM, PARTITION, XgenESeSS,RUN_LOCAL)
XG.run(workers=4)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']
print(VARNAME)

cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH, FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=4,gamma=True)
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

#This will error, since by using such small EPS, we will get no models after sorting.
vis.render_network_parallel(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,COLORMAP,horizon,model_nums[0], newmodel_name='newmodel.json')
