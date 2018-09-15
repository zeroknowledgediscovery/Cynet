
import cynet.cynet as sp
import pandas as pd
import numpy as np
from numpy import genfromtxt
import yaml
import glob

LOGFILE= 'terror.csv'
STOREFILE='terror.p'

EPS=5
DATES = []
for year in range(1995, 2014):
    period_start = str(year) + '-01-01'
    period_end = str(year + 4) + '-12-31'
    DATES.append((period_start, period_end))

grid={'Latitude':np.around(np.linspace(-4,49,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-16,84,EPS),decimals=5),
      'Eps':EPS}

tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])

S0=sp.spatioTemporal(log_file=LOGFILE,
                     log_store=STOREFILE,
                     DATE=None,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     value_limits=[0,10000],
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
S00=sp.spatioTemporal(log_store=STOREFILE,
                     DATE=None,
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

S1=sp.spatioTemporal(log_store=STOREFILE,
                     DATE=None,
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
    name = 'triplet/' + 'TERROR-'+'_' + begin + '_' + end
    sp.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    sp.splitTS(CSVfile, BEG = begin, END = end, dirname = './split', prefix = begin + '_' + end)


stream = file('config_pypi.yaml', 'r')
settings_dict=yaml.load(stream)

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

XG = sp.xgModels(TS_PATH,NAME_PATH, LOG_PATH,FILEPATH, BEG, END, NUM, PARTITION, XgenESeSS,RUN_LOCAL)
XG.run(workers=4)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
VARNAME=['ALL']

sp.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH)
VARNAMES=['Personnel','Infrastructure','Casualties']
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
sp.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)

model_path=settings_dict['FILEPATH']+'*model.json'
MAX_DIST=settings_dict['MAX_DIST']
MIN_DIST=settings_dict['MIN_DIST']
MAX_GAMMA=settings_dict['MAX_GAMMA']
MIN_GAMMA=settings_dict['MIN_GAMMA']
COLORMAP=settings_dict['COLORMAP']

sp.render_network(model_path,DATA_PATH,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,COLORMAP,horizon,model_nums[3], newmodel_name='newmodel.json')
