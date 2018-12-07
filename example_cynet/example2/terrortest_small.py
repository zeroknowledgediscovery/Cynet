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

'''
This example pipeline will serve as an example of how to use the major features
of the package. We use EPS=5 so that there are fewer tiles to calculate and so
that the entire thing runs more smoothly. However, the results will look weird
because of this. Increase the EPS for longer runtime, but more appropriate looking
results.

To use this script, you will need to have installed this package, and cartopy.
You will also need to have the terror.csv file.
'''

LOGFILE= 'terror.csv'
STOREFILE='terror.p'

EPS=5 #ONLY AN EXAMPLE. USE A LARGER EPS.
'''
Here we generate the dates,grid and tiles. for our fit. We do it for multiple
periods between 1995 and 2018.

Dates: Notice there is period start, period end, and period extended
end. For each period, period start and end represents the periods of training data.
Period extended end is the end date of data beyond our training, the test data.
In the example below, the first period has training data from 1995-01-01 to
1999-01-01. The training period, is 1999-01-01 to 2000-01-01, a year long.

grid: Here is where EPS is used. Notice that the higher the EPS, the more squares will
be used.

tiles: We generate the tiles from the grid.
'''

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

'''
Our first fit is S0. Essentially, we are looking for tiles that meet a certain
number of kills (deaths in the column nkills). We are looking for tiles with
number of kills that are greater than a certain threshold. Here that threshold
is 0.025.

The NKILL.csv is outputted. However, more importantly, the internal timeseries
datafram is changed.
'''

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
'''
Note that we are now going to use the tiles selected for in S0. S00 will be our fitting
for attack types in the categories ['Bombing/Explosion','Facility/Infrastructure Attack'].
We are counting the number of these types of events that happen in these tiles.
Note the significance threshold is still the same.

Output is the Bombing_Explosion-Facility_Infrastructure_Attack.csv. This contains
the timeseries for those types of attacks in the tiles selected for.
'''
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
'''
Similar to S00, S1 does selection fitting for the attack types:
['Armed Assault', 'Hostage Taking (Barricade Incident)','Hijacking','Assassination','Hostage Taking (Kidnapping) ']
Output is similar as well.
'''
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

'''
Now we will make use of the csvs created in previous steps. Note that the list
CSVfile contains those files. We will also make use of the dates we generated earlier.
For each period, we will generate training data, with readTS. The time of this
training data is defined by begin and end. We also produce splitfiles, which are
created by splitTS. This period is defined by begin and extended_end. As you may recall,
extended_end in this example is one year beyond end.

Inputs:
    -CSVfiles.
    -Dates.
Outputs:
    -triplets. (coord, column, .csv) files which will represent a dataframe of events.
    Each row in this dataframe represents a locaton and a type.
    -split. training data. The names of these training data designate a period,
    location, and type. There will be many of these.
'''
CSVfile=['NKILL.csv','Bombing_Explosion-Facility_Infrastructure_Attack.csv', 'Armed_Assault-Hostage_Taking_Barricade_Incident-Hijacking-Assassination-Hostage_Taking_Kidnapping_.csv']

for period in DATES:
    begin = period[0]
    end = period[1]
    extended_end = period[2]
    name = 'triplet/' + 'TERROR-'+'_' + begin + '_' + end
    cn.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    cn.splitTS(CSVfile, BEG = begin, END = extended_end, dirname = './split', prefix = begin + '_' + extended_end)
'''
Now we have our training and testing data ready. It is time to create the models.
The next few lines are just reading in settings from our config_pypi.yaml file.
xgModels are where the models get generated. Note that this should be done on a cluster.
By setting RUN_LOCAL to 0 (false), xgModels.run(*args) will produce a list of
calls. 'programcalls.txt' that needs to be run to produce all the models. One
call per model. In this example, we will use run xgModels locally. This is slower
than running on a cluster, due to fewer core count, however, since our EPS is low,
it will still run with reasonabletime.

Inputs: training data, triplets produced by readTS.
Outputs: model.json files which each represnt models.
'''

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
'''
Here we being evaluating our models. We read more settings in from the yaml file.
Note that VARNAME should be all of the types. run_pipeline will take each model
and first select for the appropriate number of models, designated by model_nums.
Note that gamma, is true, so we are sorting by gamma. To sort by distance, use
argument distance=True and remove gamma=True argument.

What happens to each model (for each model_num and source type):
    1. Selects a certain number of models. By either gamma or distance. Creates
        a model_sel json file. A filtered version of the originial.
    2. Applies cynet to the model_sel file, which generates a cynet logfile with
        predictions.
    3. Applies flexroc, once for each target type. (Each type and ALL). which returns
        test statistics. Auc, fpr, tpr. Stores this as .res files

Output:
After .res files are produced, we collect them and put them in a csv. Here it is
named 'all_res.csv'.
'''
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
'''
flexroc_only_parallel is not technically necessary in our rexample here. However,
it will produce the csv files necessary to make heatmaps. This takes every log file
(which were produced in the last step.), applies a desired tpr or fpr threshold to
it, and produces a copy of the log file as a csv with the predictions.

Output: csv files with predictions, used for drawing heatmaps. Not included in this example.
'''
cn.flexroc_only_parallel('models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=4)

VARNAMES=['Personnel','Infrastructure','Casualties']
'''
Draw plots of statistics here. Auc, tpr, fpr etc.
Output: statistics maps.
'''
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2','vartgt'],varname='fpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='tpr',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='auc',VARNAMES=VARNAMES)
cn.get_var('res_all.csv',['lattgt1','lattgt2','lontgt1','lontgt2'],varname='fpr',VARNAMES=VARNAMES)

'''
The last step of our example will be to draw a network map. This step can be done
right after the models are created by xgenESeSS. Running run_pipeline and
flexroc_only_parallel is not necessary before this step. Filters each model,
according to settings and combines them into 'newmodel.json' file.
Inputs:
    -split files.
    -models from XGenESeSS.
Outputs:
    -network map. Here will be called 'fig2.pdf.'
'''
model_path=settings_dict['FILEPATH']+'*model.json'
MAX_DIST=settings_dict['MAX_DIST']
MIN_DIST=settings_dict['MIN_DIST']
MAX_GAMMA=settings_dict['MAX_GAMMA']
MIN_GAMMA=settings_dict['MIN_GAMMA']
COLORMAP=settings_dict['COLORMAP']

#This will error, since by using such small EPS, we will get no models after sorting.
vis.render_network_parallel(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,COLORMAP,horizon,model_nums[0], newmodel_name='newmodel.json')
