'''
This path proceeds right after the models are produced from XgenESeSS. This should
be the first script that is run in this example. We will use the run_pipeline
function. This function will take each model, and apply the cynet binary to it
according to the settings set in config_pypi.yaml. This creates the cynet log files
and place them in the models folder. Afterwards, flexroc is called to get
auc, tpr, and fpr statistics. These results are put into new files, .res files.
The .res files are aggregated into a singular file, which we will use to make plots
it pathB.
'''
import cynet.cynet as cn
import yaml
import glob

DEBUG_=False

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

#Load settings
model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
VARNAME=list(set([i.split('#')[-1] for i in glob.glob(DATA_PATH+"*")]))+['ALL']
print(VARNAME) #VARNAME should not just be ['ALL']

cn.run_pipeline(MODEL_GLOB,model_nums, horizon, DATA_PATH, RUNLEN, VARNAME, RESPATH, FLEX_TAIL_LEN=FLEX_TAIL_LEN,cores=8,distance=True)
#Use run_pipeline as described above. Here we are sorting by distance.
