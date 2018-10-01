import viscynet.viscynet as vis
import yaml

stream = file('config_pypi.yaml', 'r')
settings_dict=yaml.load(stream)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
VARNAME=['ALL']

model_path=settings_dict['FILEPATH']+'*model.json'
MAX_DIST=settings_dict['MAX_DIST']
MIN_DIST=settings_dict['MIN_DIST']
MAX_GAMMA=settings_dict['MAX_GAMMA']
MIN_GAMMA=settings_dict['MIN_GAMMA']
COLORMAP=settings_dict['COLORMAP']

vis.render_network_parallel(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,\
COLORMAP,horizon,model_nums[3], newmodel_name='newmodel.json',workers=4,rendered_glob = 'models/*_rendered.json',figname='newfig')
