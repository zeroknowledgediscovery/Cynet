'''
This path can be used even before Path A. We render a visualization of the network.
Important parameters for minimum and maximum gamma/distance is defined in
the yaml settings file. This can be used right after xgenesess models.
'''

import yaml
import viscynet.viscynet as vis

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)

model_nums = settings_dict['model_nums']
MODEL_GLOB = settings_dict['MODEL_GLOB']
horizon = settings_dict['horizons'][0]
DATA_PATH = settings_dict['DATA_PATH']
RUNLEN = settings_dict['RUNLEN']
RESPATH = settings_dict['RESPATH']
FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
model_path=settings_dict['FILEPATH']+'*model.json'
MAX_DIST=settings_dict['MAX_DIST']
MIN_DIST=settings_dict['MIN_DIST']
MAX_GAMMA=settings_dict['MAX_GAMMA']
MIN_GAMMA=settings_dict['MIN_GAMMA']
COLORMAP=settings_dict['COLORMAP']

vis.render_network(model_path,MAX_DIST,MIN_DIST,MAX_GAMMA,MIN_GAMMA,COLORMAP,horizon,model_nums[2], newmodel_name='newmodel.json')
