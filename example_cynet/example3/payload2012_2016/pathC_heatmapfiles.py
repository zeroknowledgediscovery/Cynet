'''
Requires the running of path A to generate the cynet logfiles. Uses them with
flexroc to generate prediction csvs.These csvs will be combined into
integrated csvs which contains information necessary for heatmaps.
'''
import cynet.cynet as cn
import yaml

stream = file('config_pypi.yaml', 'r')
settings_dict = yaml.load(stream)


FLEX_TAIL_LEN = settings_dict['FLEX_TAIL_LEN']
#We want a tpr_threshold of 0.85
cn.flexroc_only_parallel('models/*.log',tpr_threshold=0.85,fpr_threshold=None,FLEX_TAIL_LEN=FLEX_TAIL_LEN, cores=8)

mapper=cn.mapped_events('models/*20models#ALL#*.csv') #Get all csvs where ALL is the source
mapper.concat_dataframes('20modelsALL.csv') #Concat them to one integrated csv.
