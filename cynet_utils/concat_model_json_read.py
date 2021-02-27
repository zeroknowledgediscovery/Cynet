import pandas as pd
from glob import glob
from subprocess import check_output
import sys

import concurrent.futures


JSON_READ = '/project2/ishanu/bin/json_read'

def run_one(js): 

	"""

	"""

	model_id = js.split('/')[-1].split('m')[0]
	csv_fname = js.rstrip('.json') + 'jr.csv'
	cmd = '{} -j {} -s on > {}'.format(JSON_READ, js, csv_fname)
	check_output(cmd, shell=True)
	csv = pd.read_csv(csv_fname)
	
	return csv	

if __name__ == '__main__':

	model_folder = sys.argv[1]
	models = glob('{}/*model.json'.format(model_folder))

	result_path = sys.argv[2]
        
	
	dfs = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
		for model, csv in zip(models, executor.map(run_one, models)):
			dfs.append(csv)
			print('{}: {}'.format(model, len(csv)))

	df = pd.concat(dfs)
	df.to_csv(result_path, index=False)
        
