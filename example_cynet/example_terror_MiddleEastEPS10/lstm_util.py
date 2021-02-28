import pandas as pd
import numpy as np
from subprocess import check_output
import string
import random
import re
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed

import concurrent.futures

def getData(X_raw, horizon, train_len, tgts=None, verbose=False):
	
	"""
	Organize data in the original form (num_samples, length_in_time)
	to a form that can be fed directly to RNN.
	"""
	
	if tgts is None:
		tgts = range(len(X_raw))

	X = X_raw.T
	Y = X_raw[tgts].T

	X_train = X[:train_len - horizon]
	Y_train = Y[horizon: train_len]

	X_test = X[train_len - horizon: -horizon]
	Y_test = Y[train_len:]
	
	# Reshape to fit RNN
	# The dimension of RNN input/output is (num_samples, length_in_time, data_dimension)
	X_train = X_train.reshape(1, *X_train.shape)
	Y_train = Y_train.reshape(1, *Y_train.shape)

	X_test = X_test.reshape(1, *X_test.shape)
	Y_test = Y_test.reshape(1, *Y_test.shape)

	if verbose:
		print('Training data: input dim = {}, output dim = {}'.format(X_train.shape, Y_train.shape))
		print('Out-sample data: input dim = {}, output dim = {}'.format(X_test.shape, Y_test.shape))

	return X_train, Y_train, X_test, Y_test


def train(
	X_train, 
	Y_train, *, 
	units=None, 
	epochs=200, 
    loss='mse',
	load_weight_path=None):

	model = tf.keras.Sequential()

	if units is None:
		units = [100, 10]
		
	for layer, num in enumerate(units):
		if layer == 0: 
			model.add(LSTM(units=num, input_shape=(None, X_train.shape[-1]), return_sequences=True))
		else:
			model.add(LSTM(units=num, return_sequences=True))
	
	# One output layers
	model.add(TimeDistributed(Dense(units=Y_train.shape[-1]))) #, activation='sigmoid'
	
	model.compile(loss='mse', optimizer='adam')

	if load_weight_path is not None:
		model.load_weights(load_weight_path)

	model.summary()

	model.fit(X_train, Y_train, epochs=epochs, batch_size=1, verbose=1)

	return model


def get_flexroc_single(args):
	"""
	y_true is a 1-D array of ground truth
	predictions is a 1-D array of predictions
	logfile: a dataframe with the first column being the predictions 
		and the second column, the ground truth.
		If the file name is None, generate the file temporarily and remove after use.
		If the file name is given, keep the file.
		NOTE: logfile is SPACE separated for flexroc binary
	rcfile: a dataframe with the first column being the fpr 
		and the second column, the tpr.
		If the file name is None, don't ask the flexroc binary to generate the file.
		If the file name is given, generate the file and keep.
	"""

	y_true, predictions, logfile, rcfile, rc_suffix = args
	

	if not np.any(y_true):
		return -1

	if logfile is None:
		logfile = 'tmp_' + ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)]) + '.csv'
	
	df = pd.DataFrame(data={'grt': y_true, 'prd': predictions})
	df = df[['grt', 'prd']]
	df.to_csv(logfile, header=None, index=None, sep=' ')

	if rcfile is not None:
		cmd = '../bin/flexroc -i {} -w 1 -x 0 -C 1 -L 1 -E 0 -f .2 -t .9 -r {}'.format(logfile, rcfile)
	else:
		cmd = '../bin/flexroc -i {} -w 1 -x 0 -C 1 -L 1 -E 0 -f .2 -t .9'.format(logfile)
	output = check_output(cmd, shell=True)
	
	output = output.decode('utf8')
	auc = float(output.split()[1])

	if rcfile is not None:
		if rc_suffix is not None:
			name_col = 'tpr_{}'.format(rc_suffix)
			name_index = 'fpr_{}'.format(rc_suffix)
		else:
			name_col, name_index = 'tpr', 'fpr'
		rc = pd.read_csv(rcfile, index_col=0, sep=' ', names=[name_col])
		rc.index.name = name_index
		rc.to_csv(rcfile)
	
	if logfile is None:
		check_output('rm {}'.format(logfile), shell=True)
	
	return auc


def get_flexroc_parallel(
	y_true, 
	predictions, 
	logfile_prefix=None,
	rcfile_prefix=None,
	max_workers=None,
	verbose=False):

	args = []
	for i in range(len(y_true)):
		if logfile_prefix is not None:
			logfile = '{}_{}'.format(logfile_prefix, i)
		if rcfile_prefix is not None:
			rcfile = '{}_{}'.format(rcfile_prefix, i)
		arg = [y_true[i], predictions[i], logfile, rcfile, i]
		args.append(arg)
		
	rnn_auc = []
	with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
		for arg, auc in zip(args, executor.map(get_flexroc_single, args)):
			i = arg[-1]
			rnn_auc.append(auc)
			if verbose:
				print('{}: {}'.format(i, auc))

	return rnn_auc



def get_information(data_dir):
	
	"""
	This function assumes a fixed structure of the the data folder.
	But can be make into a comprehensive and robust one if needed.
	Works find with Atlanta, Detroit, Philadelphia, and San Francisco.
	"""
	
	coordfile = glob('{}/triplet/*coords'.format(data_dir))[0]
	csvfile = glob('{}/triplet/*csv'.format(data_dir))[0]
	
	start_date, end_date = coordfile.split('/')[-1].split('.')[0].split('_')[1:]
	split_example = glob('{}/split/{}_*'.format(data_dir, start_date))[0]
	oos_end_date = re.findall(r'(\d{4}-\d{2}-\d{2})', split_example)[-1]
	split_prefix = '{}/split/{}_{}'.format(data_dir, start_date, oos_end_date)
	
	train_len = np.genfromtxt(csvfile).shape[-1]
	test_len = len(np.genfromtxt(split_example)) - train_len
	
	resolution = len(pd.date_range(start_date, end_date)) / train_len

	info_map = {
		'start_date': start_date,
		'end_date': end_date,
		'oos_end_date': oos_end_date,
		'split_prefix': split_prefix,
		'coordfile': coordfile,
		'train_len': train_len,
		'test_len': test_len,
		'resolution': resolution,
	}
	return info_map


def get_RNN_dataset(
    data_dir, 
    horizon, 
    partition_fname=None, 
    verbose=False):

    """
    For RNN dataset from split and triplet
    """

    info_map = get_information(data_dir)
 
    with open(info_map['coordfile'], 'r') as hd:
        coords = hd.readlines()

    X_raw, var_list = [], []

    if partition_fname is not None:
        partition_df = pd.read_csv(
            partition_fname, 
            names=['tile', 'partition'], 
            sep='\s+', 
            index_col=0)
    
    for i in range(len(coords)):
        coord = coords[i].strip()
        var = coord.split('#')[-1]
        var_list.append(var)

        split = info_map['split_prefix'] + '_' + coord.strip()
        ts = np.genfromtxt(split)
        
        if partition_fname is not None:
            partition = partition_df.loc[coord, 'partition']
            ts[ts < partition] = 0
        
        ts[ts > 0] = 1
        X_raw.append(ts)
        
        if verbose:
            if partition_fname is None:
                print(f'model_id = {i}, var = {var}')
            else:
                print(f'model_id = {i}, var = {var}, partition = {partition}')

    X_raw = np.array(X_raw)    
  
    # ===================== split =====================
    train_len = info_map['train_len']
    X_train, Y_train, X_test, Y_test = getData(
        X_raw, 
        horizon, 
        train_len, 
        verbose=verbose)

    result = pd.DataFrame(
        index=range(len(coords)), 
        data={'var': var_list})

    return X_train, Y_train, X_test, Y_test, result, info_map
