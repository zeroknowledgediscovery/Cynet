#!/usr/bin/python

from cynet import viz
import cynet as cn
import pandas as pd
import numpy as np
from cynet import uNetworkModels as models
import os
from tqdm import tqdm, tqdm_pandas
from matplotlib import pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings("ignore")

def getModel(jsonfile,n=5):
    M=models(jsonfile)
    M.augmentDistance()
    M.select(var='distance',n=n,
             reverse=False,
             high=2000,low=10,inplace=True)
    return M.select(var='gamma',n=n,
                    reverse=True,
                    high=1.0,low=0.02)

#------------------
N=10
path='../data/'
jsonFiles = [f for f in os.listdir(path)
             if f.endswith('.json')]

#------------------

M0={}
str_=np.arange(60,65).astype('string')
files=[path+'Q_'+i+'.json' for i in str_]

for f in tqdm(jsonFiles):
    if path+f in files:
        M0.update(getModel(path+f,n=N))

cn.to_json(M0,'modelsx'+str(N)+'.json')

#viz('models'+str(N)+'.json',jsonfile=True,figname='fige',
#    res='f',drawpoly=False,colormap='autumn')


M1=models('modelsx'+str(N)+'.json')
#M1.augmentDistance()
M1.setDataFrame(scatter='scatter4')
