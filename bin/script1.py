#!/usr/bin/python

from spin import viz
import spin as sp
import pandas as pd
import numpy as np
from spin import uNetworkModels as models
import os

import warnings
warnings.filterwarnings("ignore")

def getModel(jsonfile,n=5):
    M=models(jsonfile)
    M.augmentDistance()
    return M.select(var='delay',n=n,reverse=False)


N=2
path='../data/'
jsonFiles = [f for f in os.listdir(path) if f.endswith('.json')]
M0={}


str_=np.arange(99,110).astype('string')
files=[path+'Q_'+i+'.json' for i in str_]
print files

for f in jsonFiles:
    if f in files:
        print f
        M0.update(getModel(path+f,n=N))

sp.to_json(M0,'models'+str(N)+'.json')
    
viz('models'+str(N)+'.json',jsonfile=True,figname='figz',res='c',drawpoly=False,colormap='jet')
