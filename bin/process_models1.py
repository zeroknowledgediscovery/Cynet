#!/usr/bin/python

from spin import viz
import pandas as pd
import numpy as np
from spin import uNetworkModels as models

import warnings
warnings.filterwarnings("ignore")

FILE='jD_1041'
FILEPATH='/home/ishanu/'

M=models(FILEPATH + FILE + '.json')
M.augmentDistance()
M.select(var='gamma',n=20,store=FILE+'_short.json',reverse=True)
M0=models(FILE+'_short.json')

#M1=models('../data/Q_75.json')
#M1.augmentDistance()
#M0.append(M1.select(var='gamma',n=5,store='tmp2.json',reverse=True))



#viz('tmp1.json',jsonfile=True,figname='figx',res='c',drawpoly=True)
viz(M0.models,jsonfile=False,figname='figxxx',res='f',drawpoly=False)
M0.setDataFrame().to_csv(FILE+'.csv',index=False)
 
