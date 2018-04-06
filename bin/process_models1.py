#!/usr/bin/python

from spin import viz
import pandas as pd
import numpy as np
from spin import uNetworkModels as models

import warnings
warnings.filterwarnings("ignore")

M=models('/home/ishanu/jD_1041.json')
M.augmentDistance()
M.select(var='gamma',n=105,store='tmp1.json',reverse=True)
M0=models('tmp1.json')

#M1=models('../data/Q_75.json')
#M1.augmentDistance()
#M0.append(M1.select(var='gamma',n=5,store='tmp2.json',reverse=True))



#viz('tmp1.json',jsonfile=True,figname='figx',res='c',drawpoly=True)
viz(M0.models,jsonfile=False,figname='figxxx',res='f',drawpoly=False)
M0.setDataFrame().to_csv('tmp1.csv',index=False)
 