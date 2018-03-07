#!/usr/bin/python

from spin import viz
import pandas as pd
import numpy as np
from spin import uNetworkModels as models

import warnings
warnings.filterwarnings("ignore")

M=models('../data/model.json')
M.augmentDistance()
M.select(var='gamma',n=25,store='tmp1.json',reverse=True)

viz('tmp1.json',jsonfile=True,figname='figx',res='c',drawpoly=True)
