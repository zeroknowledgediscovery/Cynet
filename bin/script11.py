#!/usr/bin/python

from cynet import viz
import cynet as cn
import pandas as pd
import numpy as np
from cynet import uNetworkModels as models
import os

import warnings
warnings.filterwarnings("ignore")

def getModel(jsonfile,n=5):
    M=models(jsonfile)
    M.augmentDistance()
    M.select(var='gamma',n=n,reverse=True,store='tmp.json')

N=5
path='../data/'
jsonFiles = [f for f in os.listdir(path) if f.endswith('.json')]
M0={}

for f in jsonFiles:
    print f
    getModel(path+f,n=N)
    viz('tmp.json',jsonfile=True,figname='figz'+f,res='f',drawpoly=False,colormap='seismic')
