#!/usr/bin/python

import cynet as cn
import pandas as pd
import numpy as np
from cynet import uNetworkModels as models

import warnings
warnings.filterwarnings("ignore")



ts=cn.readTS('../data/TSme.csv',csvNAME='./data/TSme1',
             BEG='2010-01-01',END='2015-12-31')

cn.cnlitTS('../data/TSme.csv',csvNAME='./data/TSme1',
           BEG='2010-01-01',END='2017-01-01',
           dirname='./data/',prefix="@me")


cn.showGlobalPlot('./data/TSme1.coords',
                  ts='./data/TSme1.csv',
                  cmap='cool')
