#!/usr/bin/python

import cynet as cn
import pandas as pd
import numpy as np



ts=cn.readTS('TS.csv',csvNAME='TS1',
             BEG='2010-01-01',END='2015-12-31')

#ts.to_csv('TS1.csv',sep=" ")

cn.cnlitTS('TS.csv',csvNAME='TS1',
           BEG='2010-01-01',END='2017-01-01',
           dirname='./data/',prefix="@")


num,fig,ax,cax=cn.showGlobalPlot(coords='TS1.coords',ts='TS1.csv',F=2,cmap='jet')
