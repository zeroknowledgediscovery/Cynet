#!/usr/bin/python
import cynet as sp
import pandas as pd
import numpy as np
from numpy import genfromtxt

#STOREFILE='../DATA_REPO/alldata_crime.p'
LOGFILE='terror_small.csv'

EPS=200
grid={'Latitude':np.around(np.linspace(-37.8,59.33,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-157.8,-144.9,EPS),decimals=5),
      'Eps':EPS}


tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])

S = sp.spatioTemporal(log_file=LOGFILE,
                     DATE = None,
                     year = 'iyear',
                     month = 'imonth',
                     day = 'iday',
                     types=[['Bombing/Explosion']],
                     value_limits=None,
                     grid=tiles,
                     init_date='2001-01-01',
                     end_date='2016-05-17',
                     freq='D',
                     EVENT='attacktype1_txt',
                     coord1 = 'latitude',
                     coord2 = 'longitude',
                     threshold = 0.05)

S.fit(grid = tiles)
