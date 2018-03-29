#!/usr/bin/python

import spin as sp
import pandas as pd 
import numpy as np

EPS=0.003
grid={'Latitude':np.arange(41.5,42,EPS),
      'Longitude':np.arange(-87.75,-87.2,EPS),
      'Eps':EPS}
#log_file='/home/ishanu/Dropbox/ZED/Research/DATA_REPO/Crimes_-_2001_to_present.csv',
S=sp.spatioTemporal(log_store='alldata_crime.p',
                    types=[['BURGLARY','THEFT'],['HOMICIDE']],
                    value_limits=None,
                    grid=grid,
                    init_date='1/1/2001',
                    end_date='1/1/2017',
                    freq='12h',
                    threshold=0.05)


S.fit(csvPREF='TScrime')
#S._types=[['HOMICIDE']]
#S.fit(THRESHOLD=0.01)
#S.generateNeighborMap()
#
#print S._ts_dict
#S.showGlobalPlot(figname='crime')
