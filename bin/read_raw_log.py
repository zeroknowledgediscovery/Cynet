#!/usr/bin/python

import cynet as cn
import pandas as pd
import numpy as np

# define grid parameters
EPS=0.003
grid={'Latitude':np.arange(41.5,42,EPS),
      'Longitude':np.arange(-87.75,-87.2,EPS),
      'Eps':EPS}

# point to input raw data (as a csv)
_log_file='../../data/ChicagoCrimes2001-2018.csv',

S=cn.spatioTemporal(log_file=_log_file,
                    types=[['BURGLARY','THEFT'],['HOMICIDE']],
                    value_limits=None,
                    grid=grid,
                    init_date='1/1/2001',
                    end_date='1/1/2017',
                    freq='12h',
                    threshold=0.05)

# get timeseries with given type filters
# and no value limits and auto-detect for best frequency interval
# deafault max frequency is 24 Hrs and can be adjusted using max_incr=
# as a parameter to fit() or timeseries() methods
S.fit(csvPREF='TScrime',auto_adjust_time=True,max_incr=48)

# Now re-use the same instance fit with new filters to generate a neighbor map
# and visualize it
S._types=[['HOMICIDE']]
S.fit(THRESHOLD=0.01)
S.generateNeighborMap()

print S._ts_dict
S.showGlobalPlot(figname='crime')
