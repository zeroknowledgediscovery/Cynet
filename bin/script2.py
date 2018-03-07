#!/usr/bin/python

import spZED as sp
import pandas as pd
import numpy as np
import cProfile

EPS=3
grid={'latitude':np.arange(-56,77,EPS),
      'longitude':np.arange(-179,182,EPS),
      'Eps':EPS}

LOGFILE='../../../../../ZED/Research/DATA_REPO/terror.csv'

S=sp.spatioTemporal(log_store='alldata.p',
                    log_file=LOGFILE,
                    DATE=None,
                    year='iyear',
                    month='imonth',
                    day='iday',
                    value_limits=[0,10000],
                    grid=grid,
                    init_date='1/1/2001',
                    end_date='1/1/2017',
                    freq='D',
                    EVENT='nkill',
                    coord1='latitude',
                    coord2='longitude',
                    threshold=0.05)


S.fit()
#S._types=[['HOMICIDE']]

#S.fit(THRESHOLD=0.01)

#S.generateNeighborMap()

#print S._ts_dict

S.showGlobalPlot(figname='terrorplot')
