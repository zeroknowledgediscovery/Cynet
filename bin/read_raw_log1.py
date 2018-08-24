#!/usr/bin/python

import cynet as cn
import pandas as pd
import numpy as np

EPS=2
grid={'latitude':np.arange(-4,49,EPS),
      'longitude':np.arange(-16,84,EPS),
      'Eps':EPS}

LOGFILE='/home/ishanu/Dropbox/ZED/Research/DATA_REPO/terror.csv'

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
                    threshold=0.025)


S.fit(csvPREF='TSme')
S.showGlobalPlot(figname='terrorplotME')
