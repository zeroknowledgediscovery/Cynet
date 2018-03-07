#!/usr/bin/python

import spin as sp
import pandas as pd
import numpy as np

EPS=0.1
grid={'Latitude':np.arange(41.5,42,EPS),
      'Longitude':np.arange(-87.75,-87.2,EPS),
      'Eps':EPS}

S=sp.spatioTemporal(log_store='alldata.p',
                    types=[['BURGLARY','THEFT'],['HOMICIDE']],
                    value_limits=None,
                    grid=grid,
                    init_date='1/1/2001',
                    end_date='1/1/2002',
                    freq='6h',
                    threshold=0.1)


S.fit()
S._types=[['HOMICIDE']]

S.fit(THRESHOLD=0.01)

S.generateNeighborMap()

print S._ts_dict
