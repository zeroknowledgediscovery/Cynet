#!/usr/bin/python

import cynet.cynet as sp
import pandas as pd
import numpy as np
from numpy import genfromtxt


LOGFILE= 'terror.csv'
STOREFILE='terror.p'

EPS=50
DATES = []
for year in range(1995, 2012):
    period_start = str(year) + '-01-01'
    period_end = str(year + 4) + '-12-31'
    DATES.append((period_start, period_end))

grid={'Latitude':np.around(np.linspace(-4,49,EPS),decimals=5),
      'Longitude':np.around(np.linspace(-16,84,EPS),decimals=5),
      'Eps':EPS}


tiles=list([[grid['Latitude'][i],grid['Latitude'][i+1],grid['Longitude'][j], grid['Longitude'][j+1]]
            for i in np.arange(len(grid['Latitude'])-1)
            for j in np.arange(len(grid['Longitude'])-1)])



S0=sp.spatioTemporal(log_file=LOGFILE,
                     log_store=STOREFILE,
                     #DATE=None,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     value_limits=[0,10000],
                     #types=[['BURGLARY','THEFT','MOTOR VEHICLE THEFT']],
                     #value_limits=None,
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2016',
                     freq='D',
                     EVENT='nkill',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)

S0.fit(csvPREF='NKILL')

tiles = S0.getGrid()


S00=sp.spatioTemporal(log_store=STOREFILE,
                     #DATE=None,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     types=[['Bombing/Explosion','Facility/Infrastructure Attack']],
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2016',
                     freq='D',
                     EVENT='attacktype1_txt',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)

S00.fit(csvPREF='')

S1=sp.spatioTemporal(log_store=STOREFILE,
                     #DATE=None,
                     year='iyear',
                     month='imonth',
                     day='iday',
                     types=[['Armed Assault', 'Hostage Taking (Barricade Incident)','Hijacking','Assassination','Hostage Taking (Kidnapping) ']],
                     grid=tiles,
                     init_date='1/1/1995',
                     end_date='1/1/2016',
                     freq='D',
                     EVENT='attacktype1_txt',
                     coord1='latitude',
                     coord2='longitude',
                     threshold=0.025)
S1.fit(csvPREF='')


CSVfile=['NKILL.csv','Bombing_Explosion-Facility_Infrastructure_Attack.csv', 'Armed_Assault-Hostage_Taking_Barricade_Incident-Hijacking-Assassination-Hostage_Taking_Kidnapping_.csv']
#sp.readTS(CSVfile,csvNAME='TERROR-'+'_',BEG='1995-01-01',END='2015-12-31')
#sp.splitTS(CSVfile,BEG='1995-01-01',END='2016-01-01',dirname='./split1',prefix='@'+'TR')
#sp.splitTS(CSVfile,BEG='2012-01-01',END='2015-12-31',dirname='./split2',prefix='@'+'TR')

for period in DATES:
    begin = period[0]
    end = period[1]
    name = 'triplet/' + 'TERROR-'+'_' + begin + '_' + end
    sp.readTS(CSVfile,csvNAME=name,BEG=begin,END=end)
    sp.splitTS(CSVfile, BEG = begin, END = end, dirname = './split', prefix = begin + '_' + end)
