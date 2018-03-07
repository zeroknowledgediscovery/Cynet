#!/usr/bin/python

import spin as sp
import pandas as pd
import numpy as np



ts=sp.readTS('../data/TSme.csv',csvNAME='./data/TSme1',
             BEG='2010-01-01',END='2015-12-31')

sp.splitTS('../data/TSme.csv',csvNAME='./data/TSme1',
           BEG='2010-01-01',END='2017-01-01',
           dirname='./data/',prefix="@me")


sp.showGlobalPlot('./data/TSme1.coords',
                  ts='./data/TSme1.csv',
                  cmap='cool')
   
