#!/usr/bin/python

import os
import sys

modulename='cynet'

total = len(sys.argv)
if total > 1:
    modulename = sys.argv[1]

fname=modulename+'_help.txt'

module = __import__(modulename)
sys.stdout = open(fname, "w")
help(module)
