#!/usr/bin/env python
# coding: utf-8

# calculate bin cutpoints and labels for a feature

import numpy as np
import pandas as pd


# helper function

def hgbins(feature):
    uv = np.unique(feature)
# https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
# 'doane' makes nicer bins when it can, 
# 'auto' always gives an adequate split

    bnz, egz = np.histogram(uv, bins='doane')
    egz = egz[np.nonzero(bnz)]

    bc = len(egz)
    print('Found',bc,'Bins')

# adjust the bin count
# if there are very few, use a different algorithm
    if len(egz) < 3:
        bnz, egz = np.histogram(uv, bins='auto')
        egz = egz[np.nonzero(bnz)]

    if uv.min != 0:
        egz[0:1] = 0

# if there are a lot, "fold" each lower bin into 
# the next higher one
    while (len(egz) /2) > 6:
        egz[1::2] = [0]*(len(egz)//2)
        egz = egz[np.nonzero(egz)]
        egz = np.insert(egz, 0, 0)

    if len(egz) != bc:
        print('Adjusted to',len(egz),'Bins')

    edges = np.append(egz,[feature.max()])
    return edges


# public function

def autobin(indf, ftb):

    rv=hgbins(indf[ftb])
    
    if indf[ftb].min() != 0:
        rv[0]=int(indf[ftb].min())
    else:
        rv[0]=0.001
        rv=np.insert(rv,0,-1)

    rv=np.trunc(rv)   
    nm = ["{:.0f}-{:.0f}".format(value, rv[idx+1]-1) for idx, value in enumerate(rv[:-2])]

    if indf[ftb].min() < .001:
        print('Adding a Zero bin')
        nm[0] = '(Zero)'
        nm[1] = 'Under ' + str(int(nm[1][2:])+1)

    aa = 'Above '+ str(int(rv[len(rv)-2])-1)
    nm.append(aa)

    return rv, nm

