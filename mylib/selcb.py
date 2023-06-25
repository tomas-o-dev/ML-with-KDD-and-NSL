#!/usr/bin/env python
# coding: utf-8

# A set of functions to calculate the correlations - 
# sklearn has a Very Slow MI function, but no SU


import numpy as np
from math import log, fsum
from collections import Counter
from itertools import groupby
from operator import itemgetter


# Calculate the entropy of an array:  
#           H(X)
def entropy(x):
    return log(len(x)) - fsum(v * log(v) for v in Counter(x).values()) / len(x)


# Calculate the conditional entropy between two arrays:  
#           H(Y|X)
def conditional_entropy(x, y):
    buf = [[e[1] for e in g] for _, g in 
           groupby(sorted(zip(x, y)), itemgetter(0))]
    return fsum(entropy(group) * len(group) for group in buf) / len(x)


# Calculate the mutual information between two arrays:  
#           I(X;Y) = H(Y) - H(Y|X)
def mutual_info(x, y):
    return entropy(y) - conditional_entropy(x, y)


# Calculate the symmetric uncertainty between two arrays: 
#           U(X,Y) = 2* I(X;Y) / (H(X)+H(Y)) 
def symm_uncert(x, y):
    entropy_x = entropy(x)
    entropy_y = entropy(y)
    return 2 * ((entropy_x - conditional_entropy(y, x))
                / (entropy_x + entropy_y))


# get correlations between each feature and the target
def get_ycor(flist, indf, ingt):
    ctbl=[]
    for f in range(len(flist)):
        arr=np.array(indf[flist[f]])
        mi=mutual_info(arr, ingt)  
        su=symm_uncert(arr, ingt)
        bc = np.corrcoef(arr, ingt)[0,1]
        
        tmp=[flist[f],round(bc,4),round(su,4),round(mi,4)]
        if tmp not in ctbl:
            ctbl.append(tmp)
    return ctbl


# floor filter: keep if suy > min or pcy > min
# defaults are from experiments
def filter_fcy(indf, ingt, minpc=0.032, minsu=0.0023):
    ykeep = []
    ydrop = []
    
    cl = list(indf.columns)
    rr = get_ycor(cl, indf, ingt)
    for j in range(len(rr)):
        if (rr[j][2] > minpc) or (rr[j][2] > minsu):
            ykeep.append(rr[j])
        else:
            ydrop.append(rr[j])

    return ydrop, ykeep


# process ydrop, ykeep
def get_filter(ctbl):
    flist=[]
    for k in range(len(ctbl)):    
        if ctbl[k][0] not in flist:
            flist.append(ctbl[k][0])
    return flist


# print ydrop, ykeep
def rpt_ycor(data):
    colwid = max(len(str(word)) for row in data for word in row)+1    # +i for padding
    print("{: <{colwid}} {: >12} {: >12} {: >12}".format(
        '--Feature--','PCy   ','SUy   ','MIy   ',colwid=colwid))
    for k in range(len(data)):
        print("{: <{colwid}} {: >12} {: >12} {: >12}".format(
            data[k][0],data[k][1],data[k][2],data[k][3],colwid=colwid))

