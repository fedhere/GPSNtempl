from Functions import *
from savgol import savitzky_golay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import gzip
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pickle as pkl
import george
from george import kernels
from select_lc import *
from numpy import math
import traceback
import sys
from tqdm import tqdm

from GPSNtempl.maketemplates.elastic_data.Functions import lc_fit

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
import templutils as templutils




SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']

bands = ['R','V','r','g','U','u','J','B','H','I','i','K','m2','w1','w2']
colorTypes = {'IIb':'FireBrick',
             'Ib':'SteelBlue',
             'Ic':'DarkGreen',
             'Ic-bl':'DarkOrange',
             'Ibn':'purple'}

lsst_bands = {'0':'u',
              '1':'g',
              '2': 'r',
              '3': 'i',
              '4': 'z',
              '5': 'y'}

clrs =  {'0':'b',
              '1':'g',
              '2': 'r',
              '3': 'purple',
              '4': 'cyan',
              '5': 'k'}

dirs = ['ELASTICC_TRAIN_SNIb+HostXT_V19/',
        'ELASTICC_TRAIN_SNIb-Templates/',
        'ELASTICC_TRAIN_SNIc+HostXT_V19/',
        'ELASTICC_TRAIN_SNIc-Templates/',
        'ELASTICC_TRAIN_SNIcBL-+HostXT_V19/']

su = templutils.setupvars()
coffset = su.coffset

directory = dirs[0]
title = directory.split('_')[2]

# sne = read_lc(directory)
# pkl.dump(sne, open(directory + 'all_SNe_table_'+ title +'.pkl', "wb"))
pklf = directory + 'high_' + title + '_peak_covered_low_redshift.pkl'
# selected_lc = select_lc(sne, max_dist= 10)
# pkl.dump(selected_lc, open(pklf, "wb"))
selected_lc = pkl.load(open(pklf, "rb"))


# Test fitting algorithms to a single light curve

keys = np.asarray(selected_lc.keys())
ID = '5'
b = 'r'
# selected_lc = all_selected_lc[1]

t = selected_lc[ID][b]['t']
f = selected_lc[ID][b]['f']
ferr = selected_lc[ID][b]['ferr']

t = t[f>0]
ferr = ferr[f>0]
f = f[f>0]

if len(f) == 0:
    print(ID)

m = 27.5 - 2.5*np.log10(f) #-2.5*np.log10(f/(10**(-0.4*27.5)))
median = np.nanmedian(m)
#         m = m- median
merr = 2.5 / np.log(10) * ferr / f


x_peak_ref = selected_lc[ID]['r']['t'][np.argmax(selected_lc[ID]['r']['f'])]

x_peak = x_peak_ref + coffset[b]
y_peak = m[np.argmin(m)]

# if np.sum(t - x_peak < 0) <3:
#     print('Not enough data before the peak')

low_lim = -25
up_lim = 100

ind = (t < up_lim + x_peak) & (t > low_lim + x_peak)
y = m[ind]
yerr = merr[ind]
x = t[ind]
xx = x - x_peak_ref
f = f[ind]
ferr = ferr[ind]

t_new, func, p0 = lc_fit(np.row_stack((t[ind], f, ferr)), x_peak = x_peak_ref)

m_func = 27.5 - 2.5*np.log10(func) #-2.5*np.log10(f/(10**(-0.4*27.5)))
# median = np.nanmedian(m_func)
# m_func = m_func - m_func[np.argmin(m_func)]