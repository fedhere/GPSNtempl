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

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
import templutils as templutils

lc_direc = os.getenv("SESNPATH") + './../Somayeh_contributions/main/ELASTICC_lc/'

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

su = templutils.setupvars()
coffset = su.coffset

ref = coffset['r']
for b in coffset.keys():
    coffset[b] = coffset[b] - ref

# To select a sample of the Elasticc light curves with low redshift,
#high S/N ratio and at least 1 data point 5 days around the peak:
directory = lc_direc + 'ELASTICC_TRAIN_SNIb+HostXT_V19/'

# To read in saved selected lc:

pklf = directory + 'high_SNIb+HostXT_peak_covered_low_redshift.pkl'
selected_lc = pkl.load(open(pklf, "rb"))