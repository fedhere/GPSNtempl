import glob 
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pickle as pkl
import pandas as pd
import scipy as sp
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats 
from scipy import integrate

from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.interpolate import UnivariateSpline,splrep, splev
import scipy.optimize as op
from scipy import interpolate

import george
from george import kernels
from george.kernels import ExpSquaredKernel


#s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
#pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl
import plotutils as plotutils

pl.rcParams['font.size'] = 20
colors = {"Ib":'IndianRed', "IIb":'SteelBlue',
          "Ic":'DarkOrange', "Ic-bl":'DarkGreen'}
su = templutils.setupvars()
bands = su.bands
nbands = len(bands)

dt = 0.5
t = np.arange(-15,50,dt)
print("starting")

for j,b in enumerate(bands):
    bb = b
    if b == 'i':
        bb = 'ip'
    if b == 'u':
        bb = 'up'
    if b == 'r':
        bb = 'rp'
    print (b)
    tmpl = {}
    for snt in ["Ib", "IIb", "Ic", "Ic-bl"]:

        templatePkl = "outputs/GPalltemplfit_%s_%s_V0.pkl"%(snt,bb)
        
        tmpl[snt] = pkl.load(open(templatePkl, "rb"))

    done =[]
    
    for snt1 in ["Ib", "IIb", "Ic", "Ic-bl"]:
        for snt2 in ["Ib", "IIb", "Ic", "Ic-bl"]:
            if snt1 == snt2 :continue
            if (snt2,snt1) in done:
                continue
            print (snt2,snt1)
            fig = pl.figure(figsize=(15,10))
            ax = fig.add_subplot(211)
            print (t.shape, (~tmpl[snt1]['average'].mask).sum(),
                   tmpl[snt1]['variance'].mask.sum())
            ax.fill_between(tmpl[snt1]['t'],
                            tmpl[snt1]['average'] - tmpl[snt1]['stdev'],
                            tmpl[snt1]['average'] + tmpl[snt1]['stdev'],
                            alpha=0.3,
                            color=colors[snt1])

            ax.fill_between(tmpl[snt2]['t'],
                            tmpl[snt2]['average'] - tmpl[snt2]['stdev'],
                            tmpl[snt2]['average'] + tmpl[snt2]['stdev'],
                            alpha=0.3,
                            color=colors[snt2])

            ax.fill_between(tmpl[snt1]['t'], 
                            tmpl[snt1]['average'] - tmpl[snt1]['variance'],
                            tmpl[snt1]['average'] + tmpl[snt1]['variance'],
                            alpha=0.5,
                            color=colors[snt1])

            ax.fill_between(tmpl[snt2]['t'], 
                            tmpl[snt2]['average'] - tmpl[snt2]['variance'],
                            tmpl[snt2]['average'] + tmpl[snt2]['variance'],
                            alpha=0.5,
                            color=colors[snt2])

            ax.plot(tmpl[snt1]['t'], tmpl[snt1]['average'], '-', c=colors[snt1], label=snt1)
            ax.plot(tmpl[snt2]['t'], tmpl[snt2]['average'], '-', c=colors[snt2], label=snt2)
            ax.set_ylabel("mag")
            ax.set_xlim(-18,43)
            ylim = ax.get_ylim()
            ax.legend()
            ax.grid(True)                        
            ax.set_title(bb)

            ax = fig.add_subplot(212)

            ax.fill_between(tmpl[snt1]['t'][t<30],
                            tmpl[snt1]['averageShifted'][t<30] - tmpl[snt1]['stdShifted'][t<30],
                            tmpl[snt1]['averageShifted'][t<30] + tmpl[snt1]['stdShifted'][t<30],
                            alpha=0.3,
                            color=colors[snt1])

            ax.fill_between(tmpl[snt2]['t'][t<30],
                            tmpl[snt2]['averageShifted'][t<30] - tmpl[snt2]['stdShifted'][t<30],
                            tmpl[snt2]['averageShifted'][t<30] + tmpl[snt2]['stdShifted'][t<30],
                            alpha=0.3,
                            color=colors[snt2])
            
            ax.fill_between(tmpl[snt1]['t'][t<30],
                            tmpl[snt1]['averageShifted'][t<30] -
                            tmpl[snt1]['varianceShifted'][t<30],
                            tmpl[snt1]['averageShifted'][t<30] +
                            tmpl[snt1]['varianceShifted'][t<30],
                            alpha=0.5,
                            color=colors[snt1])

            ax.fill_between(tmpl[snt2]['t'][t<30],
                            tmpl[snt2]['averageShifted'][t<30] -
                            tmpl[snt2]['varianceShifted'][t<30],
                            tmpl[snt2]['averageShifted'][t<30] +
                            tmpl[snt2]['varianceShifted'][t<30],
                            alpha=0.5,
                            color=colors[snt2])

            ax.plot(tmpl[snt1]['t'][t<30], tmpl[snt1]['averageShifted'][t<30], '-',
                    c=colors[snt1], label=snt1)
            ax.plot(tmpl[snt2]['t'][t<30], tmpl[snt2]['averageShifted'][t<30], '-',
                    c=colors[snt2], label=snt2)
            ax.legend()
            ax.grid(True)            
            ax.set_xlim(-18,43)
            ax.set_ylim(ylim)
            ax.set_xlabel("phase (days)")
            ax.set_ylabel("mag")            

            fname = "GPcompare-%s_%s_%s.pdf"%(bb, snt1, snt2)
            pl.savefig("outputs/" + fname)
            os.system("pdfcrop outputs/%s /Users/fbianco/science/Dropbox/papers/SESNtemplates.working/figs/%s"%(fname, fname))


            done.append((snt1,snt2))
    #sys.exit()
