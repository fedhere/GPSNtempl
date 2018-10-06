import numpy as np
import glob
import os
import inspect
import sys
import pickle as pkl
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
    
from bokeh.plotting import Figure as bokehfigure
from bokeh.plotting import save as bokehsave
from bokeh.plotting import show as bokehshow
from bokeh.plotting import figure, output_file
from bokeh.layouts import column
from bokeh.models import  BoxZoomTool, HoverTool, ResetTool, TapTool
from bokeh.models import ColumnDataSource, CustomJS,  Range1d

#, HBox, VBoxForm, BoxSelectTool, TapTool
#from bokeh.models.widgets import Select
#Slider, Select, TextInput
from bokeh.io import gridplot
from bokeh.plotting import output_file
from numpy import convolve
import matplotlib.gridspec as gridspec

archetypicalSNe = ['94I', '93J', '08D', '05bf', '04aw', '10bm', '10vgv']

try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
from makePhottable import *
from colors import hexcolors, allcolors, colormaps

#prepping SN data from scratch
PREP = True 
#PREP = False
font = {'family' : 'normal',
        'size'   : 20}

#setting up snclass and other SESNCFAlib stuff
su = setupvars()

pl.rc('font', **font)

from ubertemplates import wAverageByPhase as wA
from ubertemplates import plotme  as plotme
x = np.array([np.arange(-30,150), np.arange(-30,150), np.arange(-30,150)])

mag = np.array([-np.exp((x[0]/10.)**2), -np.exp((x[0]/30.)**2),
                -np.exp((x[0]/20.)**2)])
dataphases = {'phase': x,
              'x': x,
              'y': mag,

              'yerr': np.array([np.zeros(len(x[0]))+0.1,
                                np.zeros(len(x[0]))+0.05,                        
                                np.zeros(len(x[0]))+0.2]),
              'mag': mag,
              'dmag': np.array([np.zeros(len(x[0]))+0.1,
                                np.zeros(len(x[0]))+0.05,
                                np.zeros(len(x[0]))+0.2]),
              'phases': np.array(x),
              'allSNe': np.array([['tmp1']*len(x[0]),
                                  ['tmp1']*len(x[0]),
                                  ['tmp2']*len(x[1])]),
              'name': np.array(['tmp1']* len(x[0]) +
                               ['tmp2']* len(x[0]) +
                               ['tmp2']* len(x[0])),
              'type': np.array(['Ic']* len(x[0])*3)}

data, ax = plotme (dataphases, 'V')
phs, wmu,  mu, med, std = wA(dataphases, 1.)

print (wmu, med, std)
ax[0].plot(phs, med, 'r-', alpha=0.7, lw=2)
ax[0].plot(phs, wmu, 'k-', lw=2)
ax[0].fill_between(phs, wmu-std, wmu+std,
                   color = 'k', alpha=0.5)        

ax[0].set_ylim(-20,2)
pl.show()
