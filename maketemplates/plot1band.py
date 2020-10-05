
# author Federica B Bianco
# makes a plot of three example lightcurves '93J','02ap','10bh' each in 1 band
# currently .... not working, missing 10bh --- required fixing a few things!

 
import numpy as np
import glob
import os
import inspect
import sys
from numpy import convolve
import matplotlib.gridspec as gridspec

from builtins import input


try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

RIri = False

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
from sklearn import gaussian_process
import optparse
import readinfofile as ri

import pandas as pd
import pickle as pkl
import snclasses as snstuff
import pylabsetup
print (pylabsetup.__file__)
su = setupvars()

SAVEFIG = True #set to true to save the figure, false will just show it
SAVEFIG = False

fig = pl.figure(figsize=(5,15))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
 
for f,tb, ax, ylim in zip(['93J','02ap','10bh'],['V','R','I'],
                          [ax1,ax2,ax3], [(14, 10.2),
                                          (18,11.2), (23, 18.5)]):
    print (f, tb, ax, ylim)
    input()
    # read and set up SN and look for photometry files
    print (" looking for files ")
    
    thissn = snstuff.mysn(f, addlit=True)
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        continue
    # read metadata for SN
    print("reading file")
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    # setting date of maximum if not in metadata
    if np.isnan(thissn.Vmax) or thissn.Vmax == 0:
        # only trust selected GP results (only one really)
        if '06gi' in thissn.snnameshort:
            try:
                print ("getting max from GP maybe?")
                thissn.gp = pkl.load(open('gplcvs/' + f + \
                                          "_gp_ebmv0.00.pkl", "rb"))
                if thissn.gp['maxmjd']['V'] < 2400000 and \
                   thissn.gp['maxmjd']['V'] > 50000:
                    thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                else:
                    thissn.Vmax = thissn.gp['maxmjd']['V']
                    
                print ("GP vmax", thissn.Vmax)
                #if not raw_input("should we use this?").lower().startswith('y'):
                #    continue
            except IOError:
                continue

    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        continue
    
    # load data
    print (" starting loading ")    
    lc, flux, dflux, snname = thissn.loadsn2(verbose=True, superverbose=True)
    # set up photometry
    thissn.setphot()
    thissn.getphot() 
    if np.array([n for n in thissn.filters.values()]).sum() == 0:
        continue
    
    #thissn.plotsn(photometry=True)
    thissn.setphase()
    thissn.printsn()
    #if f == '93J':
    thissn.plotsn(photometry=True, band = tb, fig=fig,
                      ax=ax, ylim=ylim, xlim=(thissn.Vmax-2400000-30,
                                              thissn.Vmax-2400000+115))
    #pl.show()
    
    if f == '93J':
        #ax.set_xlim(ax.get_xlim()[0],ax.get_xlim()[0]+140)
        ax.set_ylim(14.5,ax.get_ylim()[1])
    ax.set_ylabel("Mag")
    #ax.set_xlabel("JD - 2453000.00")
    #pl.show()    

if SAVEFIG:
    pl.tight_layout()
    pl.savefig("Bianco16_lcvexamples.pdf")
    os.system("cp Bianco16_lcvexamples.pdf %s/papers/SESNtemplates/figs"%os.getenv("DB"))

else:
    pl.show()

