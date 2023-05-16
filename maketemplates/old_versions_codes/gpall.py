import glob
import inspect
import optparse
import time
import copy
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pickle as pkl

import scipy as sp
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats
from scipy import integrate

from scipy.interpolate import InterpolatedUnivariateSpline

from scipy.interpolate import UnivariateSpline, splrep, splev
from scipy import interpolate

import multiprocessing as mpc

import json
import os
import pandas as pd

# f = open("fbb_matplotlibrc.json")
# s = json.load(f)
# # s = json.load("fbb_matplotlibrc.json")
# pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl

mpl.use('agg')

import pylab as pl
from pylab import rc
import plotutils as plotutils

# setting maximum number of processes for parallelization
MAXPROCESSES = 1000
# setting actual number of processes given the present architecture
nps = min(mpc.cpu_count() - 1 or 1, MAXPROCESSES)

VERBOSE = True

# use literature data (if False only CfA data)
LIT = True
# use NIR data too
FNIR = True
PRETTY = True
if PRETTY:
    pl.rcParams["axes.facecolor"] = "#FFF8F0"

# allsne = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNforbol.csv")['SNname'].values

# take SN name from line argument
if len(sys.argv) > 1:
    allsne = [sys.argv[1]]

else:
    # or read all SNe from csv metadata file
    allsne = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values
    PRETTY = False

# set up SESNCfalib stuff
su = templutils.setupvars()
nbands = len(su.bands)


def gpit(thissn):
    '''
     function to interpolat lcv with Gaussian Processes
     '''
    print("hello")
    PICKLE = True
    # PICKLE =  False
    # UNPICKLE = True
    UNPICKLE = False
    PLOT = True

    phaserange = [-20, 100]

    thissn.printsn(photometry=False)

    gpmag = {}

    if UNPICKLE:
        print("\n\n\n", here)
        PICKLE = False
        print('outputs/gplcvs/' + sn + "_gp_ebmv%.2f.pkl" % myebmv,
              'outputs/gplcvs/' + sn + "_gplcv_ebmv%.2f.pkl" % myebmv)
        if os.path.isfile('outputs/gplcvs/' + sn + "_gp_ebmv%.2f.pkl" % myebmv) \
                and os.path.isfile('outputs/gplcvs/' + sn + "_gplcv_ebmv%.2f.pkl" % myebmv):
            thissn.gp = pkl.load(open('outputs/gplcvs/' + sn + "_gp_ebmv%.2f.pkl" % myebmv, "rb"))

            gpmag = pkl.load(open('outputs/gplcvs/' + sn + "_gplcv_ebmv%.2f.pkl" % myebmv, "rb"))
            if PLOT:
                print("hereherehere")
                for b in thissn.filters:
                    if thissn.filters[b] > 1:
                        print("\n\n\n")
                        print("\t", b)
                        print("\n\n\n")
                        fig = pl.figure()
                        phasekey = 'phase'
                        phases = np.arange(thissn.photometry[b][phasekey].min(),
                                           thissn.photometry[b][phasekey].max(), 0.1)
                        if phaserange is None:
                            phases = np.arange(phaserange[0], phaserange[1], 0.1)
                        try:
                            print("creating timeline", phases, phases.min())
                            XX = np.log(phases - phases.min() + 1)
                        except ValueError:
                            print("could not create timeline")
                            continue

                        mu, cov = thissn.gp[b].predict(thissn.gp['gpy'][b], XX)
                        std = np.sqrt(np.diag(cov))

                        ax = fig.add_subplot(121)
                        ax.errorbar(thissn.photometry[b][phasekey],
                                    thissn.photometry[b]['mag'],
                                    yerr=thissn.photometry[b]['dmag'], fmt='.')
                        ax.plot(phases, mu, '-')
                        ax.fill_between(phases, mu - std, mu + std,
                                        alpha=.5, color='#803E75',
                                        # fc='#803E75', ec='None',
                                        label=r'$1\sigma$ C.I. ')
                        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
                        ax.set_xlabel(phasekey)
                        ax.set_ylabel(b + ' magnitude')
                        for pred in thissn.gp[b].sample_conditional(
                                thissn.gp['gpy'][b], XX, 10):
                            ax.plot(phases, pred)

                        ax = fig.add_subplot(122)
                        ax.errorbar(np.log10(thissn.photometry[b][phasekey] - \
                                             thissn.photometry[b][phasekey].min()),
                                    thissn.photometry[b]['mag'],
                                    yerr=thissn.photometry[b]['dmag'],
                                    fmt='.')
                        ax.plot(np.log10(phases - phases.min()), mu, '-')
                        ax.fill_between(np.log10(phases - phases.min()),
                                        mu - std, mu + std,
                                        alpha=.5, color='#803E75',
                                        # fc='#803E75', ec='None',
                                        label=r'$1\sigma$ C.I. ')
                        ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
                        ax.set_xlabel('log phase')
                        ax.set_ylabel(b + ' magnitude')

                        leg = ax.legend(loc='lower right', numpoints=1)
                        leg.get_frame().set_alpha(0.3)
                        # pl.show()
                        # pl.savefig("gpplots/"+thissn.name+"_"+b+".gp.png",
                        #           bbox_inches='tight')
                        if PRETTY:
                            print("outputs/gpplots/" + thissn.name + "_" + b + ".gp.pdf")
                        else:
                            print("outputs/gpplots/" + thissn.name + "_" + b + ".gp.png")
        else:
            PICKLE = True

    if PICKLE:
        print("this SN has the following photometric data")
        print(thissn.filters)  # ,  thissn.fnir,  thissn.optfiles
        for b in thissn.filters:
            print("working with band", b)
            if thissn.filters[b] > 1:
                if 1:
                    print("getting GP for ", thissn.snnameshort, b)
                    print("\n\n\n")

                    fig = pl.figure()
                    phasekey = 'phase'
                    tmp = thissn.gpphot(b, phaserange=phaserange,
                                        fig=fig, phasekey=phasekey)
                    if tmp < 0:
                        print(tmp)
                        continue
                    indx = (thissn.photometry[b][phasekey] > phaserange[0]) * \
                           (thissn.photometry[b][phasekey] < phaserange[1])
                    x = thissn.photometry[b][phasekey][indx]
                    x += 0.001 * np.random.randn(len(x))
                    y = thissn.photometry[b]['mag'][indx]
                    yerr = thissn.photometry[b]['dmag'][indx]
                    phases = np.arange(int(thissn.photometry[b][phasekey].min() + 0.5),
                                       int(thissn.photometry[b][phasekey].max() + 0.5),
                                       0.1)

                    if x.max() <= 30:
                        x = np.concatenate([x, [30]])
                        print("too few data: adding mock phase 30 days datum")
                        if x.min() <= -15:
                            # find datapoint closest to -15 days form max
                            x15 = np.where(np.abs(x + 15) == \
                                           np.abs(x + 15).min())[0]
                            # appending a mock datapoint 0.5 mag fainter than
                            # the mag at -15 days
                            y = np.concatenate([y, [y[x15[0]] + 0.5]])
                        elif (x >= 15).sum() > 1:
                            # appending a mock datapoint extrapolating from
                            # the slope at phase >15
                            slope, intercept, r_value, p_value, std_err = \
                                spstats.linregress(x[x >= 15],
                                                   y[x >= 15])
                            y = np.concatenate([y,
                                                [slope * 30. + intercept]])
                        else:
                            continue

                    try:
                        # taking log of shifted (all positive) phases
                        # that is the new independent variable
                        XX = np.log(phases - phases.min() + 1)
                    except ValueError:
                        continue

                    try:
                        # gaussian provesses prediction
                        mu, cov = thissn.gp[b].predict(y, XX)
                    except AttributeError:  # , ValueError:
                        continue
                    # print (phases)
                    gpmag[b] = mu, np.sqrt(np.diag(cov)), phases

                # except:
                #     pass#
                # pl.show()
        pkl.dump(gpmag, open('outputs/gplcvs/' + sn + \
                             "_gplcv_ebmv0.00f.pkl", "wb"))
        pkl.dump(thissn.gp, open('outputs/gplcvs/' + sn + \
                                 "_gp_ebmv0.00f.pkl", "wb"))


if __name__ == "__main__":

    inputSNe = allsne
    # pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values

    for sn in inputSNe:
        if sn.startswith("#"):
            continue
        # [30:]:
        print('\n')
        print ('####### Reading New SN #######')
        print(sn)
        # read and set up SN and look for photometry files

        print(" looking for files ")

        thissn = snstuff.mysn(sn, addlit=True)
        if len(thissn.optfiles) + len(thissn.fnir) == 0:
            continue
        # read metadata for SN
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)

        # setting date of maximum if not in metadata
        if np.isnan(thissn.Vmax) or thissn.Vmax == 0:
            # only trust selected GP results (only one really)
            if '06gi' in thissn.snnameshort:
                try:
                    print("getting max from GP maybe?")
                    thissn.gp = pkl.load(open('outputs/gplcvs/' + sn + \
                                              "_gp_ebmv0.00.pkl", "rb"))
                    if thissn.gp['maxmjd']['V'] < 2400000 and thissn.gp['maxmjd']['V'] > 50000:
                        thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                    else:
                        thissn.Vmax = thissn.gp['maxmjd']['V']

                    print("GP vmax", thissn.Vmax)
                    # if not raw_input("should we use this?").lower().startswith('y'):
                    #    continue
                except IOError:
                    continue

            if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
                continue

        print(" starting loading ")
        thissn.lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
        print("here")

        # set up photometry
        thissn.setphot()
        thissn.getphot()
        thissn.printsn()
        if np.array([n for n in thissn.filters.values()]).sum() == 0:
            continue
        thissn.setphase()
        thissn.sortlc()
        # try:
        gpit(thissn)
        # except:
        #     print('Got error!')
        #     continue
