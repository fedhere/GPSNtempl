from __future__ import print_function

import sys
import glob
import os
# ,re,numpy,math,pyfits,glob,shutil,glob
# import scipy as sp
import pickle
import inspect
import itertools
import time
import george

import scipy
from scipy import stats
import scipy.optimize as op
import numpy as np
import pylab as pl
from numpy import nanmean, nanmedian
from mpmath import polyroots
from scipy.interpolate import interp1d, splrep, splev

from matplotlib.ticker import MultipleLocator, FormatStrFormatter


try:
    os.environ['SESNPATH']
except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNPATH")+ "/maketemplates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNPATH") + "/maketemplates/utils"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from templutils import *
from utils import *
from plotutils import *
import smooth
from makePhottable import *


cmd_folder = os.getenv("UTILPATH")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)


from myastrotools import *

# from fedsastroutils import *
# from fedsastroutils.myastrotools import absmag


# from templates import *

# from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title,
#     grid, savefig, show

# checking for strings in python 2/3 compatible way
try:
    basestring
except NameError:
    basestring = str

coffset = {'U': -3.3, 'u': -3.3,
           'B': -2.3,
           'V': 0, 'g': 0,
           'R': 1.8, 'r': 1.8,
           'I': 3.5, 'i': 3.5,
           'J': 8.5, 'H': 10.1, 'K': 10.5,
           'w1': 2600. * 1.42e-3 - 8.32,
           'w2': 1928. * 1.42e-3 - 8.32,
           'm2': 2246. * 1.42e-3 - 8.32}
# ,
#           
#           'w1': -3.8, 'w2': -4.3, 'm2': -4.8}

goodU = ['sn2004aw', 'sn2005bf', 'sn2005hg', 'sn2006aj', 'sn2007gr', 'sn2009iz', 'sn2009jf']
P13U = ['sn2006jc', 'sn2007uy', 'sn2008ax', 'sn2008D', 'sn2008bo']
goodIR = ['sn2005bf', 'sn2005hg', 'sn2005kl', 'sn2005mf', 'sn2006fo', 'sn2006jc', 'sn2007c', 'sn2007gr', 'sn2007uy',
          'sn2009er', 'sn2009iz', 'sn2009jf', 'sn2006aj', 'sn2008d']
forejpar = ['sn1983V', 'sn1993J', 'sn1994I', 'sn1996cb', 'sn1998bw', 'sn1999ex', 'sn2002ap', 'sn2003dh', 'sn2003jd',
            'sn2004aw', 'sn2004gq', 'sn2004fe', 'sn2005az', 'sn2005bf', 'sn2005hg', 'sn2005kl', 'sn2006aj', 'sn2007C',
            'sn2007Y', 'sn2007gr', 'sn2007ru', 'sn2008D', 'sn2008ax', 'sn2008bo', 'sn2009bb', 'sn2009jf', 'sn2009mg',
            'sn2010as', 'sn2010bh', 'sn2011ei', 'sn2011bm', 'sn2011fu', 'sn2011hs', 'sn2013df', 'sn2013dx']

survey_dates = (53232.00000, 53597.00000, 55058.000000, 56000.)
kp1 = {'CfA3': 'CfA3-kep', 'CfA4': 'kep1'}
survey = ['fsho', 'mini', kp1, 'CfA4-kep2']


def fixsnname(anarray):
    if isinstance(anarray, (str)):
        return [anarray.lower().strip() if anarray.lower().startswith('sn')
                else 'sn' + anarray.lower().strip()]
    else:
        return np.array([a.lower().strip() if a.lower().startswith('sn')
                     else 'sn' + a.lower().strip() for a in anarray])


def readinfofromfile(key='Type', verbose=False, earliest=False):
    import pandas as pd
    if verbose:
        print("environmental variable for lib:", os.getenv("SESNPATH"))

    if os.getenv("SESNPATH") == '':
        print("must set environmental variable SESNPATH")
        sys.exit()
    input_file = pd.read_csv(os.getenv("SESNPATH") + "utils/CfA.SNIbc.BIGINFO.csv")

    if key in input_file.columns:
        snall = pd.Series(input_file[key].values, index=input_file.SNname).to_dict()
        if verbose:
            print("Returning a dictionary with SNName -> %s pairs" % key)
    else:
        print("this key %s is not in the cvs BIGINFO file!")
        return None
    return snall


def getskgpreds(ts, x, y, yerr, phases, fig=None):
    t0, t1 = ts

    if t0 == 0 or t1 == 0:
        return 1e9

    tmpphases = np.sort(np.concatenate([x[:int(len(x) / 3)],
                                        x[:int(len(x) / 3 )- 1] \
                                        + np.diff(x[:int(len(x) / 3)]) / 3]))
    if len(tmpphases) < 3:
        return 1e9
    gp1, gp2, epochs = georgegp(x, y, yerr, tmpphases, t0, t1)

    if (np.diff(gp1[:int(len(gp1) / 3)]) > 3).sum() > 0:
        return 1e9

    gp1, gp2, epochs = georgegp(x, y, yerr, x, t0, t1)

    s1 = np.abs(1.0 - sum(((gp1 - y) / yerr) ** 2) / len(y))
    if np.isnan(gp1).any():
        return 1e9

    gp1, gp2, epochs = georgegp(x, y, yerr, phases, t0, t1)
    if np.isnan(gp1).any():
        return 1e9

    s2 = sum(np.abs((gp1[2:] + gp1[:-2] - 2 * gp1[1:-1]) / \
                    (np.diff(np.exp(phases)[1:]) +
                     np.diff(np.exp(phases)[:-1]))))

    # print ("%.3f"%t0, "%.3f"%t1, "%.1f"%s1, "%.3f"%s2, s1*s2)
    if fig:
        pl.errorbar(x, y, yerr=yerr, fmt='.')
        pl.plot(phases, gp1, '-')
        pl.fill_between(phases, gp1 - gp2, gp1 + gp2, color='k')
        pl.title("%.3f %.3f %.3f" % (t0, t1, (s1 * s2)), fontsize=15)
    if np.isfinite(s1 * s2) and not np.isnan(s1 * s2):
        return s1 * np.sqrt(s2)
    return 1e9


def kernelfct(kc):
    from george.kernels import ExpSquaredKernel  # , WhiteKernel, ExpKernel, Matern32Kernel
    return ExpSquaredKernel(kc)  # Matern32Kernel(kc)


def georgegp(x, y, yerr, phases, kc, kc1):
    # Set up the Gaussian process.
    kernel = kc1 * 10 * kernelfct(kc)  # ExpSquaredKernel(1.0)
    gp = george.GP(kernel)
    # print ("wtf", gp.kernel)

    # adding  a small random offset to the phase so that i never have
    # 2 measurements at the same time which would break the GP

    # Pre-compute the factorization of the matrix.
    XX = np.log(x - x.min() + 0.1)

    # You need to compute the GP once before starting the optimization.
    try:
        gp.compute(x, yerr)
    except ValueError:
        return (np.zeros(1) * np.nan,
                np.zeros(1) * np.nan,
                np.zeros(1) * np.nan)

    # Print the initial ln-likelihood.
    # print("here", gp.lnlikelihood(y))
    # print("here", gp.grad_lnlikelihood(y))

    # Run the optimazation routine.
    # if OPT:
    #    p0 = gp.kernel.vector
    #
    #    results = op.minimize(nll, p0, jac=grad_nll, args=(gp))
    #    print results.x
    #    # Update the kernel and print the final log-likelihood.
    #    gp.kernel[:] = results.x
    # print(gp.lnlikelihood(y))

    # gp.compute(XX, yerr)

    # Compute the log likelihood.
    # print(gp.lnlikelihood(y))

    # t = np.linspace(0, 10, 500)
    ##xx = np.log(phases-min(X)+1)
    xx = np.log(phases - min(phases) + 1)
    mu, cov = gp.predict(y, xx)
    std = np.sqrt(np.diag(cov))
    return (mu, std, xx)


class sntype:
    def __init__(self, sntype):
        self.su = setupvars()
        self.type = sntype
        self.count = 0

        self.photometry = {}
        for b in self.su.bands:
            self.photometry[b] = {'mjd': np.zeros((0), float), 'mag': np.zeros((0), float),
                                  'dmag': np.zeros((0), float), 'camsys': ['']}

        self.colors = {}
        for c in self.su.cs:
            self.colors[c] = {'mjd': [], 'mag': [], 'dmag': []}
        # np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}

        self.colormeans = {}
        for c in self.su.cs:
            self.colormeans[c] = {'epoch': [], 'median': [], 'std': []}

        self.maxcol = {}
        for c in self.su.cs:
            self.maxcol[c] = {'mean': 0, 'median': 0, 'std': 0, 'n': 0}

        self.dist = -1

    def printtype(self):
        print("######### SN TYPE " + self.type + " ############")
        print("number of lcvs: %d" % self.count)
        for c in self.su.cs.keys():
            print(c + " max color (median, std, n datapoints): ",
                  self.maxcol[c]['median'], self.maxcol[c]['std'],
                  self.maxcol[c]['n'])

    def sncount(self, snlist):
        for sn in snlist:
            if sn.type == self.type:
                self.count += 1

    def addcolor(self, band, sn):
        # check type:
        if sn.type == self.type:
            self.colors[band]['mjd'] = np.concatenate( \
                (np.array(self.colors[band]['mjd']),
                 np.array(sn.colors[band]['mjd'])))
            self.colors[band]['mag'] = self.colors[band]['mag'] + \
                                       sn.colors[band]['mag']
            self.colors[band]['dmag'] = self.colors[band]['dmag'] + \
                                        sn.colors[band]['dmag']
        else:
            print("the supernova you passed is not the right type!")

    def plottype(self, photometry=False, band='', color=False, c='',
                 fig=None, show=False, verbose=False,
                 save=False, alpha=1.0):

        print("######## PLOTTING: ", self.type, " including ",
              self.count, "sn #############")

        if photometry:
            print("not implemented yet")
            return (-1)

        if color:
            if photometry:
                print("need new fig number")
            myfig = fig
            if not myfig:
                myfig = pl.figure()
            ax = myfig.add_subplot(1, 1, 1)
            legends = []
            notused = []
            if c == '':
                mybands = [k for k in self.su.cs.keys()]
            else:
                mybands = [c]
            myylim = (0, 20)
            myxlim = (-10, 10)

            for b in mybands:
                if len(self.colors[b]['mjd']) == 0:
                    if verbose:
                        print("nothing to plot for ", b)
                        notused.append(b)
                    continue
                if verbose:
                    print("plotting band ", b, " for ", self.name)
                    # print self.[b]['mjd'],self.colors[b]['mag']

                m1yxlim = (float(min(myxlim[0], min(self.colors[b]['mjd']) - 10)),
                           float(max(myxlim[1], max(self.colors[b]['mjd']) + 10)))
                myylim = (float(min(myylim[0], min(self.colors[b]['mag']) - 0.5)),
                          float(max(myylim[1], max(self.colors[b]['mag']) + 0.5)))

                myplot_setlabel(xlabel='JD - 2453000.00', ylabel='color', title=self.type)
                legends.append(myplot_err(self.colors[b]['mjd'],
                                          self.colors[b]['mag'],
                                          yerr=self.colors[b]['dmag'],
                                          xlim=myxlim, ylim=myylim, symbol='%so' % self.su.mycolors[b[0]],
                                          alpha=alpha))  #

            loc = 1
            pl.legend(legends, mybands, loc=loc, ncol=1, prop={'size': 8}, numpoints=1, framealpha=0.2)
            for i in notused:
                mybands.remove(i)
            if save:
                pl.savefig(self.type + ".color_" + ''.join(mybands) + '.png', bbox_inches='tight')
        if show:
            pl.show()

        return myfig


class snstats:
    def __init__(self):
        try:
            os.environ['SESNPATH']
        except KeyError:
            print("must set environmental variable SESNPATH")
            sys.exit()
        self.band = ''
        self.maxjd = [0.0, 0.0]
        self.m15data = [0.0, 0.0]
        self.dm15 = 0.0
        self.dm15lin = [0.0, 0.0]
        self.Rdm15 = 0.0
        self.Rdm15lin = [0.0, 0.0]
        self.polydeg = 0.0
        self.polyrchisq = 0.0
        self.polyresid = None
        self.templrchisq = 0.0
        self.templresid = None
        self.tlim = [0.0, 0.0]
        self.maglim = [0.0, 0.0]
        self.templatefit = {'stretch': 1.0, 'xoffset': 0.0, 'xstretch': 1.0, 'yoffset': 0.0}
        self.stretch = 0.0
        self.norm = 0.0

        self.flagmissmax = 0
        self.flagmiss15 = 0
        self.flagbadfit = 0
        self.success = 0

    def printstats(self):
        print("############## sn statistics: ###########")
        print("maxjd ", self.maxjd[:], "band ", self.band)
        print("m15   ", self.m15data[:], "band ", self.band)
        print("dm15  ", self.dm15, "band ", self.band)
        print("dm15 l ", self.dm15lin, "band ", self.band)
        print("deg    ", self.polydeg)

        print("poly  chisq  ", self.polyrchisq)
        if not self.polyresid == None:
            print("poly  resids ",
                  sum((self.polyresid) * (self.polyresid)))
        print("templ chisq  ", self.templrchisq)
        if not self.templresid == None:
            print("templ resids ",
                  sum((self.templresid) * (self.templresid)))
        print("########################\n\n")


class mysn:
    def __init__(self, name, lit=False, addlit=False,
                 verbose=False, quiet=False,
                 fnir=True, noload=False):
        self.snnameshort = None
        self.optfiles = []
        self.lit = lit + addlit
        self.addlit = addlit
        if fnir:
            self.fnir = []
        # print ("sn name or file: ", name)
        surveys = ['ASASSN-', 'ASASSN', 'Gaia', 'LSQ', 'DES', 'CSS1', \
                   'LSQ', 'OGLE-2013-SN-', 'OGLE', 'PSN', 'SMT', 'SCP', \
                   'PS1-', 'PS1', 'PS', 'PTF', 'iPTF', 'ESSENCE', 'GRB',
                   'MLS', 'MASTER', 'SNF', 'SNLS', 'SNhunt', 'SDSS', 'smt']
        # fix SN name
        # if a file name is passed find the SN name from it
        #  hopefully the SN name is passed when it starts with a number
        if name.startswith('0') or name.startswith('1'):
            self.snnameshort = name
            self.name = 'sn20' + name

        elif name.startswith('8') or name.startswith('9'):
            self.snnameshort = name
            self.name = 'sn19' + name
            print("name, shortname", self.name, self.snnameshort)

        elif name.startswith('sn'):
            self.snnameshort = name.replace('sn20', '').replace('sn19', '')
            self.name = name

        elif name.startswith('SN'):
            self.snnameshort = name.replace('SN20', '').replace('SN19', '')
            self.name = name.replace('SN20', 'sn20').replace('SN19', 'sn19')

        for beginnin in surveys:
            if name.startswith(beginnin):
                self.snnameshort = name
                self.name = self.snnameshort

            #        if not "/" in name:
        #            if not isinstance(name, basestring) :
        #                OneName = self.name
        #            else:
        #                OneName = self.name
        #            print ("OneName", OneName)

        if '/' in name:
            if verbose:
                print("/ in name: its a file")
            self.name = name.split('/')[-1].split('.')
            # initial set up with different name convensions.
            # sn5 for other names (e.g.. PTF)
            for s in self.name:
                s = s.lower()
                s = s.replace('snsn', 'sn')
                if 'sn9' in s:
                    self.name = s.replace('sn', 'sn19')
                    break
                elif 'sn8' in s:
                    self.name = s.replace('sn', 'sn19')
                    break
                elif 'sn7' in s:
                    self.name = s.replace('sn', 'sn19')
                    break
                elif 'sn6' in s:
                    self.name = s.replace('sn', 'sn19')
                    break
                elif 'sn5' in s:
                    self.name = s.replace('sn', 'sn20')
                    break
                elif 'sn0' in s:
                    self.name = s.replace('sn', 'sn20')
                    break
                elif 'sn1' in s and not s.startswith('sn19'):
                    self.name = s.replace('sn', 'sn20')
                    break
                elif 'sn199' in s or 'sn198' in s:
                    self.name = s
                    break
                elif 'sn20' in s:
                    self.name = s
                    break
                elif 'sn19' in s:
                    self.name = s
                    break
                elif sum(np.array([(sv.lower() in s) for sv in surveys])):
                    # "PS" in name:
                    # print ("s", s)
                    for beginnin in surveys:
                        # print  ("sn"+beginnin.lower(), self.name, s.startswith("sn"+beginnin.lower()))
                        if s.startswith(beginnin.lower()):
                            self.snnameshort = s
                            self.name = self.snnameshort
                        elif s.startswith("sn" + beginnin.lower()):
                            self.snnameshort = s.replace('-', '').replace(':', '').replace("+", "")
                            # self.snnameshort = s
                            self.name = self.snnameshort
                            # self.name = self.snnameshort

                else:
                    if verbose:
                        print("what is this??", self.name)

            if verbose:
                print("final name:", self.name)

        if not isinstance(name, basestring):
            OneName = self.name
        else:
            OneName = self.name
        # print ("OneName", OneName)

        # print ("final name:", self.name)
        assert isinstance(self.name, basestring), \
            "something went wrong in setting name"

        # set short name
        if self.snnameshort is None:
            self.snnameshort = self.name.replace('sn19', ''). \
                replace('sn20', '').strip()
            if len(self.snnameshort) == 3:
                self.snnameshort = self.snnameshort.upper()
        if not quiet:
            print("SN name short:", self.snnameshort)
            print("\n")

        if not noload:
            if verbose:
                print("loading", OneName)
            if self.findfiles(OneName, verbose=verbose,
                              quiet=quiet) == -1:
                if not quiet:
                    print("no photometry files found")

        self.su = setupvars()
        self.nomaxdate = False
        self.type = ''
        self.camsystem = ''
        self.pipeline = ''
        self.n = 0
        self.Vmax = 0.0
        self.dVmax = 0.0
        self.Vmaxmag = 0.0
        self.filters = {}
        self.gp = {}
        self.gpmax = {}
        self.gp['gpy'] = {}
        self.gp['max'] = {}
        self.gp['maxmjd'] = {}
        self.gp['maxmag'] = {}
        self.gp['result'] = {}
        for b in self.su.bands:
            self.filters[b] = 0
            self.gp[b] = None
            self.gp['result'][b] = None
            self.gp['gpy'][b] = None
            self.gp['max'][b] = None
            self.gp['maxmjd'][b] = None
            self.gp['maxmag'][b] = None
        self.polysol = {}
        self.snspline = {}
        self.templsol = {}
        self.solution = {}
        self.photometry = {}
        self.stats = {}
        self.colors = {}
        self.maxcolors = {}
        self.maxmags = {}
        self.flagmissmax = True
        self.lc = {}
        self.ebmvtot = 0.0
        self.Dl = 0.0
        for b in self.su.bands:
            self.photometry[b] = {'mjd': np.zeros((0), float),
                                  'mag': np.zeros((0), float),
                                  'dmag': np.zeros((0), float),
                                  'extmag': np.zeros((0), float),
                                  'camsys': [''],
                                  'natmag': np.zeros((0), float),
                                  'flux': np.zeros((0), float),
                                  'phase': np.zeros((0), float)
                                  }
            self.stats[b] = snstats()
            self.polysol[b] = None
            self.snspline[b] = None
            self.templsol[b] = None
            self.solution[b] = {'sol': None, 'deg': None, 'pars': None, 'resid': None}
            self.maxmags[b] = {'epoch': 0.0, 'mag': float('NaN'), 'dmag': float('NaN')}

        for c in self.su.cs:
            self.maxcolors[c] = {'epoch': 0.0, 'color': float('NaN'), 'dcolor': float('NaN')}
            self.colors[c] = {'mjd': [], 'mag': [],
                              'dmag': []}  # np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}

        self.polyfit = None
        self.metadata = {}
        self.Rmax = {}
        self.dr15 = 0.0

    def findfiles(self, OneName=None, verbose=False, quiet=False):
        # verbose = True
        if quiet: verbose = False
        if verbose:
            print("names in findfile", "NIR now:", self.fnir,
                  "self.name:", self.name)
            print("name we are looking for:", OneName)

        ##we are passing files to identify the SN
        if not OneName == self.name:
            if self.fnir:
                if 'nir' in OneName:
                    self.fnir = [OneName]
                    if not quiet:
                        print("fnir", self.fnir)
                return 1
            self.optfiles = [OneName]
            if verbose: print("optfiles", self.optfiles)
            return 1

        if not quiet:
            print("\n#######Optical & UV#######\n")

        # we passed a SN name
        optarr = glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                           self.snnameshort.upper() + ".*[cf]") + \
                 glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                           self.snnameshort.lower() + ".*[cf]")

        if len(optarr) > 0:
            self.optfiles = [optarr[0]]
            if not quiet:
                print("CfA optical file:", self.optfiles)
            # raw_input()
        else:
            if not quiet:
                print("No CfA optical files")

        if verbose:
            print("looking in literature data:", self.lit > 0)
            print("looking in literature in addition to CfA data:",
                  self.addlit)
            # print "fnir files", self.fnir
            print("looking for NIR data", not (self.fnir == False))
            print("\n")

        if self.lit:
            # find the optical photometry literaature files
            if self.addlit:
                litoptfiles = list(set(glob.glob(os.environ['SESNPATH'] + \
                                                 "literaturedata/phot/*" + \
                                                 self.snnameshort.upper() + ".[cf]") + \
                                       glob.glob(os.environ['SESNPATH'] + \
                                                 "literaturedata/phot/*" + \
                                                 self.snnameshort.lower() + ".[cf]") + \
                                       glob.glob(os.environ['SESNPATH'] + \
                                                 "literaturedata/phot/*" + \
                                                 self.snnameshort + ".[cf]")))

                self.optfiles = self.optfiles + litoptfiles
            else:
                self.optfiles = list(set(glob.glob(os.environ['SESNPATH'] + \
                                                   "literaturedata/phot/*" + \
                                                   self.snnameshort.upper() + ".[cf]") + \
                                         glob.glob(os.environ['SESNPATH'] + \
                                                   "literaturedata/phot/*" + \
                                                   self.snnameshort.lower() + ".[cf]") + \
                                         glob.glob(os.environ['SESNPATH'] + \
                                                   "literaturedata/phot/*" + \
                                                   self.snnameshort + ".[cf]")))

        if not quiet:
            print("all optical files:", self.optfiles)
            print("\n")
            print("#######NIR#######\n")
        # find NIR data
        if not (self.fnir == False):

            nirarr = glob.glob(os.environ['SESNPATH'] + \
                               "/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag//*" + \
                               self.snnameshort.upper() + '_*') + \
                     glob.glob(os.environ['SESNPATH'] + \
                               "/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag//*" + \
                               self.snnameshort.lower() + '_*')
            if len(nirarr) > 0:
                self.fnir = [nirarr[0]]
                if not quiet:
                    print("CfA NIR:", self.fnir)
            else:
                if not quiet:
                    print("No CfA NIR files")

            if self.lit:
                if self.addlit:
                    nirarr = nirarr + glob.glob(os.environ['SESNPATH'] + \
                                                "literaturedata/nirphot/*" + \
                                                self.snnameshort + '.*dat')
                    if len(nirarr) > 0:
                        self.fnir = list(set(nirarr))
                else:
                    nirarr = glob.glob(os.environ['SESNPATH'] + \
                                       "literaturedata/nirphot/*" + \
                                       self.snnameshort + '.*dat')

            if verbose:
                print("NIR file:", self.fnir)
                print("allfiles : ", self.optfiles + self.fnir)
            if len(self.optfiles) + len(self.fnir) == 0:
                return -1
            else:
                return 1

        if len(self.optfiles) == 0: return -1
        return 1

    def setVmax(self, loose=True, earliest=False, verbose=False, D11=False):
        """
        This function follows the procedure in Bianco+ 2014. Vmax, R/r max, Bmax, and/or I/i max
        are precomuted. If no peak in V, then peaks in R/r, I/i and/or B are converted to V with offsets from
        Bianco+ 2014.
        """
        self.flagmissmax = True
        if verbose:
            print("Vmax: ", self.Vmax)  # , self.flagmissmax)
        try:
            # print ('finalmaxVjd')
            # print (self.metadata['finalmaxVjd'])
            self.Vmax = float(self.metadata['finalmaxVjd'])
            if self.Vmax < 2400000:
                self.Vmax = self.Vmax + 2400000.5
            self.dVmax = float(self.metadata['finalmaxVjderr'])
            self.flagmissmax = False
        except:
            pass

        if np.isnan(self.Vmax) or self.Vmax is None or self.Vmax == 0.0:
            try:
                if verbose:
                    print('CfA VJD bootstrap')
                self.Vmax = float(self.metadata['CfA VJD bootstrap'])
                if self.Vmax < 2400000:
                    self.Vmax = self.Vmax + 2400000.5
                if verbose:
                    print("here is Vmax", self.Vmax)

                self.dVmax = float(self.metadata['CfA VJD bootstrap error'])
                self.flagmissmax = False
            except:
                pass
            if verbose:
                print("Vmax so far", self.Vmax)
            if np.isnan(self.Vmax) or self.Vmax is None or self.Vmax == 0.0:
                try:
                    self.Vmax = float(self.metadata['MaxVJD'])
                    self.dVmax = 1.5
                    self.flagmissmax = False
                except:
                    pass
                if verbose:
                    print("Vmax so far2", self.Vmax)
                if np.isnan(self.Vmax) or self.Vmax is None or self.Vmax == 0.0:
                    if D11:
                        if verbose:
                            print("trying D11 V max")
                        try:
                            self.Vmax = float(self.metadata['D11Vmaxdate'])
                            self.dVmax = float(self.metadata['D11Vmaxdateerr'])
                        except:
                            self.Vmag = None
                            self.flagmissmax = True
                    if np.isnan(self.Vmax) or self.Vmax is None or self.Vmax == 0.0:
                        # print (self.metadata)
                        if not loose:
                            self.Vmag = None
                            self.flagmissmax = True
                        else:
                            if verbose:
                                print("trying with other color max's")
                            Rmax = np.nan
                            Bnax = np.nan
                            Imax = np.nan
                            try:
                                if verbose:
                                    print("Rmax: ",
                                          self.metadata['CfA RJD bootstrap'])
                                Rmax = float(self.metadata['CfA RJD bootstrap'])
                                dRmax = float(self.metadata['CfA RJD error'])
                                if Rmax > 2400000: Rmax -= 2400000.5
                                if verbose:
                                    print("here Rmax", Rmax, dRmax)
                                if not np.isnan(Rmax):
                                    Rmaxflag = True
                                else:
                                    Rmaxflag = False
                                pass
                            except:
                                Rmaxflag = False
                                pass
                            try:
                                if verbose:
                                    print("Bmax: ",
                                          self.metadata['CfA BJD bootstrap'])
                                Bmax = float(self.metadata['CfA BJD bootstrap'])
                                dBmax = float(self.metadata['CfA BJD error'])
                                if Bmax > 2400000: Bmax -= 2400000.5
                                if verbose:
                                    print("here Bmax", Bmax, dBmax)
                                if not np.isnan(Bmax):
                                    Bmaxflag = True
                                else:
                                    Bmaxflag = False
                            except:
                                Bmaxflag = False
                                pass
                            try:
                                if verbose:
                                    print("Imax: ",
                                          self.metadata['CfA IJD bootstrap'])
                                Imax = float(self.metadata['CfA IJD bootstrap'])
                                dImax = float(self.metadata['CfA IJD error'])
                                if Imax > 2400000: Imax -= 2400000.5
                                if verbose:
                                    print("here Imax", Imax, dImax)
                                if not np.isnan(Imax):
                                    Imaxflag = True
                                else:
                                    Imaxflag = False
                            except:
                                Imaxflag = False
                                pass

                            if verbose:
                                print("Rmaxflag: ", Rmaxflag)
                            if verbose:
                                print("Bmaxflag: ", Bmaxflag)
                            if verbose:
                                print("Imaxflag: ", Imaxflag)
                            if Rmaxflag + Bmaxflag + Imaxflag >= 2:
                                if Bmaxflag and Rmaxflag:
                                    self.Vmax = np.mean([Rmax - 1.5 + 2400000.5, Bmax + 2.3 + 2400000.5])
                                    self.dVmax = np.sqrt(sum([dRmax ** 2, dBmax ** 2, 1.3 ** 2, 1.3 ** 2]))
                                elif Bmaxflag and Imaxflag:
                                    self.Vmax = np.mean([Imax - 3.1 + 2400000.5, Bmax + 2.3 + 2400000.5])
                                    self.dVmax = np.sqrt(sum([dImax ** 2, dBmax ** 2, 1.3 ** 2, 1.5 ** 2]))
                                elif Rmaxflag and Imaxflag:
                                    self.Vmax = np.mean([Imax - 3.1 + 2400000.5, Rmax - 1.5 + 2400000.5])
                                    self.dVmax = np.sqrt(sum([dRmax ** 2, dImax ** 2, 1.3 ** 2, 1.5 ** 2]))
                                self.flagmissmax = False
                            elif Rmaxflag + Bmaxflag + Imaxflag >= 1 and loose:
                                if Imaxflag:
                                    self.Vmax = Imax - 3.1 + 2400000.5
                                    self.dVmax = np.sqrt(sum([dImax ** 2, 1.5 ** 2]))
                                    if verbose:
                                        print(self.Vmax, Imax, self.dVmax)
                                if Rmaxflag:
                                    self.Vmax = Rmax - 1.5 + 2400000.5
                                    self.dVmax = np.sqrt(sum([dRmax ** 2, 1.3 ** 2]))
                                    if verbose:
                                        print(self.Vmax, Rmax, self.dVmax)
                                if Bmaxflag:
                                    self.Vmax = Bmax + 2.3 + 2400000.5
                                    self.dVmax = np.sqrt(sum([dBmax ** 2, 1.3 ** 2]))
                                    if verbose:
                                        print(self.Vmax, Bmax, self.dVmax)
                                self.flagmissmax = False
                            else:
                                if earliest:
                                    self.Vmax = earliestv
                                    self.flagmissmax = False
                                else:
                                    self.Vmag = None
                                    self.flagmissmax = True

        # print ("wtf here", self.Vmax)
        # raw_input()
        if verbose:
            print("Vmax: ", self.Vmax, self.flagmissmax)

    #         if not  self.flagmissmax

    def readinfofileall_old(self,input_file, verbose=False, earliest=False,
                            loose=False, D11=False):
        import csv
        import os
        import sys
        # if verbose:
        #     print("environmental variable for lib:", os.getenv("SESNPATH"))
        snall = {}
        # if os.getenv("SESNPATH") == '':
        #     print("must set environmental variable SESNPATH")
        #     sys.exit()
        # input_file = csv.DictReader(open(os.getenv("SESNPATH") + "/CfA.SNIbc.BIGINFO.csv"))
        # for row in input_file:
        #     snall[row['SNname']] = row['Type']
        #     if row['SNname'].lower().strip() == self.name.lower().strip():
        #         for k in row.keys():
        #             if verbose:
        #                 print(k)
        #             self.metadata[k] = row[k]
        #             if verbose:
        #                 print(self.metadata[k])

        # self.Vmaxflag = False
        #
        # self.setVmax(loose=loose, verbose=verbose, D11=D11)
        #
        # if verbose:
        #     print(self.name, self.metadata)
        # self.type = self.metadata['Type']

        # print snall
        # return snall

    def readinfofileall(self, verbose=False, earliest=False,
                        loose=False, D11=False,
                        bigfile=False, quiet=False):
        import pandas as pd
        if verbose:
            print("environmental variable for lib:", os.getenv("SESNPATH"))
        snall = {}
        if os.getenv("SESNPATH") == '':
            print("must set environmental variable SESNPATH")
            sys.exit()
        if not bigfile:
            print("reading small file")
            input_file = pd.read_csv(os.getenv("SESNPATH") + \
                                     "utils/SESNessentials.csv", encoding = "ISO-8859-1")
            # print input_file
        else:
            input_file = pd.read_csv(os.getenv("SESNPATH") + \
                                     "utils/CfA.SNIbc.BIGINFO.csv")

        # for i in range(len(input_file)):
        #     snall[input_file['SNname'][i]] = input_file['Type'][i]
        #     if input_file['SNname'][i].lower().strip() == self.name.lower().strip():
        #         for k in input_file.columns:
        #             if verbose:
        #                 print(k)
        #             self.metadata[k] = input_file[k][i]
        #             if verbose:
        #                 print(self.metadata[k])

        # print (input_file[["SNname", "finalmaxVjd"]])
        # print (input_file.head())
        # snall[row['SNname']] = row['Type']
        # print "\n\n", fixsnname(input_file['SNname']),
        # print (self.name.lower().strip())

        snn = self.setVmaxFromFile(input_file, verbose=verbose, earliest=earliest,
                                   loose=loose, D11=D11,
                                   bigfile=bigfile, quiet=quiet)
        return input_file, snn

    def setVmaxFromFile(self, input_file, verbose=False, earliest=False,
                        loose=False, D11=False, bigfile=False, quiet=False):
        try:
            snn = np.where(fixsnname(input_file['SNname']) == \
                           fixsnname(self.name.lower().strip()))

            tmp = input_file[fixsnname(input_file['SNname']) == \
                             fixsnname(self.name.lower().strip())].to_dict()
            ind = input_file.index[fixsnname(input_file['SNname']) == \
                                   fixsnname(self.name.lower().strip())].values[0]
        except:
            print (self.name + ' is not in the  list.')

        for k, v in tmp.items():
            # print (k,v)
            try:
                self.metadata[k] = v[ind]
            except:
                self.metadata[k] = np.nan
            if verbose:
                print(k, self.metadata[k])

        self.Vmaxflag = False
        if verbose:
            print("setting Vmax\n\n\n\n")
        self.setVmax(loose=loose, verbose=verbose, D11=D11)
        if verbose:
            for k, i in self.metadata.items():
                print(k, i)
        self.type = self.metadata['Type']
        if not quiet:
            print("Vmax", self.Vmax)
        return snn

    def setsn(self, sntype, Vmax, ndata=None, filters=None, camsystem=None, pipeline=None):
        self.type = sntype
        self.n = ndata
        try:
            self.Vmax = float(Vmax)
            self.flagmissmax = False
        except:
            self.Vmax = Vmax
        if camsystem:
            self.camcode = camsystem
        if pipeline:
            self.pipeline = pipeline

    def setphot(self):
        # throw away datapoints with 0 error
        # if not isinstance(self.lc, (np.ndarray)):
        #     return 0
        indx = self.lc['dmag'] == 0

        if sum(indx) > 0:
            self.lc['photcode'][indx] = np.nan

        uniqpc = set(self.lc['photcode'])
        for b in self.filters.keys():
            for i in uniqpc:
                i = i.decode('utf-8')
                if i == self.su.photcodes[b][0] or \
                        i == self.su.photcodes[b][1] or i == self.su.photcodes[b][2]:

                    n = sum(np.asarray([j.decode('utf-8') for j in (self.lc['photcode']) ]) == i )

                    self.filters[b] = n
                    self.photometry[b] = {'mjd': np.zeros(n, float),
                                          'phase': np.zeros(n, float),
                                          'mag': np.zeros(n, float),
                                          'dmag': np.zeros(n, float),
                                          'extmag': np.zeros(n, float),
                                          'natmag': np.zeros(n, float),
                                          'mag': np.zeros(n, float),
                                          'flux': np.zeros(n, float),
                                          'camsys': np.array(['S4'] * n)}

    # def setsnabsR(self):
    #     if is_empty(self.metadata):
    #         print("reading info file")
    #         self.readinfofileall(verbose=False, earliest=False, loose=True)
    #         print("done reading")
    #     # print self.filters['r'],self.filters['R']
    #     if self.filters['r'] == 0 and self.filters['R'] > 0:
    #         if not self.getmagmax('R', quiet=True) == -1:
    #             self.Rmax['mjd'], self.Rmax['mag'], self.Rmax['dmag'] = self.maxmags['R']['epoch'], self.maxmags['R'][
    #                 'mag'], self.maxmags['R']['dmag']
    #             r15 = self.getepochmags('R', epoch=(self.Rmax['mjd'] + 15.0))
    #             self.Rmax['dm15'] = self.Rmax['mag'] - r15[1]
    #             self.Rmax['ddm15'] = np.sqrt(self.Rmax['dmag'] ** 2 + r15[2] ** 2)
    #
    #     else:
    #         if not self.getmagmax('r', quiet=True) == -1:
    #             self.Rmax['mjd'], self.Rmax['mag'], self.Rmax['dmag'] = self.maxmags['r']['epoch'], self.maxmags['r'][
    #                 'mag'], self.maxmags['r']['dmag']
    #             imag = self.getepochmags('i', epoch=self.Rmax['mjd'])
    #             self.Rmax['mag'] = self.Rmax['mag'] - 0.2936 * (self.Rmax['mag'] - imag[1]) - 0.1439
    #             self.Rmax['dmag'] = np.sqrt((self.Rmax['dmag']) ** 2)  # + imag[2]**2)
    #             r15 = self.getepochmags('r', epoch=(self.Rmax['mjd'] + 15.0))
    #             i15 = self.getepochmags('i', epoch=(self.Rmax['mjd'] + 15.0))
    #             self.Rmax['dm15'] = self.Rmax['mag'] - (r15[1])  # - 0.2936*(self.Rmax['mag'] - i15[1]) - 0.1439)
    #             self.Rmax['ddm15'] = np.sqrt(self.Rmax['dmag'] ** 2)  # +(r15[2]**2+i15[2]**2))
    #     # print "Rmax",self.Rmax
    #     if not is_empty(self.Rmax):
    #         self.dist = cosmdist.cosmo_dist([0], [float(self.metadata['z'])], lum=1, Mpc=1)[0]
    #         if self.dist == -1:
    #             self.dist = float(self.metadata['distance Mpc'])
    #         self.Rmax['absmag'] = absmag(self.Rmax['mag'], self.dist, dunits='Mpc')
    #     # float(self.metadata['distance Mpc'])
    #     for k in self.Rmax.keys():
    #         print("here", k, self.Rmax[k])

    #       pl.show()

    def gpphot(self, b, phaserange=None, fig=None, ax=None,
               phasekey='phase'):
        if 'jd' in phasekey:
            phaseoffset = 0
        else:
            phaseoffset = coffset[b]
        # x = np.concatenate([[self.photometry[b]['phase'][0]-30],
        #                    [self.photometry[b]['phase'][0]-20],
        #                    self.photometry[b]['phase'],                                               `
        #                    [self.photometry[b]['phase'][-1]+200],
        #                    [self.photometry[b]['phase'][-1]+250]])

        if phaserange is None:
            phaserange = (-999, 999)

        indx = (np.array(self.photometry[b][phasekey]) > phaserange[0]) * \
               (np.array(self.photometry[b][phasekey]) < phaserange[1])
        x = self.photometry[b][phasekey][indx]
        if len(x) < 3:
            print("returning here")
            return -1
        #         print phaserange, x
        #         raw_input()
        # x = self.photometry[b]['mjd']

        x += 0.001 * np.random.randn(len(x))
        # y = np.concatenate([[self.photometry[b]['mag'].max()+3],
        #                    [self.photometry[b]['mag'].max()+3],
        #                    self.photometry[b]['mag'],
        #                    [self.photometry[b]['mag'].max()+3],
        #                    [self.photometry[b]['mag'].max()+3]])
        y = self.photometry[b]['mag'][indx]
        # yerr = np.concatenate([[3],[3],
        #                       self.photometry[b]['dmag'],
        #                       [2],[2]])
        yerr = self.photometry[b]['dmag'][indx]
        # print "x", x

        # phases = np.arange(x.min(), x.max(), 0.5)
        # print (x.min)
        # raw_input()
        phases = np.arange(-15, 100, 0.5)
        # np.arange(np.ceil(x.min())-1, 100, 0.5)
        # phases = np.arange(-10,20,0.1)
        if x.max() <= 30:
            if x.min() <= -15:
                x15 = np.where(np.abs(x + 15) == np.abs(x + 15).min())[0]
                # print x15, y[x15[0]]+0.5
                x = np.concatenate([x, [30]])
                y = np.concatenate([y, [y[x15[0]] + 0.5]])
                yerr = np.concatenate([yerr, [0.5]])
                # print (x,y,yerr)
            elif (x >= 15).sum() > 1:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[x >= 15], y[x >= 15])
                x = np.concatenate([x, [30]])
                y = np.concatenate([y, [slope * 30. + intercept]])
                yerr = np.concatenate([yerr, [yerr.max() * 2]])
                # print (x,y,yerr)
            else:
                #print("returning here2")
                self.gp['result'][b] = (np.nan, np.nan, np.nan)
                return -1

        result = op.minimize(getskgpreds, (4.0, 1.0), args=(x, y,
                                                            yerr,
                                                            phases), bounds=((3.0, None), (10, None)),
                             tol=1e-5, options={'maxiter': 2})
        kernel = result.x[1] * 10 * kernelfct(result.x[0])
        # ExpSquaredKernel(1.0)
        self.gp[b] = george.GP(kernel)
        if 'gpy' not in self.gp.keys():
            self.gp['gpy'] = {}
        self.gp['gpy'][b] = y

        XX = np.log(x - x.min() + 0.1)

        try:
            self.gp[b].compute(XX, yerr)
        except ValueError:
            print("Error: cannot compute GP")
            return -1

        try:
            epochs = np.log(phases - phases.min() + 0.1)
        except ValueError:
            print("Error: cannot set phases")
            return -1
        tmptime = np.abs(phases - x[1])
        mu, cov = self.gp[b].predict(y, epochs)
        print(len(epochs), len(mu), len(phases))

        indx = np.where(tmptime == tmptime.min())[0][0]
        if indx == 0:
            indx = indx + 1

        mu[:indx + 1] = np.poly1d(np.polyfit(x[:2],
                                             y[:2],
                                             1))(phases[:indx + 1])
        std = np.sqrt(np.diag(cov))
        self.gp['result'][b] = (phases, mu, std)
        if fig or ax:

            self.gp['max'][b] = (np.where(mu == mu.min())[0],
                                 mu.min())
            self.gp['maxmag'][b] = mu.min()

            if len(self.gp['max'][b][0]) > 1:
                if self.Vmax:
                    bmaxs = np.array([np.abs(phases[bmax] + phaseoffset) \
                                      for bmax in self.gp['max'][b][0]])
                    self.gp['max'][b] = (self.gp['max'][b][0][bmaxs == bmaxs.min()],
                                         y.min())

            self.gp['maxmjd'][b] = phases[self.gp['max'][b][0]]
        if ax:
            ax.fill_between(phases,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')

            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            # ax.set_ylim(ax.get_ylim()[1],ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(phases, pred, 'k-', lw=0.2)
            ax.plot(phases, mu, '-', color='DarkOrange')

        elif fig:
            ax = fig.add_subplot(211)
            ax.errorbar(self.photometry[b][phasekey],
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'], fmt='.')

            ax.plot(phases, mu, '-')
            ax.fill_between(phases, mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.plot([self.gp['maxmjd'][b], self.gp['maxmjd'][b]],
                    ax.get_ylim(), 'k-')
            # ax.plot([coffset[b],coffset[b]], ax.get_ylim(), 'k-',
            #        alpha = 0.5, label='GP max')
            ax.set_xlabel(phasekey)
            ax.set_ylabel(b + ' magnitude')

            ax = fig.add_subplot(212)
            ax.errorbar(np.log(self.photometry[b][phasekey] - \
                               self.photometry[b][phasekey].min() + 0.1),
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'],
                        fmt='.')
            ax.plot(epochs, mu, '-')
            ax.fill_between(epochs,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')
            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            leg = ax.legend(loc='lower right', numpoints=1)
            leg.get_frame().set_alpha(0.3)

            pl.savefig("outputs/gpplots/" + self.name + "_" + b + ".gp.png", bbox_inches='tight')

            fig = pl.figure()
            ax = fig.add_subplot(211)
            ax.errorbar(self.photometry[b][phasekey],
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'], fmt='.')

            ax.fill_between(phases, mu - std, mu + std,
                            alpha=.3, color='#803E75')
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel(phasekey)
            ax.set_ylabel(b + ' magnitude')
            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(phases, pred, 'k-', lw=0.2)

            ax = fig.add_subplot(212)
            ax.errorbar(np.log(self.photometry[b][phasekey] - \
                               self.photometry[b][phasekey].min() + 0.1),
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'],
                        fmt='.')

            ax.fill_between(epochs,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')

            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(epochs, pred, 'k-', lw=0.2)
            # pl.show()
            pl.savefig("outputs/gpplots/" + self.name + "_" + b + ".gpsample.png", bbox_inches='tight')

        return 1

    # def gpphot_skl(self, b, t0=0, phaserange=None, fig=None, ax=None,
    #                phasekey='phase'):
    #     from sklearn.gaussian_process import GaussianProcess  # Regressorx
    #
    #     X = self.photometry[b]['phase'] + \
    #         np.random.randn(self.filters[b]) * 0.01
    #     y = (self.photometry[b]['mag']).ravel()
    #     dy = self.photometry[b]['dmag']
    #     XX = np.atleast_2d(np.log(X - min(X) + 1)).T
    #     # gp = GaussianProcess
    #     # Regressor(alpha=(dy / y) ** 2,
    #     #                     n_restarts_optimizer=10)
    #     self.gp[b] = GaussianProcess(corr='squared_exponential',
    #                                  theta0=t0,
    #                                  thetaL=t0 * 0.1,
    #                                  thetaU=t0 * 10,
    #                                  nugget=(dy / y) ** 2,
    #                                  random_start=100)
    #     self.gp[b].fit(XX, y)
    #     x = np.atleast_2d(np.linspace(0, np.log(max(X) - min(X) + 1))).T
    #     if fig:
    #         # Make the prediction on the meshed x-axis (ask for MSE as well)
    #         y_pred, MSE = self.gp[b].predict(x, eval_MSE=True)
    #         sigma = np.sqrt(MSE)
    #
    #         ax = fig.add_subplot(111)
    #         ax.errorbar(np.exp(XX) + min(X) - 1, y, dy, fmt='.', color='#FFB300',
    #                     markersize=10, label=u'Observations')
    #         ax.plot(np.exp(x) + min(X) - 1, y_pred, '-', color='#803E75')
    #         ax.fill(np.concatenate([np.exp(x) + min(X) - 1,
    #                                 np.exp(x[::-1]) + min(X) - 1]),
    #                 np.concatenate([y_pred - 1.9600 * sigma,
    #                                 (y_pred + 1.9600 * sigma)[::-1]]),
    #                 alpha=.5, fc='#803E75', ec='None', \
    #                 label='95% confidence interval')
    #         ax.set_xlabel('phase')
    #         ax.set_ylabel(b + ' magnitude')
    #         leg = ax.legend(loc='lower right', numpoints=1)
    #         leg.get_frame().set_alpha(0.3)
    #
    #         ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
    #         pl.savefig(self.name + "_" + b + ".gp.png", bbox_inches='tight')
    #
    #     return x

    # def printsn(self, flux=False, AbsM=False, template=False,
    #             printlc=False, photometry=False, color=False,
    #             extended=False, band=None, cband=None,
    #             fout=None, nat=False):
    #     print("\n\n\n##############  THIS SUPERNOVA IS: ###############\n")
    #     print("name: ", self.name)
    #     print("type: ", self.type)
    #     if self.Vmax is None:
    #         print("Vmax date: None")
    #     else:
    #         print("Vmax date: %.3f" % self.Vmax)
    #     print("Vmax  mag: %.2f" % self.Vmaxmag)
    #     print("filters: ", self.filters)
    #     try:
    #         Vmax = float(self.Vmax)
    #     except:
    #         Vmax = 0.0
    #     if Vmax > 2400000.5:
    #         Vmax -= 2400000.5
    #     if band:
    #         bands = [band]
    #     else:
    #         bands = self.su.bands
    #     if cband:
    #         cbands = [cband]
    #     else:
    #         cbands = self.su.cs
    #     if printlc:
    #         print("all lightcurve: ", self.lc)
    #     if fout:
    #         f = open(fout, 'w')
    #     if photometry:
    #         print("##############  photometry by band: ###############")
    #         for b in bands:
    #             if self.filters[b] == 0:
    #                 continue
    #             order = np.argsort(self.photometry[b]['phase'])
    #
    #             if fout is None:
    #                 print("#band ", b,
    #                       "mjd\t \tphase\t \tmag \tdmag \tcamsys " +
    #                       "\t \tAbsM \tflux (Jy) \tdflux (Jy)")
    #                 for i in order:
    #                     print(b, end="")
    #                     print("\t%.3f" % self.photometry[b]['mjd'][i],
    #                           end="")
    #                     print("\t%.3f\t" % self.photometry[b]['phase'][i],
    #                           end="")
    #                     print("\t%.2f" % self.photometry[b]['mag'][i],
    #                           end="")
    #                     print("\t%.2f" % self.photometry[b]['dmag'][i],
    #                           end="")
    #                     print("\t%s" % self.photometry[b]['camsys'][i],
    #                           end="")
    #                     if AbsM:
    #                         print("\t%.2f" % self.photometry[b]['AbsMag'][i],
    #                               end="")
    #                         if flux:
    #                             print("\t%.2e" % \
    #                                   self.photometry[b]['flux'][i],
    #                                   end="")
    #                             print("\t%.2e" % \
    #                                   self.photometry[b]['dflux'][i],
    #                                   end="")
    #                     print("")
    #
    #             else:
    #
    #                 f.write("#band ", b, "mjd\t \tphase\t \tmag \tdmag")
    #                 for i in order:
    #                     f.write("\t%.3f" % self.photometry[b]['mjd'][i],
    #                             end="")
    #                     f.write("\t%.3f\t" % (self.photometry[b]['phase'][i]),
    #                             end="")
    #                     f.write("\t%.2f" % self.photometry[b]['mag'][i],
    #                             end="")
    #                     f.write("\t%.2f" % self.photometry[b]['dmag'][i],
    #                             end="")
    #                     if nat:
    #                         f.write("\t%s" % self.photometry[b]['camsys'][i],
    #                                 end="")
    #                     elif AbsM:
    #                         f.write("\t%.2f" % self.photometry[b]['AbsMag'][i],
    #                                 end="")
    #                         if flux:
    #                             f.write("\t%.2e" % \
    #                                     self.photometry[b]['flux'][i],
    #                                     end="")
    #                             f.write("\t%.2e" % \
    #                                     self.photometry[b]['dflux'][i],
    #                                     end="")
    #                     else:
    #                         f.write("")
    #
    #     if color:
    #         for c in cbands:
    #             print("\n\n\n", c)
    #             print("\n\n\ncolors : ", self.colors[c])
    #
    #             if len(self.colors[c]['mjd']) == 0:
    #                 continue
    #             if fout is None:
    #                 print("#band ", c, "mjd\t \tphase\t \tcolor \tdmag")
    #                 for i in range(len(self.colors[c]['mjd'])):
    #                     print("\t%.3f\t" % (self.colors[c]['mjd'][i] + Vmax),
    #                           end="")
    #                     print("\t%.3f" % self.colors[c]['mjd'][i],
    #                           end="")
    #                     print("\t%.2f" % self.colors[c]['mag'][i],
    #                           end="")
    #                     print("\t%.2f" % self.colors[c]['dmag'][i])
    #             else:
    #                 f.write("#band ", c, "mjd\t \tphase\t \tcolor \tdmag")
    #                 for i in range(len(self.colors[c]['mjd'])):
    #                     f.write("\t%.3f\t" % (self.colors[c]['mjd'][i] + Vmax),
    #                             end="")
    #                     f.write("\t%.3f" % self.colors[c]['mjd'][i],
    #                             end="")
    #                     f.write("\t%.2f" % self.colors[c]['mag'][i],
    #                             end="")
    #                     f.write("\t%.2f" % self.colors[c]['dmag'][i])
    #
    #     if template:
    #         for b in self.su.bands:
    #             print(b, " band: ")
    #             print("  stretch:  ", self.stats[b].templatefit['stretch'])
    #             print("  x-stretch:", self.stats[b].templatefit['xstretch'])
    #             print("  x-offset: ", self.stats[b].templatefit['xoffset'])
    #             print("  y-offset: ", self.stats[b].templatefit['yoffset'])
    #
    #     if extended:
    #         for b in self.su.bands:
    #             if self.filters[b] == 0:
    #                 continue
    #             print(b, " band: ")
    #             self.stats[b].printstats()
    #     print("\n##################################################\n\n\n")

    def printsn_fitstable(self, fout=None):
        import pyfits as pf
        print("\n\n\n##############  THIS SUPERNOVA IS: ###############\n")
        print("name: ", self.name)
        print("type: ", self.type)
        print("Vmax date: %.3f" % self.Vmax)
        print("Vmax  mag: %.2f" % self.Vmaxmag)
        print("filters: ", self.filters)
        bands = self.su.bands
        allcamsys = np.array([])
        for b in bands:
            allcamsys = np.concatente(allcamsys, self.photometry[b]['camsys'])
        allcamsys = [a for a in set(allcamsys) if not a == '']
        fitsfmt = {}
        if not fout:
            fout = self.name + ".fits"
        col = [ \
            pf.Column(name='SNname', format='8A', unit='none', array=[self.name]), \
            pf.Column(name='SNtype', format='10A', unit='none', array=[self.type]), \
            pf.Column(name='Vmaxdate', format='D', unit='MJD', array=[self.Vmax]), \
            pf.Column(name='Vmax', format='D', unit='mag', array=[self.Vmaxmag]), \
            pf.Column(name='pipeversion', format='10A', unit='none', array=allcamsys), \
            ]
        for b in bands:
            if b == 'i':
                bb = 'ip'
            elif b == 'u':
                bb = 'up'
            elif b == 'r':
                bb = 'rp'
            else:
                bb = b
            if self.filters[b] == 0:
                continue
            #            fitsfmt[b]=str(self.filters[b])+'D'
            fitsfmt[b] = 'D'
            col = col + [ \
                pf.Column(name=bb + 'pipeversion', format='10A', unit='none',
                          array=[a for a in set(self.photometry[b]['camsys'])]), \
 \
                pf.Column(name=bb + 'epochs', format=fitsfmt[b],
                          unit='MJD', array=self.photometry[b]['mjd']), \
                pf.Column(name=bb, format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['mag']), \
                pf.Column(name='d' + bb, format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['dmag']), \
                pf.Column(name=bb + '_nat', format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['natmag']), \
                ]
        '''
        col=col+[pf.Column(name='bands',          format='2A',  unit='none', array=[b for b in bands if self.filters[b] > 0])]
        '''
        # create headers
        table_hdu = pf.new_table(col)
        table_hdu.name = "TDC Challenge Light Curves"
        phdu = pf.PrimaryHDU()
        hdulist = pf.HDUList([phdu, table_hdu])

        # write to file
        hdulist.writeto(fout, clobber=True)

        print("\n##################################################\n\n\n")

    def printsn_textable(self, template=False, printlc=False, photometry=False, color=False, extended=False, band=None,
                         cband=None, fout=None):
        print("#name: ", self.name)
        print("#type: ", self.type)
        print("#Vmax date: %.3f" % self.Vmax)
        #        print "Vmax  mag: %.2f"%self.Vmaxmag
        print("#filters: ", self.filters)
        bands = self.su.bands
        if fout:
            fout = fout.replace('.tex', 'opt.tex')
            print(fout)
            fo = open(fout, 'w')
            fout = fout.replace('opt.tex', 'nir.tex')
            print(fout)
            fir = open(fout, 'w')
            print(fo, fir)
        import operator
        print("\n################################################\n")

        maxn = max(self.filters.items(), key=operator.itemgetter(1))[0]
        maxn = self.filters[maxn]
        if self.filters['u'] == 0:
            del self.filters['u']
            myu = 'U'
        elif self.filters['U'] == 0:
            del self.filters['U']
            myu = 'u\''
        if self.filters['r'] == 0:
            del self.filters['r']
            myr = 'R'
        elif self.filters['R'] == 0:
            del self.filters['R']
            myr = 'r\''
        if self.filters['i'] == 0:
            del self.filters['i']
            myi = 'I'
        elif self.filters['I'] == 0:
            del self.filters['I']
            myi = 'i\''
        if not fout is None:
            fo.write('''\\begin{deluxetable*}{ccccccccccccccc}
\\tablecolumns{15}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{''' + self.name.replace('sn', 'SN~') + '''}} Optical Photometry}''')

            fnir.write('''\\begin{deluxetable*}{ccccccccc}
\\tablecolumns{9}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{''' + self.name.replace('sn', 'SN~') + '''}} NIR Photometry}''')

        if fout is None:
            print("mjdtU\tdU\tmjd\tB\tdB\tmjd\tV\tdV\tmjd\t" + myr +
                  "\td" + myr + "\tmjd\t" + myi + "\td" + myi +
                  "\tmjd\tH\tdH\tmjd\tJ\tdJ\tmjd\tK_s\tdK_s")
        else:
            f = fo
            f.write("\\tablehead{\\colhead{MJD}&")
            f.write("\\colhead{$" + myu + "$}&")
            f.write("\\colhead{d$" + myu + "$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$B$}&")
            f.write("\\colhead{d$B$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$V$}&")
            f.write("\\colhead{d$V$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$" + myr + "$}&")
            f.write("\\colhead{d$" + myr + "$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$" + myi + "$}&")
            f.write("\\colhead{d$" + myi + "$}}")
            f.write("\\startdata")
            f = fir
            f.write("\\tablehead{\\colhead{MJD}&")
            f.write("\\colhead{$H$}&")
            f.write("\\colhead{d$H$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$J$}&")
            f.write("\\colhead{d$J$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$K_s$}&")
            f.write("\\colhead{d$K_s$}}")
            f.write("\\startdata")

        if fout:
            f = fo
        for i in range(maxn):
            for b in [myu[0], 'V', 'B', myr[0], myi[0]]:
                if i < len(self.photometry[b]['mjd']):
                    if fout is None:
                        print("%.3f\t" % self.photometry[b]['mjd'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['mag'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['dmag'][i],
                              end="")
                    else:
                        f.write("%.3f &" % self.photometry[b]['mjd'][i],
                                end="")
                        f.write("%.2f &" % self.photometry[b]['mag'][i],
                                end="")
                        if myi[0] in b:
                            f.write("%.2f\\\\ " % self.photometry[b]['dmag'][i],
                                    end="")
                        else:
                            f.write("%.2f & " % self.photometry[b]['dmag'][i],
                                    end="")

                else:
                    if fout is None:
                        print("-\t", "-\t", "-\t",
                              end="")
                    else:
                        if b.startswith(myi[0]):
                            f.write("-&", "-&", "-\\\\", end="")
                        else:
                            f.write("-&", "-&", "-&", end="")

            if fout is None:
                print("")
            else:
                f.write("")
        if fout:
            f.write('''\\enddata
\\label{tab:snoptphot}
\\end{deluxetable*}''')

        if fout:
            f = fir
        for i in range(maxn):
            for b in ['H', 'J', 'K']:

                if i < len(self.photometry[b]['mjd']):
                    if fout is None:
                        print("%.3f\t" % self.photometry[b]['mjd'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['mag'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['dmag'][i],
                              end="")
                    else:
                        f.write("%.3f &" % self.photometry[b]['mjd'][i],
                                end="")
                        f.write("%.2f &" % self.photometry[b]['mag'][i],
                                end="")
                        if 'K' in b:
                            f.write("%.2f\\\\ " % self.photometry[b]['dmag'][i],
                                    end="")
                        else:
                            f.write("%.2f & " % self.photometry[b]['dmag'][i],
                                    end="")

                else:
                    if fout is None:
                        print("-\t", "-\t", "-\t", end="")
                    else:
                        if 'K' in b:
                            f.write("-&", "-&", "-\\\\",
                                    end="")
                        else:
                            f.write("-&", "-&", "-&",
                                    end="")

            if fout is None:
                print("")
            else:
                f.write("")
        if fout:
            f.write('''\\enddata
\\label{tab:snnirphot}
\\end{deluxetable*}''')

    def colorcolorplot(self, band1='B-V', band2='r-i', fig=None, legends=[], label='', labsize=24, plotred=True):
        # b-v vs v-r
        if len(legends) == 0:
            legends = []
        myfig = fig
        if not myfig:
            myfig = pl.figure()
        print(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'])

        ax = myfig.add_subplot(1, 1, 1)

        print("color-color", self.name, self.type, self.Vmax,
              band1, self.maxcolors[band1]['color'], band2,
              self.maxcolors[band2]['color'],
              self.maxcolors[band1]['dcolor'], band2,
              self.maxcolors[band2]['dcolor'], end="")
        #        print self.su.mysymbols[self.type]
        typekey = self.type
        if typekey not in self.su.mytypecolors.keys():
            typekey = 'other'
            legends = myplot_err(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'],
                                 yerr=self.maxcolors[band1]['dcolor'],
                                 xerr=self.maxcolors[band2]['dcolor'],
                                 symbol=self.su.mytypecolors[typekey] + self.su.mysymbols[typekey],
                                 alpha=1, offset=0, fig=myfig, fcolor=self.su.mytypecolors[typekey], ms=15,
                                 markeredgewidth=2)
        else:
            legends = myplot_err(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'],
                                 yerr=self.maxcolors[band1]['dcolor'],
                                 xerr=self.maxcolors[band2]['dcolor'],
                                 symbol=self.su.mytypecolors[typekey] + self.su.mysymbols[typekey],
                                 alpha=0.5, offset=0, fig=myfig, fcolor=self.su.mytypecolors[typekey], ms=15)

        ax.annotate("", xy=(2.6, 2.6), xycoords='data',
                    xytext=(2.3, 2.3), textcoords='data', ha='center', va='center',
                    arrowprops=dict(arrowstyle="->", color='#b20000'), )
        myplot_setlabel(xlabel=band1, ylabel=band2, title=None, label=label, xy=(0.75, 0.8), labsize=labsize)
        if plotred:
            pl.figtext(0.88, 0.9, "red", fontsize=labsize)

        # pl.plot(self.maxcolors[band1]['color'],self.maxcolors[band2]['color'],c=self.su.mytypecolors[typekey],marker=self.su.mysymbols[typekey], markernsize=8, alpha=0.5)
        return (myfig, legends, (self.maxcolors[band1]['color'], self.maxcolors[band2]['color']))

    def plotsn(self, photometry=False, band='', color=False, c='',
               fig=None, ax=None, show=False,
               verbose=False, save=False, savepng=False, symbol='', title='',
               Vmax=None, plottemplate=False, plotpoly=False, plotspline=False,
               relim=True, xlim=None, ylim=None, offsets=False, ylabel='Mag',
               aspect=1, nir=False, allbands=True, fcolor=None, legendloc=1,
               nbins=None, singleplot=False, noylabel=False, ticklabelsz=16):

        su = setupvars()
        from pylab import rc
        rc('axes', linewidth=2)
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 20
        mpl.rcParams['font.family'] = 'Times New Roman'
        # mpl.rcParams['font.serif'] = 'Times'
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['xtick.labelsize'] = ticklabelsz
        mpl.rcParams['ytick.labelsize'] = ticklabelsz
        if save:
            mpl.rcParams['ytick.major.pad'] = '6'
        offset = 0.0
        boffsets = su.bandoffsets
        # 'J': -3, 'H': -4, 'K': -5, 'w1': 3, 'm2': 4, 'w2': 5}

        print("\n##############  PLOTTING SUPERNOVA : ", self.name, "###############")
        myfig = fig  # None
        if not myfig:
            myfig = pl.figure()  # , figsize=(30,60))
        if photometry:
            print("plotting...")
            if not ax:
                ax1 = myfig.add_subplot(1, 1, 1)
                adjustFigAspect(myfig, aspect=aspect)
            else:
                ax1 = ax

            ax1.minorticks_on()
            majorFormatter = FormatStrFormatter('%d')
            minorLocator = MultipleLocator(0.2)
            #            majorLocator   = MultipleLocator()
            #            ax.yaxis.set_minor_locator(minorLocator)
            ax1.yaxis.set_major_formatter(majorFormatter)

            legends = []
            notused = []
            if band == '':
                if allbands:
                    mybands = self.su.bands
                else:
                    mybands = self.su.bandsnonir
            else:
                mybands = [band]

            #            if self.stats[mybands[0]].maxjd[1]==0.0:
            #                self.getstats(mybands[0])
            scaleAdjust = int(1e-3 *
                              np.nanmin([self.photometry[mb]['mjd'].min() \
                                             if self.filters[mb] > 0 else np.nan \
                                         for mb in mybands])) * 1.0e3
            print([self.photometry[mb]['mjd'].min() \
                       if self.filters[mb] > 0 \
                       else np.nan for mb in mybands])
            print("scaleAdjust, xlim", scaleAdjust, xlim)
            xAdjust = scaleAdjust
            if scaleAdjust < 2400000:
                scaleAdjust += 2400000
                print(scaleAdjust)
            limAdjust = scaleAdjust - 2400000
            if xlim:
                myxlim = (xlim[0] - xAdjust, xlim[1] - xAdjust)
            elif not relim:
                myxlim = ax.get_xlim()
            else:

                if self.Vmax > 10e5:
                    scaleAdjust = int(float(self.Vmax) * 1e-3) * 1.0e3
                    myxlim = (float(self.Vmax) - 10 - scaleAdjust,
                              float(self.Vmax) + 10 - scaleAdjust)
                    limAdjust = scaleAdjust - 2400000.0
                elif self.Vmax <= 10e5:
                    scaleAdjust = int(float(self.Vmax) * 1e-3) * 1.0e3
                    myxlim = (float(self.Vmax) - 10 - scaleAdjust,
                              float(self.Vmax) + 10 - scaleAdjust)
                    limAdjust = scaleAdjust
                else:
                    scaleAdjust = 0
                    myxlim = (min([self.photometry[mb]['mjd'].min() \
                                       if self.filters[mb] > 0 else np.nan \
                                   for mb in mybands]),
                              max([self.photometry[mb]['mjd'].min() \
                                       if self.filters[mb] > 0 else np.nan \
                                   for mb in mybands]))
                    limAdjust = 0

                if not self.stats[mybands[0]].maxjd == [0.0, 0.0]:
                    xlim = [float(self.stats[mybands[0]].maxjd[0]) - limAdjust - 10,
                            float(self.stats[mybands[0]].maxjd[0]) + 10 - limAdjust]
                else:
                    xlim = [self.Vmax - scaleAdjust - 10.5,
                            self.Vmax - scaleAdjust + 10]

                    # print xlim
                    # print self.photometry
            if ylim:
                myylim = ylim
                if 1:  # verbose:
                    print("ylim ", myylim)
            elif not self.stats[mybands[0]].maxjd[0] == 0.0:
                myylim = (0, 0)

            else:
                bandind = 0
                while len(self.photometry[mybands[bandind]]['mjd']) == 0:
                    bandind += 1
                    if bandind > len(mybands):
                        pass
                #                print mybands[bandind],self.photometry[mybands[bandind]]['mjd']
                myxlim = (min(self.photometry[mybands[bandind]]['mjd']) - 20 - limAdjust,
                          max(self.photometry[mybands[bandind]]['mjd']) + 20 - limAdjust)
                myylim = (0, 0)

            if int(str(int(myxlim[1]))[-1]) < 5:
                myxlim = (myxlim[0], myxlim[1] + 5)

            #            majorLocator   = MultipleLocator(int((myxlim[1]-myxlim[0])/6))
            majorFormatter = FormatStrFormatter('%d')
            bandswdata = []
            # if title=='':
            #    title =self.name
            ylabel = ylabel.replace('+0', '')
            print(self.type)
            if self.name[-2].isdigit():
                label = self.name.upper()
            else:
                label = self.name
            label = label.replace('sn', 'SN ') + "\n" + self.type

            # ax.locator_params(tight=True, nbins=4)
            majorLocator = MultipleLocator(5)
            try:
                if (ylim[0] - ylim[1]) < 10:
                    majorLocator = MultipleLocator(2)
            except:
                pass
            ax1.yaxis.set_major_locator(majorLocator)

            if noylabel:
                myplot_setlabel(xlabel='JD - %.1f' % scaleAdjust, title=title,
                                label=label, ax=ax1, ylabel="  ",
                                rightticks=True, labsize=21)
            else:
                myplot_setlabel(xlabel='JD - %.1f' % scaleAdjust,
                                ylabel=ylabel, title=title,
                                label=label, ax=ax1, labsize=18)

            for b in mybands:
                if offsets:
                    offset = boffsets[b]
                else:
                    offset = 0.0
                if self.filters[b] == 0:
                    if verbose:
                        print("nothing to plot for ", b)
                        notused.append(b)
                    continue
                bandswdata.append(b.replace('r', 'r\'').replace('i', 'i\''))

                if verbose:
                    print("plotting band ", b, " for ", self.name)
                if not relim or ylim:
                    ylim = ax1.get_ylim()
                if relim:
                    xlim = [min(myxlim[0], min(self.photometry[b]['mjd']) - 10 - limAdjust),
                            max(myxlim[1], max(self.photometry[b]['mjd']) + 10 - limAdjust)]
                    myxlim = xlim
                    if myylim == (0, 0):
                        ylim = (20, 0)
                        myylim = (max(ylim[1], max(self.photometry[b]['mag']) + 1 + offset),
                                  min(ylim[0], min(self.photometry[b]['mag']) - 1 + offset))
                        if verbose:
                            print("this is the new myylim", myylim)
                    elif myylim is None:
                        myylim = (max(ylim[1], (max(myylim[0], max(self.photometry[b]['mag']) + 1 + offset))),
                                  min(ylim[0], min(myylim[1], min(self.photometry[b]['mag']) - 1 + offset)))

                if 'J' in b or 'H' in b or 'K' in b:
                    fcolor = 'None'
                else:
                    fcolor = self.su.mycolors[b]
                if symbol == '':
                    symbol = '%s%s' % (self.su.mycolors[b], self.su.myshapes[b])

                try:
                    l, = myplot_err(np.asarray(self.photometry[b]['mjd']) - xAdjust,
                                    np.asarray(self.photometry[b]['mag']),
                                    yerr=np.asarray(self.photometry[b]['dmag']),
                                    xlim=myxlim,
                                    ylim=myylim, symbol=symbol, offset=offset,
                                    fcolor=fcolor, fig=myfig, ax=ax1,
                                    litsn=np.array(self.photometry[b]['camsys']) == 'lit')
                    # print self.photometry[b]['mjd']-53000.0,
                except:
                    l, = myplot_err(np.asarray(self.photometry[b]['mjd']) - xAdjust,
                                    np.asarray(self.photometry[b]['mag']),
                                    xlim=myxlim,
                                    ylim=myylim, symbol=symbol, offset=offset,
                                    fcolor='None',
                                    # fcolor,
                                    fig=myfig, ax=ax1,
                                    litsn=np.array(self.photometry[b]['camsys']) == 'lit')

                legends.append(l)
                ax1.legend(legends, bandswdata, loc=legendloc, ncol=1,
                           prop={'size': 12}, numpoints=1, framealpha=0.5)

                symbol = ''
                if plotpoly or plottemplate:
                    if self.Vmax:
                        fullxrange = np.arange(float(self.Vmax) - scaleAdjust - 10.,
                                               float(self.Vmax) - scaleAdjust + 40.0, 0.1)
                    else:
                        try:
                            fullxrange = np.arange(self.stats['V'].maxjd[0] - 10.,
                                                   self.stats['V'].maxjd[0] + 40., 0.1)
                        except:
                            continue
                    if plotpoly and self.solution[b]['sol']:
                        myplot_err(fullxrange, self.solution[b]['sol'](fullxrange),
                                   symbol='%s-' % self.su.mycolors[b], offset=offset)
                    if plottemplate and not self.stats[b].templrchisq == 0:
                        myplot_err(fullxrange, self.templsol[b](fullxrange, [self.stats[b].templatefit['stretch'],
                                                                             self.stats[b].templatefit['xoffset'],
                                                                             self.stats[b].templatefit['yoffset'],
                                                                             self.stats[b].templatefit['xstretch']], b),
                                   symbol='%s--' % self.su.mycolors[b], offset=offset)
                if plottemplate:
                    if savepng:
                        pl.savefig(self.name + "_" + b + ".template.png", bbox_inches='tight')
                    if save:
                        thisname = self.name + "_" + b + ".template.pdf"
                        try:
                            pl.savefig(thisname)
                        except RuntimeError:
                            pl.show()
                        thisdir = os.environ['SESNPATH']
                        os.system("perl %s/pdfcrop.pl %s" % (thisdir, thisname))
                if plotspline:
                    if verbose:
                        print("plotting spline")
                    x = self.photometry[b]['mjd'].astype(np.float64)
                    #                         print "here", x, self.snspline[b]
                    fullxrange = np.arange(min(x), max(x), 0.1)
                    a = self.snspline[b](fullxrange)
                    smoothed = smooth(a, window_len=5)
                    #            smoothed=sp.signal.filter.medfilter(a,5)
                    #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                    #                    results = zip([x[0] for x in results], smoothed)
                    # print "fr",fullxrange,smoothed
                    myplot_err(fullxrange - limAdjust, smoothed, symbol='%s.' % self.su.mycolors[b], offset=offset,
                               settopx=False, fig=myfig)
                    pl.show()

            if Vmax and not self.flagmissmax:
                if self.Vmax:
                    try:
                        myplotarrow(float(self.Vmax) - scaleAdjust,
                                    min(self.photometry['V']['mag']) - 0.5,
                                    label="V max")
                    except:
                        pass
            #                else:
            #                    try:
            #                        myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
            #                    except:
            #                        pass
            print("Vmax:", self.Vmax, self.flagmissmax)
            print(self.photometry[b]['phase'])
            if verbose:
                print("Vmax:", self.Vmax, self.flagmissmax)
            ax1.tick_params('both', length=10, width=1, which='major')
            ax1.tick_params('both', length=5, width=1, which='minor')
            pl.setp(ax1.get_xticklabels(), fontsize=ticklabelsz,
                    rotation=20)

            Vmax4plot = 0
            if not self.flagmissmax:
                ax2 = ax1.twiny()
                ax2.tick_params('both', length=10, width=1, which='major')
                ax2.tick_params('both', length=5, width=1, which='minor')
                ax2.set_xlabel("phase (days)")
                pl.setp(ax2.get_xticklabels(), fontsize=ticklabelsz)
                if verbose:
                    print("putting second axis")
                Vmax4plot = self.Vmax
                if Vmax4plot > 2400000:
                    Vmax4plot -= scaleAdjust
                if Vmax4plot > 53000:
                    Vmax4plot -= limAdjust
                print(Vmax4plot)
                # print "here", (myxlim)

                ax2.set_xlim((myxlim[0] - Vmax4plot, myxlim[1] - Vmax4plot))
                if (myxlim[1] - myxlim[0]) < 100:
                    ax2.xaxis.set_major_locator(MultipleLocator(20))
                ax2.xaxis.set_minor_locator(MultipleLocator(10))

                if myxlim[0] - Vmax4plot < 0 and myxlim[1] - Vmax4plot > 0:
                    ax2.plot([0, 0], [ax1.get_ylim()[0],
                                      ax1.get_ylim()[1]], 'k-', alpha=0.3)
                    ax2.fill_between([0 - self.dVmax, 0 + self.dVmax],
                                     [ax1.get_ylim()[0], ax1.get_ylim()[0]],
                                     [ax1.get_ylim()[1], ax1.get_ylim()[1]],
                                     color='k', alpha=0.2)

            for i in notused:
                mybands.remove(i)
            if savepng:
                pl.savefig(self.name + "_" + ''.join(mybands) + '.png', bbox_inches='tight')
            if save:
                thisname = self.name + "_" + ''.join(mybands) + '.pdf'
                try:
                    pl.savefig(thisname)
                except RuntimeError:
                    self.printsn(photometry=True)
                    pl.show()

                print("running pdfcrop.pl")

                os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))

            if nir:
                legends = []
                myfig_nir = pl.figure()
                ax = myfig_nir.add_subplot(1, 1, 1)
                if band == '':
                    mybands = self.su.bandsnir
                else:
                    mybands = [band]
                if xlim:
                    myxlim = xlim - limAdjust

                elif not relim:
                    xlim = pl.xlim()
                else:
                    xlim = [float(self.stats[mybands[0]].maxjd[0]) - limAdjust - 10,
                            float(self.stats[mybands[0]].maxjd[0]) - limAdjust + 10]
                if ylim:
                    myylim = ylim
                elif not self.stats[mybands[0]].maxjd[0] == 0.0:
                    myxlim = (min(xlim[0], float(self.stats[mybands[0]].maxjd[0]) - 10 - 53000),
                              max(xlim[1], float(self.stats[mybands[0]].maxjd[0]) + 10 - 53000))
                elif self.Vmax:
                    if self.Vmax > 10e5:
                        myxlim = (float(self.Vmax) - 10 - 2400000 - limAdjust,
                                  float(self.Vmax) + 10 - 2400000 - limAdjust)
                    else:
                        myxlim = (float(self.Vmax) - 10 - limAdjust,
                                  float(self.Vmax) + 10 - limAdjust)
                else:
                    myxlim = (min(self.photometry[mybands[0]]['mjd']) - 20 - limAdjust,
                              max(self.photometry[mybands[0]]['mjd']) + 20 - limAdjust)
                bandswdata = []
                for b in mybands:
                    if offsets:
                        offset = boffsets[b]
                    else:
                        offset = 0.0
                    if verbose:
                        print("band here", b)
                    if self.filters[b] == 0:
                        if verbose:
                            print("nothing to plot for ", b)
                        notused.append(b)
                        continue
                    bandswdata.append(b)

                    if verbose:
                        print("plotting band ", b, " for ", self.name)
                    if not relim or ylim:
                        ylim = pl.ylim()
                    elif relim:
                        ylim = [float(self.stats[mybands[0]].maxjd[1]) + 10,
                                float(self.stats[mybands[0]].maxjd[1]) - 1 - 5]
                        if myylim == (0, 0):
                            myylim = (max(ylim[1], max(self.photometry[b]['mag']) + 1),
                                      min(ylim[0], min(self.photometry[b]['mag']) - 1))
                        else:
                            myylim = (max(ylim[1], (max(myylim[0], max(self.photometry[b]['mag']) + 1))),
                                      min(ylim[0], min(myylim[1], min(self.photometry[b]['mag']) - 1)))
                    if verbose:
                        print("myylim", myylim)
                    #                    if title=='':
                    #                        title =self.name
                    #                    myplot_setlabel(xlabel='JD - 2453000.00',ylabel=ylabel,title=title, ax=ax, label=label)
                    if symbol == '':
                        symbol = '%s%s' % (self.su.mycolors[b], self.su.myshapes[b])

                    legends.append(myplot_err(self.photometry[b]['mjd'] - limAdjust,
                                              self.photometry[b]['mag'],
                                              yerr=self.photometry[b]['dmag'],
                                              xlim=myxlim,
                                              ylim=myylim, symbol=symbol, offset=offset,
                                              fig=my_nir))

                    symbol = ''
                    if verbose:
                        print("vmax", self.Vmax)
                    if plotpoly or plottemplate:
                        if self.Vmax:
                            fullxrange = np.arange(float(self.Vmax) - scaleAdjust - 10.,
                                                   float(self.Vmax) - scaleAdjust + 40.0, 0.1)
                        else:
                            try:
                                fullxrange = np.arange(self.stats['V'].maxjd[0] - 10., self.stats['V'].maxjd[0] + 40.,
                                                       0.1)
                            except:
                                continue

                        if plotpoly and self.solution[b]['sol']:
                            myplot_err(fullxrange, self.solution[b]['sol'](fullxrange),
                                       symbol='%s-' % self.su.mycolors[b], offset=offset)
                        if plottemplate and not self.stats[b].templrchisq == 0:
                            myplot_err(fullxrange, self.templsol[b](fullxrange, [self.stats[b].templatefit['stretch'],
                                                                                 self.stats[b].templatefit['xoffset'],
                                                                                 self.stats[b].templatefit['yoffset'],
                                                                                 self.stats[b].templatefit['xstretch']],
                                                                    b), symbol='%s--' % self.su.mycolors[b],
                                       offset=offset)
                    if savepng:
                        pl.savefig(self.name + "_" + b + ".template.png", bbox_inches='tight')
                    if save:
                        thisname = self.name + "_" + b + ".template.pdf"
                        pl.savefig(thisname)
                        os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))
                    if plotspline:
                        if versbose:
                            print("plotting spline")
                        x = self.photometry[b]['mjd'].astype(np.float64)
                        fullxrange = np.arange(min(x), max(x), 0.1)
                        a = self.snspline[b](fullxrange)
                        smoothed = smooth(a, window_len=5)
                        #            smoothed=sp.signal.filter.medfilter(a,5)
                        #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                        #                    results = zip([x[0] for x in results], smoothed)

                        myplot_err(fullxrange, smoothed, symbol='%s-' % self.su.mycolors[b], offset=offset)
                        #                    print smoothed

                if Vmax:
                    if self.Vmax:
                        myplotarrow(float(self.Vmax) - scaleAdjust, min(self.photometry['V']['mag']) - 0.5,
                                    label="V max")
                #                    else:
                #                        try:
                #                            myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
                #                        except:
                #                            pass

                pl.legend(legends[::-1], bandswdata[::-1], loc=1, ncol=1, prop={'size': 12}, numpoints=1,
                          framealpha=0.5)
                for i in notused:
                    mybands.remove(i)
                if savepng:
                    bnds = "UBVRIriHJK.png"
                    pl.savefig(self.name + "_" + bnds, bbox_inches='tight')
                if save:
                    bnds = "UBVRIriHJK.pdf"
                    thisname = self.name + "_" + bnds
                    pl.savefig(thisname)
                    os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))

        if color:
            rc('axes', linewidth=1)
            if photometry:
                print("need new fig number")

            myfig = fig
            if not myfig:
                myfig = pl.figure()
            if not singleplot:
                ax = myfig.add_subplot(2, 1, 1)
            else:
                adjustFigAspect(myfig, aspect=2)
                ax = myfig.add_subplot(1, 1, 1)

            ax.minorticks_on()

            majorFormatter = FormatStrFormatter('%.1f')
            minorLocator = MultipleLocator(0.2)
            majorLocator = MultipleLocator(1.0)
            if '06jc' in self.name:
                majorLocator = MultipleLocator(2.0)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_major_formatter(majorFormatter)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)
                # specify integer or one of preset strings, e.g.
                # tick.label.set_fontsize('x-small')

                #            adjustFigAspect(myfig)#,aspect=aspect)
            legends = []
            notused = []
            if c == '':
                mybands = [k for k in self.su.cs.keys()]
                if mybands.index("B-i"):
                    del mybands[mybands.index("B-i")]
                # print mybands
            else:
                mybands = [c]
            myylim = (0, 0)
            myxlim = (-15, 85)
            workingbands = []
            for b in mybands:
                if len(self.colors[b]['mjd']) == 0:
                    if verbose:
                        print("nothing to plot for ", b)
                    notused.append(b)
                    continue
                if verbose:
                    print("plotting band ", b, " for ", self.name)
                    print(self.colors[b]['mjd'], self.colors[b]['mag'])

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                myylim = (float(min(myylim[0], min(self.colors[b]['mag']) - 0.5)),
                          float(max(myylim[1], max(self.colors[b]['mag']) + 0.5)))
                if self.name[-2].isdigit():
                    thename = self.name[:-1] + self.name[-1].upper()
                else:
                    thename = self.name
                myplot_setlabel(xlabel='', ylabel='color (mag)', label=thename, ax=ax, labsize=15)
                if '06jc' in self.name:
                    myxlim = (0, 85)
                    myylim = (-2, 7.5)

                l, = myplot_err(self.colors[b]['mjd'],  # )-53000.0,
                                self.colors[b]['mag'],
                                yerr=self.colors[b]['dmag'],
                                xlim=myxlim, ylim=myylim, symbol='%so' % self.su.mycolorcolors[b], offset=offset,
                                alpha=0.5)
                workingbands.append(b)
                legends.append(l)

            if '06jc' in self.name:
                loc = 2
                ncol = 4
                pl.xlim(pl.xlim()[0] - 10, pl.xlim()[1])
            else:
                loc = 1
                ncol = 1

            #            sort2vectors(v1,v2)
            pl.legend(legends, workingbands, loc=loc, ncol=ncol, prop={'size': 12}, numpoints=1, framealpha=0.2)
            if singleplot:
                ax.set_xlabel("phase (days)")
            else:
                ax2 = myfig.add_subplot(2, 1, 2, sharex=ax)
                ax2.minorticks_on()
                majorFormatter = FormatStrFormatter('%.1f')
                minorLocator = MultipleLocator(0.2)
                ax2.yaxis.set_minor_locator(minorLocator)
                ax2.yaxis.set_major_formatter(majorFormatter)

                for b in mybands:
                    if len(self.colors[b]['mjd']) == 0:
                        if verbose:
                            print("nothing to plot for ", b)
                        notused.append(b)
                        continue
                    if verbose:
                        print("plotting band ", b, " for ", self.name)
                        print(self.colors[b]['mjd'], self.colors[b]['mag'])

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                    myylim = (float(min(myylim[0], min(self.colors[b]['mag']) - 0.5)),
                              float(max(myylim[1], max(self.colors[b]['mag']) + 0.5)))

                    myplot_setlabel(xlabel='phase (days)', ylabel='color', ax=ax2)

                    if '06jc' in self.name:

                        ax2.annotate("red", xy=(-10, 2.05), xycoords='data',
                                     xytext=(-10, 1.4), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#b20000'), )
                        ax2.annotate("blue", xy=(-10, -0.65), xycoords='data',
                                     xytext=(-10, -0.2), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#0066cc'), )
                    else:
                        ax2.annotate("red", xy=(80, 2.05), xycoords='data',
                                     xytext=(80, 1.4), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#b20000'), )
                        ax2.annotate("blue", xy=(80, -0.65), xycoords='data',
                                     xytext=(80, -0.2), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#0066cc'), )
                    myplot_hist(self.colors[b]['mjd'],  # )-53000.0,
                                self.colors[b]['mag'],
                                xlim=myxlim, ylim=(-0.75, 2.2), symbol='%so' % self.su.mycolorcolors[b], offset=offset,
                                ax=ax2, nbins=nbins)  #

                    # print b,b[0],self.su.mycolors[b[0]]
                    #                legends.append(l)
            for i in notused:
                try:
                    mybands.remove(i)
                except:
                    pass
            if savepng:
                pl.savefig(self.name + "_color" + '.png', bbox_inches='tight', dpi=150)
            if save:
                thisname = self.name + "_color" + '.pdf'
                pl.savefig(thisname)
                os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))
        if show:
            pl.show()

        return myfig

    def cleanphot(self):
        self.Vmaxmag = 0.0
        self.filters = {}
        for b in self.su.bands:
            self.filters[b] = 0
        self.polysol = {}
        self.snspline = {}
        self.templsol = {}
        self.solution = {}
        self.photometry = {}
        self.stats = {}
        self.colors = {}
        self.maxcolors = {}
        self.maxmags = {}

        # self.flagmissmax = True
        #        self.lc={}
        for b in self.su.bands:
            self.photometry[b] = {'mjd': np.zeros((0), float), 'mag': np.zeros((0), float),
                                  'dmag': np.zeros((0), float), 'extmag': np.zeros((0), float),
                                  'natmag': np.zeros((0), float), 'flux': np.zeros((0), float),
                                  'phases': np.zeros((0), float), 'camsys': ['']
                                  }
            self.stats[b] = snstats()
            self.polysol[b] = None
            self.snspline[b] = None
            self.templsol[b] = None
            self.solution[b] = {'sol': None, 'deg': None, 'pars': None, 'resid': None}
            self.maxmags[b] = {'epoch': 0.0, 'mag': float('NaN'), 'dmag': float('NaN')}

        for c in self.su.cs:
            self.maxcolors[c] = {'epoch': 0.0, 'color': float('NaN'), 'dcolor': float('NaN')}
            self.colors[c] = {'mjd': [], 'mag': [],
                              'dmag': []}  # np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}

        self.polyfit = None
        self.setphot()

    # TODO: setphase() and SetAbsmag() and getphotband() and extcorrect() and getmaxcolors() and getepochcolors()
    #  getmagmax() and getmagmax_band() and getepochmax() and formatlitsn() were Commented by Somayeh because they are repeated.

    # def getphot(self, ebmv=0, RIri=False, verbose=False, quiet=False):
    #     self.snnameshort = self.name.replace('sn19', '').replace('sn20', '').strip()
    #     if verbose:
    #         print(self.snnameshort, "E(B-V)", ebmv)
    #         print(self.filters)
    #     for b in self.su.bands:
    #         print (b)
    #
    #         ##############################setting up band#########################
    #         if self.filters[b] <= 0:
    #             continue
    #         litindx = []
    #
    #         indx = np.array(np.where(self.lc['photcode'] == self.su.photcodes[b][0])[0])
    #         print(indx)
    #         if not self.su.photcodes[b][1] == self.su.photcodes[b][0]:
    #             if len(indx) == 0:
    #                 indx = np.array(np.where(self.lc['photcode'] == self.su.photcodes[b][1])[0])
    #                 print(indx)
    #             else:
    #                 phot1indx = np.where(self.lc['photcode'] == self.su.photcodes[b][1])[0]
    #                 if len(phot1indx) > 0:
    #                     newindx = np.concatenate([phot1indx, indx])
    #                     indx = newindx
    #         if not b in ['w1', 'w2', 'm2']:
    #             if sum(self.lc['photcode'] == self.su.photcodes[b][2][:2]) > 0:
    #                 litindx = np.where(self.lc['photcode'] == self.su.photcodes[b][2][:2].encode('utf-8'))[0]
    #                 newindx = np.concatenate([litindx, indx])
    #                 indx = newindx
    #
    #         print (indx, b, litindx)
    #         self.getphotband(indx, b, litindx)
    #         if verbose:
    #             print(self.snnameshort, b, self.photometry[b]['mjd'])
    #         try:
    #             self.stats[b].tlim = (min(self.photometry[b]['mjd']), max(self.photometry[b]['mjd']))
    #             self.stats[b].maglim = (min(self.photometry[b]['mag']), max(self.photometry[b]['mag']))
    #
    #         except:
    #             pass
    #         # self.filters[b] = len(self.photometry[b]['mjd'])
    #     if not ebmv == 0:
    #         self.extcorrect(ebmv, verbose=verbose)
    #     try:
    #         self.Vmax = float(self.Vmax)
    #         if self.Vmax < 2400000 and not self.Vmax == 0:
    #             self.Vmax += 2453000.5
    #     except:
    #         if self.Vmax.startswith("<0") and len(self.photometry["V"]['mjd']) > 0:
    #             self.Vmax = "<24" + str(self.photometry["V"]['mjd'][0])
    #         if self.Vmax.startswith("<"):
    #             try:
    #                 self.Vmax = self.Vmax.replace("<", "")
    #
    #                 self.Vmax = float(self.Vmax)
    #             #                    if self.Vmax=float(self.Vmax)
    #             except:
    #                 pass
    #         self.nomaxdate = True
    #
    #     if self.filters['R'] == 0 and self.filters['I'] == 0 and self.filters['r'] > 0 and self.filters[
    #         'i'] > 0 and RIri == True:
    #
    #         self.getonecolor('r-i', quiet=quiet)
    #         #             print "right here",self.filters
    #         #             self.printsn(color=True,cband='r-i')
    #         tmptimeline = np.array(self.colors['r-i']['mjd']) + self.Vmax
    #
    #         if tmptimeline[0] > 2300000:
    #             tmptimeline -= 2400000.5
    #         tmplin = interp1d(tmptimeline, self.colors['r-i']['mag'], kind='linear', bounds_error=False)
    #         tmplinerr = interp1d(tmptimeline, self.colors['r-i']['dmag'], kind='linear', bounds_error=False)
    #         self.photometry['R']['mjd'] = self.photometry['r']['mjd']
    #         self.photometry['R']['mag'] = self.photometry['r']['mag'] - 0.153 * tmplin(
    #             self.photometry['R']['mjd']) - 0.117
    #         self.photometry['R']['dmag'] = np.sqrt(
    #             self.photometry['r']['dmag'] ** 2 + tmplinerr(self.photometry['R']['mjd']) ** 2 + 0.00043 ** 2)
    #
    #         self.photometry['I']['mjd'] = self.photometry['R']['mjd']
    #         self.photometry['I']['mag'] = self.photometry['R']['mag'] - 0.930 * tmplin(
    #             self.photometry['I']['mjd']) - 0.259
    #         self.photometry['I']['dmag'] = np.sqrt(
    #             self.photometry['R']['dmag'] ** 2 + tmplinerr(self.photometry['I']['mjd']) ** 2 + 0.00055 ** 2)
    #         self.filters['R'] = self.filters['r']
    #         self.filters['I'] = self.filters['R']
    #
    #     if self.filters['r'] == 0 and self.filters['i'] == 0 and self.filters['B'] > 0 and self.filters[
    #         'V'] > 0 and RIri == True:
    #         ##Jester et al. (2005) transformations via
    #         ##https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
    #         self.getonecolor('B-V', quiet=quiet)
    #         # print self.Vmax, np.array(self.colors['B-V']['mjd'])+self.Vmax
    #         tmptimeline = np.array(self.colors['B-V']['mjd']) + self.Vmax
    #         if tmptimeline[0] > 2300000:
    #             tmptimeline -= 2400000.5
    #         tmplin = interp1d(tmptimeline, self.colors['B-V']['mag'], kind='linear', bounds_error=False)
    #         tmplinerr = interp1d(tmptimeline, self.colors['B-V']['dmag'], kind='linear', bounds_error=False)
    #         self.photometry['r']['mjd'] = self.photometry['V']['mjd']
    #         self.photometry['r']['mag'] = self.photometry['V']['mag'] - 0.42 * tmplin(
    #             self.photometry['r']['mjd']) + 0.11
    #         self.photometry['r']['dmag'] = np.sqrt(
    #             self.photometry['V']['dmag'] ** 2 + tmplinerr(self.photometry['r']['mjd']) ** 2 + 0.03 ** 2)
    #         self.filters['r'] = self.filters['V']
    #
    #         self.getonecolor('R-I', quiet=quiet)
    #         tmptimeline = np.array(self.colors['R-I']['mjd']) + self.Vmax
    #         if len(tmptimeline) > 0:
    #             if tmptimeline[0] > 2300000:
    #                 tmptimeline -= 2400000.5
    #             tmplin = interp1d(tmptimeline, self.colors['R-I']['mag'], kind='linear', bounds_error=False)
    #             tmplinerr = interp1d(tmptimeline, self.colors['R-I']['dmag'], kind='linear', bounds_error=False)
    #             self.photometry['i']['mjd'] = self.photometry['r']['mjd']
    #             self.photometry['i']['mag'] = self.photometry['r']['mag'] - 0.91 * tmplin(
    #                 self.photometry['i']['mjd']) + 0.20
    #             self.photometry['i']['dmag'] = np.sqrt(
    #                 self.photometry['r']['dmag'] ** 2 + tmplinerr(self.photometry['i']['mjd']) ** 2 + 0.03 ** 2)
    #             self.filters['i'] = self.filters['r']

    # def setphase(self, verbose=False):
    #     for b in self.su.bands:
    #         if self.filters[b] == 0:
    #             continue
    #         # print self.photometry[b]['mjd'][0], self.Vmax, self.photometry[b]['mjd'][0]<2400000, self.photometry[b]['mjd'][0]>4e4, 4e4
    #         if self.photometry[b]['mjd'][0] < 2400000 and \
    #                 self.photometry[b]['mjd'][0] > 1e4:
    #             if verbose:
    #                 print("case 1")
    #             self.photometry[b]['phase'] = self.photometry[b]['mjd'] - \
    #                                           self.Vmax + 2400000.5
    #         elif self.photometry[b]['mjd'][0] > 2400000:
    #             if verbose:
    #                 print("case 2")
    #             self.photometry[b]['phase'] = self.photometry[b]['mjd'] - \
    #                                           self.Vmax
    #         else:
    #             if verbose:
    #                 print("case 3")
    #             self.photometry[b]['phase'] = self.photometry[b]['mjd'] * \
    #                                           float('NaN')
    #         if verbose:
    #             print(self.photometry[b]['phase'][0])


    # def setAbsmag(self, Dl=None, verbose=False):
    #     if Dl:
    #         self.Dl = Dl
    #     if self.Dl == 0:
    #         print("must set Dl first")
    #         return -1
    #     for b in self.su.bands:
    #         self.photometry[b]['AbsMag'] = self.photometry[b]['mag'] - \
    #                                        5.0 * (np.log10(self.Dl) - 1)
    #
    #         self.photometry[b]['flux'] = 10 ** (23. - \
    #                                             (self.photometry[b]['AbsMag'] + 48.6) / 2.5)
    #         self.photometry[b]['dflux'] = self.photometry[b]['dmag'] * \
    #                                       self.photometry[b]['flux'] / 2.5 * np.log(10.)

    # def getphotband(self, indx, b, litindx):
    #     # if 1:
    #     try:
    #         self.photometry[b] = {'mjd': self.lc['mjd'][indx],
    #                               'phase': np.empty_like(self.lc['mjd']),
    #                               'mag': self.lc['ccmag'][indx],
    #                               'dmag': self.lc['dmag'][indx], \
    #                               'natmag': self.lc['mag'][indx],
    #                               'extmag': self.lc['mag'][indx] * np.nan,
    #                               'flux': self.lc['mag'][indx] * np.nan,
    #                               'camsys': np.array([survey[0] if i < survey_dates[0] else survey[1] \
    #                                   if i < survey_dates[1] \
    #                                   else survey[2][self.pipeline] \
    #                                   if i < survey_dates[2] else survey[3] \
    #                                                   for i in self.lc['mjd'][indx]])
    #                               }
    #         for i in range(len(indx)):
    #             if indx[i] in litindx:
    #                 self.photometry[b]['camsys'][i] = 'lit'
    #     #            print indx, litindx, self.photometry[b]['camsys']
    #     #            raw_input()
    #     # try: 1
    #     except:
    #         print("#############\n\n\n failed to get photometry \n\n\n##############")
    #         pass
            # self.printsn(photometry=True)

    #            self.photometry[b]={'mjd':self.lc['mjd'][indx],'mag':self.lc['mag'][indx],'dmag':self.lc['dmag'][indx], 'camsys':[survey[0] if i<survey_dates[0] else survey[1] if i<survey_dates[1] else survey[2][self.pipeline]  if i<survey_dates[2] else survey[3]   for i in self.lc['mjd'][indx]]}

    # def extcorrect(self, ebmv, verbose=False):
    #     if verbose:
    #         print("ebmv", ebmv)
    #     for b in self.su.bands:
    #         R = self.su.AonEBmV[b] * ebmv
    #         self.photometry[b]['mag'] -= R

    # def getmaxcolors(self, band, tol=5, quiet=False):
    #     (self.maxcolors[band]['epoch'], self.maxcolors[band]['color'],
    #      self.maxcolors[band]['dcolor']) = self.getepochcolors(band,
    #                                                            tol=tol,
    #                                                            quiet=False)
    #
    #     #
    #     print(self.name, self.maxcolors[band]['epoch'],
    #           self.maxcolors[band]['color'],
    #           self.maxcolors[band]['dcolor'])  # #

    #
    #        if len(self.colors[band]['mjd'])<=1:
    #            "print no data"
    #            return -1
    #        else:
    #            indx=np.where(abs(np.array(self.colors[band]['mjd']))==min(abs(np.array(self.colors[band]['mjd']))))[0]
    #            if len(indx)>0:
    #                indx=indx[0]###
    #
    #            if abs(self.colors[band]['mjd'][indx]) > tol:
    #                print "#########\n\n\n no data within 5 days ",self.name,self.type,"\n\n\n############"
    #                self.maxcolors[band]['epoch'] = float('NaN')
    #                self.maxcolors[band]['color'] =float('NaN')
    #                self.maxcolors[band]['dcolor'] =float('NaN')
    #            else:
    #                print "#########\n\n\n YES data within 5 days ",self.name,self.type,"\n\n\n############"
    #                self.maxcolors[band]['epoch'] = self.colors[band]['mjd'][indx]
    #                self.maxcolors[band]['color'] = self.colors[band]['mag'][indx]
    #                self.maxcolors[band]['dcolor'] =self.colors[band]['dmag'][indx]
    #            print self.maxcolors[band]['epoch'],self.maxcolors[band]['color'],self.maxcolors[band]['dcolor']

    # def getepochcolors(self, band, epoch=0.0, tol=5,
    #                    interpolate=False, quiet=False):
    #     #        print self.name,band,self.colors[band]['mjd'], self.colors[band]['mag'], tol
    #
    #     if len(self.colors[band]['mjd']) < 1:
    #         print("no data")
    #         return (float('NaN'), float('NaN'), float('NaN'))
    #     else:
    #         if not interpolate:
    #             indx, = np.where(abs(np.array(self.colors[band]['mjd']) - epoch) == min(
    #                 abs(np.array(self.colors[band]['mjd']) - epoch)))
    #             #            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
    #             #            print indx
    #             if len(indx) > 0:
    #                 indx = indx[0]
    #             if abs(self.colors[band]['mjd'][indx] - epoch) > tol:
    #                 return (float('NaN'), float('NaN'), float('NaN'))
    #             return (self.colors[band]['mjd'][indx], self.colors[band]['mag'][indx], self.colors[band]['dmag'][indx])
    #         else:
    #             ddate = self.colors[band]['mjd'] - epoch
    #             try:
    #                 indlft, = np.arange(ddate.size)[ddate == ddate[ddate < 0].max()]
    #                 indrgt, = np.arange(ddate.size)[ddate == ddate[ddate > 0].min()]
    #             except ValueError:
    #                 if not quiet:
    #                     print("the datapoints don't surround epoch, " +
    #                           "im at the edge ", epoch)
    #                 return (float('NaN'), float('NaN'), float('NaN'))
    #
    #             mag = interp1d(self.colors[band]['mjd'][indlft:indrgt + 1], self.colors[band]['mag'][indlft:indrgt + 1],
    #                            kind='linear')(epoch)
    #             err = np.sqrt(self.colors[band]['dmag'][indlft] ** 2 + self.colors[band]['dmag'][indrgt] ** 2)
    #             return (epoch, float(mag), err)


    # def getmagmax(self, band, tol=5, forceredo=False, verbose=False, quiet=False):
    #     if is_empty(self.metadata, verbose=verbose):
    #         return -1
    #     if verbose:
    #         print(self.name, 'cfa' + band.upper() + 'max',
    #               self.metadata['cfa' + band.upper() + 'max'])
    #     nomax = True
    #     if not forceredo:
    #         try:
    #             self.maxmags[band]['epoch'] = float(self.metadata['CfA ' + band.upper() + 'JD bootstrap'])
    #             self.maxmags[band]['mag'] = float(self.metadata['cfa' + band.upper() + 'max'])
    #             self.maxmags[band]['dmag'] = float(self.metadata['cfa' + band.upper() + 'maxerr'])
    #             if verbose:
    #                 print("we have max's", self.maxmags[band]['epoch'],
    #                       self.maxmags[band]['mag'],
    #                       self.maxmags[band]['dmag'])
    #             nomax = False
    #         except:
    #             if verbose:
    #                 print("no max's")
    #     if nomax or forceredo:
    #         if self.Vmax:
    #             if verbose:
    #                 print("self.Vmax:", self.Vmax)
    #             if not type(self.Vmax) == float:
    #                 pass
    #             if float(self.Vmax) > 2000000:
    #                 Vmax = float(self.Vmax) - 2400000.5
    #             if verbose:
    #                 print("Vm:", self.Vmax, Vmax)
    #             self.maxmags[band]['epoch'], \
    #             self.maxmags[band]['mag'], \
    #             self.maxmags[band]['dmag'] = self.getmagmax_band(band,
    #                                                              epoch=Vmax + coffset[band],
    #                                                              verbose=verbose, tol=tol,
    #                                                              quiet=quiet)

    # self.getepochmags(band,epoch=Vmax+coffset)))[band],tol=tol)

    #                  print self.maxmags[band]['epoch'],self.maxmags[band]['mag'],self.maxmags[band]['dmag'],"after getepochmags"

    # def getmagmax_band(self, band, epoch=None, tol=5, verbose=False, quiet=True):
    #     if not epoch:
    #         epoch = self.Vmax + coffset[band]
    #         if self.Vmax or not type(self.Vmax) == float:
    #             if verbose:
    #                 print("self.Vmax:", self.Vmax)
    #             if float(self.Vmax) > 2000000:
    #                 Vmax = float(self.Vmax) - 2400000.5
    #                 if verbose:
    #                     print("Vm in getmagmax_band:", self.Vmax, Vmax)
    #             else:
    #                 return (0, 0, 0, 0)
    #     indx, = np.where((self.photometry[band]['mjd'] < epoch + 15) & (
    #             self.photometry[band]['mjd'] > self.photometry[band]['mjd'] - 8))
    #     if verbose:
    #         print(indx, self.photometry[band]['mjd'], epoch)
    #     x = self.photometry[band]['mjd'][indx]
    #     y = self.photometry[band]['mag'][indx]
    #     e = self.photometry[band]['dmag'][indx]
    #     if verbose:
    #         print(x, y)
    #     #        pl.figure()
    #     #        pl.plot(x-self.Vmax+2400000.5,y)
    #     #        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
    #     #        try:
    #     try:
    #         nodes = splrep(x, y, w=1.0 / (self.photometry[band]['dmag'][indx]) ** 2, k=2)
    #         newx = np.arange(x[0], x[-1], 0.1)
    #         splx = splev(newx, nodes)
    #         mymax = min(splx)
    #         if verbose:
    #             print(mymax, end="")
    #         epmax = newx[np.where(splx == mymax)][0]
    #     #            print epmax
    #     #            pl.plot(newx-self.Vmax+2400000.5,splx)
    #     #            return (epmax,mymax)
    #     except:
    #         # if not quiet: print "splining to find max mag failed for band ",band
    #         return (0, 0, 0)
    #
    #     #        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
    #     #
    #     #        pl.ylim(pl.ylim()[1],pl.ylim()[0])
    #     #        pl.show()
    #     #        accept=int(raw_input("is this spline reasonable? 1 yes 0 no"))
    #     # if accept>0:
    #     #        print epmax,mymax, np.sqrt((e[np.where(x<epmax)[0][-1]])**2+(e[np.where(x>epmax)[0][0]])**2)
    #     try:
    #         return (epmax, mymax, np.sqrt((e[np.where(x < epmax)[0][-1]]) ** 2 + (e[np.where(x > epmax)[0][0]]) ** 2))
    #     except IndexError:
    #         return (epmax, mymax, e[np.where(x == epmax)[0][0]])

    #        elif accept<0:
    #             sys.exit()
    #        else:
    #             return(0,0,0)

    #        (self.maxmags[band]['epoch'], self.maxmags[band]['mag'], self.maxmags[band]['dmag'])=self.getepochmags(band, tol=tol, epoch=bandepoch)

    # def getepochmags(self, band, phase=None, epoch=None, tol=5, interpolate=False, verbose=False, plot=False,
    #                  quiet=False):
    #     if verbose:
    #         print("getting band", band, "magnitudes")
    #     if self.filters[band] == 0:
    #         if verbose:
    #             print("no data in filter ", band)
    #         return (float('NaN'), float('NaN'), float('NaN'))
    #     if not epoch:
    #         if not phase:
    #             epoch = self.Vmax - 2400000.5
    #         if phase:
    #             epoch = self.Vmax - 2400000.5 + phase
    #     myc = band.lower()
    #     if myc == 'i':
    #         myc = 'b'
    #     if myc == 'j':
    #         myc = 'm'
    #     if myc == 'h':
    #         myc = 'c'
    #     if myc == 'k':
    #         myc = 'm'
    #     if plot:
    #         if phase:
    #             pl.plot(self.photometry[band]['mjd'] - self.Vmax + 2400000.5, self.photometry[band]['mag'], '%s-' % myc)
    #             pl.errorbar(self.photometry[band]['mjd'] - self.Vmax + 2400000.5, self.photometry[band]['mag'],
    #                         self.photometry[band]['dmag'], fmt='k.')
    #         else:
    #             pl.plot(self.photometry[band]['mjd'], self.photometry[band]['mag'], '%s-' % myc)
    #             pl.errorbar(self.photometry[band]['mjd'], self.photometry[band]['mag'], self.photometry[band]['dmag'],
    #                         fmt='k.')
    #         pl.title(self.name)
    #
    #         #            pl.draw()
    #     try:
    #         ddate = abs(np.array(self.photometry[band]['mjd']) - epoch)
    #         indx, = np.where(ddate == min(ddate))
    #     except ValueError:
    #         print("no min ")
    #         indx = []
    #     #            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
    #     if verbose:
    #         print(indx, self.photometry[band]['mjd'][indx],
    #               abs(self.photometry[band]['mjd'][indx] - epoch))
    #
    #     if len(indx) > 0:
    #         indx = indx[0]
    #     #            print  abs(self.colors[band]['mjd'][indx]-epoch)
    #
    #     if min(ddate) == 0:
    #         if verbose:
    #             print("observaions at same exact epoch!")
    #         # print self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx]
    #         return (self.photometry[band]['mjd'][indx],
    #                 self.photometry[band]['mag'][indx],
    #                 self.photometry[band]['dmag'][indx])
    #
    #     if min(ddate) > tol and not interpolate:
    #         if verbose:
    #             print("nodata within ", tol, "days of ", epoch)
    #         return (float('NaN'), float('NaN'), float('NaN'))
    #     if verbose:
    #         print(self.photometry[band]['mjd'][indx],
    #               self.photometry[band]['mag'][indx],
    #               self.photometry[band]['dmag'][indx])
    #     if not interpolate:
    #         if plot:
    #             if phase:
    #                 pl.errorbar(self.photometry[band]['mjd'][indx] - \
    #                             self.Vmax + 2400000.5,
    #                             self.photometry[band]['mag'][indx],
    #                             yerr=self.photometry[band]['dmag'][indx],
    #                             fmt='r')
    #             else:
    #                 pl.errorbar(self.photometry[band]['mjd'][indx],
    #                             self.photometry[band]['mag'][indx],
    #                             yerr=self.photometry[band]['dmag'][indx],
    #                             fmt='r')
    #         return (self.photometry[band]['mjd'][indx],
    #                 self.photometry[band]['mag'][indx],
    #                 self.photometry[band]['dmag'][indx])
    #
    #     if interpolate:
    #         if verbose:
    #             print("interpolating")
    #         ddate = self.photometry[band]['mjd'] - epoch
    #         try:
    #             indlft, = np.arange(ddate.size)[ddate == ddate[ddate < 0].max()]
    #             indrgt, = np.arange(ddate.size)[ddate == ddate[ddate > 0].min()]
    #         except ValueError:
    #             if not quiet:
    #                 print(band, ": the datapoint dont surround epoch, " +
    #                       "im at the edge ", epoch, ddate)
    #                 # ,self.photometry[band]['mjd'])
    #             return (self.photometry[band]['mjd'][indx],
    #                     self.photometry[band]['mag'][indx],
    #                     self.photometry[band]['dmag'][indx])
    #
    #         mag = interp1d(self.photometry[band]['mjd'][indlft:indrgt + 1],
    #                        self.photometry[band]['mag'][indlft:indrgt + 1],
    #                        kind='linear')(epoch)
    #         err = np.sqrt(self.photometry[band]['dmag'][indlft] ** 2 +
    #                       self.photometry[band]['dmag'][indrgt] ** 2)
    #         if plot:
    #             if phase:
    #                 pl.errorbar(epoch - self.Vmax + 2400000.5, float(mag),
    #                             yerr=err, fmt='r')
    #             else:
    #                 pl.errorbar(epoch, float(mag), yerr=err, fmt='r')
    #         return (epoch, float(mag), err)

    #            return (self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx])

    # def getcolors(self, BmI=False, Bmi=False, verbose=False, quiet=False):
    #     ###setup B-I for bolometric correction as per Lyman 2014
    #     for ckey in self.su.cs.keys():
    #         ##################iterate over the color keys to get the colors cor each object
    #         ##############THIS IS LAME AND I MUST FIND A BETTER WAY TO DO IT!!###########
    #         if verbose:
    #             print(ckey)
    #         self.getonecolor(ckey, verbose=verbose, quiet=quiet)
    #         if len(self.colors['r-i']['mjd']) == 0:
    #             self.getonecolor('r-i', quiet=quiet)
    #         if BmI:
    #             if self.filters['I'] == 0 and (self.filters['i'] > 0 and \
    #                                            self.filters['r'] > 0):
    #                 if verbose:
    #                     print(self.photometry['r'], self.photometry['i'])
    #
    #                 tmpmjd = []
    #                 tmpI = []
    #                 tmpIerr = []
    #                 for k, mjd in enumerate(self.photometry['r']['mjd']):
    #                     timediff = np.abs(np.array(self.colors['r-i']['mjd']) + \
    #                                       self.Vmax - 2400000.5 - mjd)
    #                     if verbose:
    #                         print("timediff", timediff)
    #                     if min(timediff) < 1.5:
    #                         mjdind = np.where(timediff == min(timediff))[0]
    #                         if len(mjdind) > 1:
    #                             mjdind = mjdind[0]
    #                         tmpmjd.append(np.mean([self.colors['r-i']['mjd'][mjdind] + self.Vmax - 2400000.5, mjd]))
    #                         tmpI.append(
    #                             self.photometry['r']['mag'][k] - 1.2444 * (self.colors['r-i']['mag'][mjdind]) - 0.3820)
    #                         tmpIerr.append(np.sqrt(self.photometry['r']['dmag'][k] ** 2 + self.colors['r-i']['dmag'][
    #                             mjdind] ** 2 + 0.0078 ** 2))
    #                 self.photometry['I']['mjd'] = np.array(tmpmjd)
    #                 self.photometry['I']['mag'] = np.array(tmpI)
    #                 self.photometry['I']['dmag'] = np.array(tmpIerr)
    #                 self.filters['I'] = len(tmpmjd)
    #                 #        self.printsn(photometry=True)
    #                 self.getonecolor('B-I', verbose=verbose, quiet=quiet)
    #                 #        self.printsn(color=True)
    #
    #         if Bmi:
    #             if self.filters['i'] == 0 and (self.filters['I'] > 0 and self.filters['R'] > 0):
    #                 if verbose:
    #                     print(self.photometry['R'], self.photometry['I'])
    #
    #                 tmpmjd = []
    #                 tmpi = []
    #                 tmpierr = []
    #                 for k, mjd in enumerate(self.photometry['I']['mjd']):
    #                     timediff = np.abs(np.array(self.colors['R-I']['mjd']) + self.Vmax - 2400000.5 - mjd)
    #                     if verbose:
    #                         print("timediff", timediff)
    #                     if min(timediff) < 1.5:
    #                         mjdind = np.where(timediff == min(timediff))[0]
    #                         if len(mjdind) > 1:
    #                             mjdind = mjdind[0]
    #                         tmpmjd.append(np.mean([self.colors['R-I']['mjd'][mjdind] + self.Vmax - 2400000.5, mjd]))
    #                         tmpi.append(
    #                             self.photometry['I']['mag'][k] + 0.247 * self.colors['R-I']['mag'][mjdind] + 0.329)
    #
    #                         tmpierr.append(np.sqrt(self.photometry['I']['dmag'][k] ** 2 + \
    #                                                self.colors['R-I']['dmag'][mjdind] ** 2 \
    #                                                + 0.003 ** 2))
    #                 self.photometry['i']['mjd'] = np.array(tmpmjd)
    #                 self.photometry['i']['mag'] = np.array(tmpi)
    #                 self.photometry['i']['dmag'] = np.array(tmpierr)
    #                 self.filters['i'] = len(tmpmjd)
    #                 #        self.printsn(photometry=True)
    #                 self.getonecolor('B-i', verbose=verbose, quiet=quiet)
                    #        self.printsn(color=True)

    # def getonecolor(self, ckey, verbose=False, quiet=False):
    #     if verbose:
    #         print("ckey", ckey)
    #         print(self.colors[ckey])
    #     if not self.colors[ckey]['mjd'] == [] and not len(self.colors[ckey]['mjd']) == 0:
    #         if not ckey == 'r-i' and not ckey == 'B-i':
    #             if not quiet:
    #                 print("color", ckey,
    #                       "is already there. clean it first " +
    #                       "if you want me to redo it")
    #     else:
    #         if isinstance(self.colors[ckey]['mjd'], (np.ndarray, np.generic)):
    #             self.colors[ckey]['mjd'] = []
    #             self.colors[ckey]['mag'] = []
    #             self.colors[ckey]['dmag'] = []
    #         indx = np.nan
    #         for k, mjd in enumerate(self.photometry[ckey[0]]['mjd']):
    #             # check vmax:
    #             if not type(self.Vmax) == float or float(self.Vmax) < 200000:
    #                 self.Vmax = float(self.photometry[ckey[0]]['mjd'][0]) + 2453000.5
    #             mjd = float(mjd)
    #             try:
    #                 timediff = min(abs(self.photometry[ckey[2]]['mjd'] - mjd))
    #                 if verbose:
    #                     print("timediff ", timediff)
    #             except:
    #                 continue
    #             if timediff < 1.5:
    #                 indx = np.where(abs(self.photometry[ckey[2]]['mjd'] - mjd) == timediff)[0]
    #                 indx = indx[0]
    #                 if verbose:
    #                     print("mags ", mjd,
    #                           self.photometry[ckey[2]]['mag'][indx])
    #
    #         if ~np.isnan(indx):
    #             self.colors[ckey]['mjd'].append(mjd - float(self.Vmax) + 2400000.5)
    #
    #             self.colors[ckey]['mag'].append(self.photometry[ckey[0]]['mag'][k] - \
    #                                             self.photometry[ckey[2]]['mag'][indx])
    #             self.colors[ckey]['dmag'].append( \
    #                 np.sqrt(self.photometry[ckey[0]]['dmag'][k] ** 2 + \
    #                         self.photometry[ckey[2]]['dmag'][indx] ** 2))
    #         self.colors[ckey]['mjd'] = np.array(self.colors[ckey]['mjd']).flatten()
    #         self.colors[ckey]['mag'] = np.array(self.colors[ckey]['mag']).flatten()
    #         self.colors[ckey]['dmag'] = np.array(self.colors[ckey]['dmag']).flatten()
    #         if verbose:
    #             print(self.colors[ckey])

    # def savecolors(self, band=None):
    #     if band is None:
    #         mybands = [k for k in self.su.cs.keys()]
    #     else:
    #         mybands = [band]
    #
    #     for c in mybands:
    #         fout = open(self.name + "_" + c + ".dat", "w")
    #         if len(self.colors[c]['mjd']) > 0:
    #             for i, mjd in enumerate(self.colors[c]['mjd']):
    #                 fout.write(self.colors[c]['mjd'][i],
    #                            self.colors[c]['mag'][i],
    #                            self.colors[c]['dmag'][i])

    # def formatlitsn(self, lit_lc, verbose=True):
    #     nir = False
    #
    #     #        print lit_lc.dtype
    #     #        sys.exit()
    #     thissnkeys = lit_lc.dtype.names
    #     print(thissnkeys)
    #     if not 'mjd' in thissnkeys:
    #         if not 'JD' in thissnkeys:
    #             print("need 'mjd' or 'JD' in the literature lcv")
    #             return -1
    #         import numpy.lib.recfunctions as rf
    #         print(lit_lc['JD'])
    #         lit_lc = rf.append_fields(lit_lc, 'mjd', lit_lc['JD'] - 2400000.5,
    #                                   dtypes=lit_lc['JD'].dtype, usemask=False,
    #                                   asrecarray=True)
    #         # print lit_lc
    #
    #     if not self.snnameshort:
    #         self.snnameshort = self.name.replace('sn19', '') \
    #             .replace('sn20', '').strip()
    #
    #     fileout = open(os.environ['SESNPATH'] + \
    #                    "literaturedata/slc.sn" + \
    #                    self.snnameshort + '.f', 'w')
    #
    #     for b in self.su.bands:
    #         print(b, b in thissnkeys)
    #         if b == 'J' or b == 'K' or b == 'H':
    #             nir = True
    #             # print "NIR:", nir
    #             continue
    #         if not b in thissnkeys:
    #             continue
    #         if not 'd' + b in thissnkeys:
    #             print("missing mag errors, include them in literature lcv " +
    #                   "as a vectore named 'd<b>' for every filter b " +
    #                   "for which you have magnitudes (e.g. V -> dV)")
    #         if not b in self.su.photcodes.keys():
    #             continue
    #         for i, dp in enumerate(lit_lc[b]):
    #             # print dp
    #             if np.isnan(dp):
    #                 continue
    #             fileout.write(self.su.photcodes[b][2], lit_lc['mjd'][i],
    #                           'nan', 'nan', lit_lc['d' + b][i], dp)
    #     if verbose:
    #         print("file out:", fileout)
    #     print("NIR", nir)
    #     if nir:
    #         if len(self.name) == 7:
    #             fileout = open(os.environ['SESNPATH'] + "literaturedata/nirphot/" + self.name[:-1] + self.name[
    #                 -1].upper() + '.dat', 'w')
    #         else:
    #             fileout = open(os.environ['SESNPATH'] + "literaturedata/nirphot/" + self.name + '.dat', 'w')
    #
    #         for b in ['K', 'J', 'H']:
    #             if not b in thissnkeys:
    #                 continue
    #             if not 'd' + b in thissnkeys:
    #                 print("missing mag errors, include them in literature " +
    #                       "lcv as a vectore named 'd<b>' for every filter " +
    #                       "b for which you have magnitudes (e.g. V -> dV)")
    #             if not b in self.su.photcodes.keys():
    #                 continue
    #             for i, dp in enumerate(lit_lc[b]):
    #                 # print dp
    #                 if np.isnan(dp):
    #                     continue
    #                 fileout.write(b + 'l', lit_lc['mjd'][i], dp,
    #                               lit_lc['d' + b][i])
    #         if verbose:
    #             print("file out:", fileout)

    # def loadCfA3(self, f, superverbose=False):
    #     try:
    #         self.lc = np.loadtxt(f, usecols=(0, 1, 6, 7, 8,),
    #                              dtype={'names': ('photcode', 'mjd', 'mag', 'dmag',
    #                                               'ccmag'), \
    #                                     'formats': ('S2', 'f', 'f', 'f', 'f')})
    #         self.lc['photcode'] = ['%02d' % int(p) for p in self.lc['photcode']]
    #         flux = 10 ** (-self.lc['ccmag'] / 2.5) * 5e10
    #         dflux = flux * self.lc['dmag'] / LN10x2p5
    #         if superverbose:
    #             print(self.lc['mjd'])
    #             print(self.lc['ccmag'])
    #             print(self.lc['photcode'])
    #
    #     except:
    #         return None, None
    #     return flux, dflux

#TODO: Commented by Somayeh. It is repeated twice!
    # def loadCfA4(self, f, verbose=False):
    #     try:
    #         lc = np.loadtxt(f, usecols=(0, 1, 5, 3, 7), \
    #                         dtype={'names': ('photcode', 'mjd', \
    #                                          'mag', 'dmag', 'ccmag'), \
    #                                'formats': ('S2', 'f', 'f', 'f', 'f')})
    #
    #         flux = 10 ** (-self.lc['ccmag'] / 2.5) * 5e10
    #         dflux = flux * self.lc['dmag'] / LN10x2p5
    #     except:
    #         if verbose:
    #             print("trying again", f)
    #         try:
    #             lc = np.loadtxt(f, usecols=(0, 1, 5, 4, 5), \
    #                             dtype={'names': ('photcode', 'mjd', \
    #                                              'mag', 'dmag', 'ccmag'), \
    #                                    'formats': ('S2', 'f', 'f', 'f', 'f')})
    #             #                    self.lc['ccmag'] = self.lc['mag']
    #             flux = 10 ** (-lc['mag'] / 2.5) * 5e10
    #
    #             dflux = flux * lc['dmag'] / LN10x2p5
    #
    #         except:
    #             if verbose:
    #                 print("failed at loading CfA4")
    #             return None, None, None
    #     # print  lc, flux, dflux
    #     return lc, flux, dflux

    # def loadNIR(self, f, verbose=False):
    #     # loaading NIR data
    #     try:
    #         if verbose:
    #             print(f)
    #         self.nirlc = np.loadtxt(f, usecols=(0, 1, 2, 3), \
    #                                 dtype={'names': ('photcode', 'mjd',
    #                                                  'mag', 'dmag'), \
    #                                        'formats': ('S1', 'f',
    #                                                    'f', 'f')})
    #
    #         nirflux = 10 ** (-self.nirlc['mag'] / 2.5) * 5e10
    #         nirdflux = nirflux * self.nirlc['dmag'] / LN10x2p5
    #         if len(self.nirlc['mjd']) > 0:
    #             self.nir = True
    #             lc = {}
    #             lc['photcode'] = np.concatenate([self.lc['photcode'],
    #                                              self.nirlc['photcode']], axis=0)
    #             lc['mjd'] = np.concatenate([self.lc['mjd'], self.nirlc['mjd']])
    #             lc['ccmag'] = np.concatenate([self.lc['ccmag'], self.nirlc['mag']])
    #             lc['mag'] = np.concatenate([self.lc['mag'], self.nirlc['mag']])
    #             lc['dmag'] = np.concatenate([self.lc['dmag'], self.nirlc['dmag']])
    #             self.lc = lc
    #             flux = 10 ** (-lc['mag'] / 2.5) * 5e10
    #
    #             dflux = flux * lc['dmag'] / LN10x2p5
    #         else:
    #             flux, dflux = None, None
    #     except ValueError:
    #         if verbose:
    #             print("passing Value Error in loadsn nir, " +
    #                   "no nir data presusmibly")
    #         return None, None
    #
    #     return flux, dflux
#TODO: commented by Somayeh. Note that this version is different from the other version.
    # def loadlit(self, f):
    #     print("loading lit data")
    #
    #     lc, flux, dflux = self.loadCfA4(f, verbose=True)
    #     newlc = {}
    #     newlc['photcode'] = np.concatenate([self.lc['photcode'], lc['photcode']], axis=0)
    #     newlc['mjd'] = np.concatenate([self.lc['mjd'], lc['mjd']])
    #     newlc['ccmag'] = np.concatenate([self.lc['ccmag'], lc['mag']])
    #     newlc['mag'] = np.concatenate([self.lc['mag'], lc['mag']])
    #     newlc['dmag'] = np.concatenate([self.lc['dmag'], lc['dmag']])
    #
    #     uniqpc = set(self.lc['photcode'])
    #     # print uniqpc
    #     for b in self.filters.keys():
    #         for i in uniqpc:
    #             if i == self.su.photcodes[b][0] or \
    #                     i == self.su.photcodes[b][1] or i == self.su.photcodes[b][2]:
    #                 n = sum(self.lc['photcode'] == i)
    #                 self.filters[b] = n
    #                 self.photometry[b] = {'mjd': np.zeros(n, float),
    #                                       'phase': np.zeros(n, float),
    #                                       'mag': np.zeros(n, float),
    #                                       'dmag': np.zeros(n, float),
    #                                       'extmag': np.zeros(n, float),
    #                                       'natmag': np.zeros(n, float),
    #                                       'mag': np.zeros(n, float),
    #                                       'flux': np.zeros(n, float),
    #                                       'camsys': np.array(['S4'] * n)}

    def setsnabsR(self):

        if is_empty(self.metadata):
            print("reading info file")
            self.readinfofileall(verbose=False, earliest=False, loose=True)
            print("done reading")
        # print self.filters['r'],self.filters['R']
        if self.filters['r'] == 0 and self.filters['R'] > 0:
            if not self.getmagmax('R', quiet=True) == -1:
                self.Rmax['mjd'], self.Rmax['mag'], self.Rmax['dmag'] = self.maxmags['R']['epoch'], self.maxmags['R'][
                    'mag'], self.maxmags['R']['dmag']
                r15 = self.getepochmags('R', epoch=(self.Rmax['mjd'] + 15.0))
                self.Rmax['dm15'] = self.Rmax['mag'] - r15[1]
                self.Rmax['ddm15'] = np.sqrt(self.Rmax['dmag'] ** 2 + r15[2] ** 2)

        else:
            if not self.getmagmax('r', quiet=True) == -1:
                self.Rmax['mjd'], self.Rmax['mag'], self.Rmax['dmag'] = self.maxmags['r']['epoch'], self.maxmags['r'][
                    'mag'], self.maxmags['r']['dmag']
                imag = self.getepochmags('i', epoch=self.Rmax['mjd'])
                self.Rmax['mag'] = self.Rmax['mag'] - 0.2936 * (self.Rmax['mag'] - imag[1]) - 0.1439
                self.Rmax['dmag'] = np.sqrt((self.Rmax['dmag']) ** 2)  # + imag[2]**2)
                r15 = self.getepochmags('r', epoch=(self.Rmax['mjd'] + 15.0))
                i15 = self.getepochmags('i', epoch=(self.Rmax['mjd'] + 15.0))
                self.Rmax['dm15'] = self.Rmax['mag'] - (r15[1])  # - 0.2936*(self.Rmax['mag'] - i15[1]) - 0.1439)
                self.Rmax['ddm15'] = np.sqrt(self.Rmax['dmag'] ** 2)  # +(r15[2]**2+i15[2]**2))
        # print "Rmax",self.Rmax
        if not is_empty(self.Rmax):
            self.dist = cosmo_dist([0], [float(self.metadata['z'])], lum=1, Mpc=1)[0]
            if self.dist == -1:
                self.dist = float(self.metadata['distance Mpc'])
            self.Rmax['absmag'] = absmag(self.Rmax['mag'], self.dist, dunits='Mpc')
        # float(self.metadata['distance Mpc'])
        for k in self.Rmax.keys():
            print("here", k, self.Rmax[k])

    #       pl.show()

    def gpphot3(self, b, phaserange=None, fig=None, ax=None,
                phasekey='phase'):
        print("here")
        if 'jd' in phasekey:
            phaseoffset = 0
        else:
            phaseoffset = coffset[b]
        # x = np.concatenate([[self.photometry[b]['phase'][0]-30],
        #                    [self.photometry[b]['phase'][0]-20],
        #                    self.photometry[b]['phase'],
        #                    [self.photometry[b]['phase'][-1]+200],
        #                    [self.photometry[b]['phase'][-1]+250]])

        if phaserange is None:
            phaserange = (-999, 999)
        # print phasekey, self.photometry[b][phasekey]
        indx = (np.array(self.photometry[b][phasekey]) > phaserange[0]) * \
               (np.array(self.photometry[b][phasekey]) < phaserange[1])
        x = self.photometry[b][phasekey][indx]
        print(self.photometry[b]['phase'], self.photometry[b][phasekey], indx, x)
        if len(x) < 3:
            self.gp['result'][b] = (np.nan, np.nan, np.nan)
            print("here3")
            return -1
        #         print phaserange, x
        #         raw_input()
        # x = self.photometry[b]['mjd']

        x += 0.001 * np.random.randn(len(x))
        # y = np.concatenate([[self.photometry[b]['mag'].max()+3],
        #                    [self.photometry[b]['mag'].max()+3],
        #                    self.photometry[b]['mag'],
        #                    [self.photometry[b]['mag'].max()+3],
        #                    [self.photometry[b]['mag'].max()+3]])
        y = self.photometry[b]['mag'][indx]
        # yerr = np.concatenate([[3],[3],
        #                       self.photometry[b]['dmag'],
        #                       [2],[2]])
        yerr = self.photometry[b]['dmag'][indx]
        # print "x", x

        phases = np.arange(x.min(), x.max(), 0.1)
        # phases = np.arange(-10,20,0.1)
        if x.max() <= 30:
            if x.min() <= -15:
                x15 = np.where(np.abs(x + 15) == np.abs(x + 15).min())[0]
                # print x15, y[x15[0]]+0.5
                x = np.concatenate([x, [30]])
                y = np.concatenate([y, [y[x15[0]] + 0.5]])
                yerr = np.concatenate([yerr, [0.5]])
                # print (x,y,yerr)
            elif (x >= 15).sum() > 1:
                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x[x >= 15], y[x >= 15])
                x = np.concatenate([x, [30]])
                y = np.concatenate([y, [slope * 30. + intercept]])
                yerr = np.concatenate([yerr, [yerr.max() * 2]])
                # print (x,y,yerr)
            else:
                print("here4")
                return -1

        result = op.minimize(getskgpreds, (4.0, 1.0), args=(x, y,
                                                            yerr,
                                                            phases), bounds=((3.0, None), (10, None)),
                             tol=1e-5)
        kernel = result.x[1] * 10 * kernelfct(result.x[0])
        # ExpSquaredKernel(1.0)
        self.gp[b] = george.GP(kernel)
        if 'gpy' not in self.gp.keys():
            self.gp['gpy'] = {}
        self.gp['gpy'][b] = y

        XX = np.log(x - x.min() + 0.1)

        try:
            self.gp[b].compute(XX, yerr)
        except ValueError:
            print("Error: cannot compute GP")
            return -1

        try:
            epochs = np.log(phases - phases.min() + 0.1)
        except ValueError:
            print("Error: cannot set phases")
            return -1

        tmptime = np.abs(phases - x[1])

        if fig or ax:
            mu, cov = self.gp[b].predict(y, epochs)
            indx = np.where(tmptime == tmptime.min())[0][0]
            if indx == 0:
                indx = indx + 1

            mu[:indx + 1] = np.poly1d(np.polyfit(x[:2],
                                                 y[:2],
                                                 1))(phases[:indx + 1])
            std = np.sqrt(np.diag(cov))

            self.gp['max'][b] = (np.where(mu == mu.min())[0],
                                 mu.min())
            self.gp['maxmag'][b] = mu.min()

            if len(self.gp['max'][b][0]) > 1:
                if self.Vmax:
                    bmaxs = np.array([np.abs(phases[bmax] + phaseoffset) \
                                      for bmax in self.gp['max'][b][0]])
                    self.gp['max'][b] = (self.gp['max'][b][0][bmaxs == bmaxs.min()],
                                         y.min())

            self.gp['maxmjd'][b] = phases[self.gp['max'][b][0]]
        if ax:
            ax.fill_between(phases,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')

            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            # ax.set_ylim(ax.get_ylim()[1],ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(phases, pred, 'k-', lw=0.2)
            ax.plot(phases, mu, '-', color='DarkOrange')

        elif fig:
            ax = fig.add_subplot(211)
            ax.errorbar(self.photometry[b][phasekey],
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'], fmt='.')

            ax.plot(phases, mu, '-')
            ax.fill_between(phases, mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.plot([self.gp['maxmjd'][b], self.gp['maxmjd'][b]],
                    ax.get_ylim(), 'k-')
            # ax.plot([coffset[b],coffset[b]], ax.get_ylim(), 'k-',
            #        alpha = 0.5, label='GP max')
            ax.set_xlabel(phasekey)
            ax.set_ylabel(b + ' magnitude')

            ax = fig.add_subplot(212)
            ax.errorbar(np.log(self.photometry[b][phasekey] - \
                               self.photometry[b][phasekey].min() + 0.1),
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'],
                        fmt='.')
            ax.plot(epochs, mu, '-')
            ax.fill_between(epochs,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')
            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            leg = ax.legend(loc='lower right', numpoints=1)
            leg.get_frame().set_alpha(0.3)

            pl.savefig("outputs/gpplots/" + self.name + "_" + b + ".gp.png", bbox_inches='tight')

            fig = pl.figure()
            ax = fig.add_subplot(211)
            ax.errorbar(self.photometry[b][phasekey],
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'], fmt='.')

            ax.fill_between(phases, mu - std, mu + std,
                            alpha=.3, color='#803E75')
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel(phasekey)
            ax.set_ylabel(b + ' magnitude')
            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(phases, pred, 'k-', lw=0.2)

            ax = fig.add_subplot(212)
            ax.errorbar(np.log(self.photometry[b][phasekey] - \
                               self.photometry[b][phasekey].min() + 0.1),
                        self.photometry[b]['mag'],
                        yerr=self.photometry[b]['dmag'],
                        fmt='.')

            ax.fill_between(epochs,
                            mu - std, mu + std,
                            alpha=.5, color='#803E75',
                            # fc='#803E75', ec='None',
                            label=r'$1\sigma$ C.I. ')

            tmp = np.log(self.gp['maxmjd'][b] - \
                         self.photometry[b][phasekey].min() + 0.1)
            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            ax.set_xlabel('log days from first dp')
            ax.set_ylabel(b + ' magnitude')
            ax.plot([tmp, tmp], ax.get_ylim(), 'k-', label='GP max')
            ax.plot([np.log(phaseoffset - self.photometry[b][phasekey].min() + 0.1)] * 2,
                    ax.get_ylim(), 'k-', alpha=0.5, label='filter max')

            for pred in self.gp[b].sample_conditional(y, epochs, 30):
                ax.plot(epochs, pred, 'k-', lw=0.2)
            # pl.show()
            pl.savefig("outputs/gpplots/" + self.name + "_" + b + ".gpsample.png", bbox_inches='tight')

        return 1

    def gpphot_skl(self, b, t0=0, phaserange=None, fig=None, ax=None,
                   phasekey='phase'):
        from sklearn.gaussian_process import GaussianProcess  # Regressorx

        X = self.photometry[b]['phase'] + \
            np.random.randn(self.filters[b]) * 0.01
        y = (self.photometry[b]['mag']).ravel()
        dy = self.photometry[b]['dmag']
        XX = np.atleast_2d(np.log(X - min(X) + 1)).T
        # gp = GaussianProcess
        # Regressor(alpha=(dy / y) ** 2,
        #                     n_restarts_optimizer=10)
        self.gp[b] = GaussianProcess(corr='squared_exponential',
                                     theta0=t0,
                                     thetaL=t0 * 0.1,
                                     thetaU=t0 * 10,
                                     nugget=(dy / y) ** 2,
                                     random_start=100)
        self.gp[b].fit(XX, y)
        x = np.atleast_2d(np.linspace(0, np.log(max(X) - min(X) + 1))).T
        if fig:
            # Make the prediction on the meshed x-axis (ask for MSE as well)
            y_pred, MSE = self.gp[b].predict(x, eval_MSE=True)
            sigma = np.sqrt(MSE)

            ax = fig.add_subplot(111)
            ax.errorbar(np.exp(XX) + min(X) - 1, y, dy, fmt='.', color='#FFB300',
                        markersize=10, label=u'Observations')
            ax.plot(np.exp(x) + min(X) - 1, y_pred, '-', color='#803E75')
            ax.fill(np.concatenate([np.exp(x) + min(X) - 1,
                                    np.exp(x[::-1]) + min(X) - 1]),
                    np.concatenate([y_pred - 1.9600 * sigma,
                                    (y_pred + 1.9600 * sigma)[::-1]]),
                    alpha=.5, fc='#803E75', ec='None', \
                    label='95% confidence interval')
            ax.set_xlabel('phase')
            ax.set_ylabel(b + ' magnitude')
            leg = ax.legend(loc='lower right', numpoints=1)
            leg.get_frame().set_alpha(0.3)

            ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
            pl.savefig(self.name + "_" + b + ".gp.png", bbox_inches='tight')

        return x

    def printsn(self, flux=False, AbsM=False, template=False,
                printlc=False, photometry=False, color=False,
                extended=False, band=None, cband=None,
                fout=None, nat=False):
        print("\n\n\n##############  THIS SUPERNOVA IS: ###############\n")
        print("name: ", self.name)
        print("type: ", self.type)
        if self.Vmax is None:
            print("Vmax date: None")
        else:
            print("Vmax date: %.3f" % self.Vmax)
        print("Vmax  mag: %.2f" % self.Vmaxmag)
        print("filters: ", self.filters)
        try:
            Vmax = float(self.Vmax)
        except:
            Vmax = 0.0
        if Vmax > 2400000.5:
            Vmax -= 2400000.5
        if band:
            bands = [band]
        else:
            bands = self.su.bands
        if cband:
            cbands = [cband]
        else:
            cbands = self.su.cs
        if printlc:
            print("all lightcurve: ", self.lc)
        if fout:
            f = open(fout, 'w')
        if photometry:
            print("##############  photometry by band: ###############")
            for b in bands:
                if self.filters[b] == 0:
                    continue
                order = np.argsort(self.photometry[b]['phase'])

                if fout is None:
                    print("#band ", b,
                          "mjd\t \tphase\t \tmag \tdmag \tcamsys " +
                          "\t \tAbsM \tflux (Jy) \tdflux (Jy)")
                    for i in order:
                        print(b, end="")
                        print("\t%.3f" % self.photometry[b]['mjd'][i],
                              end="")
                        print("\t%.3f\t" % self.photometry[b]['phase'][i],
                              end="")
                        print("\t%.2f" % self.photometry[b]['mag'][i],
                              end="")
                        print("\t%.2f" % self.photometry[b]['dmag'][i],
                              end="")
                        print("\t%s" % self.photometry[b]['camsys'][i],
                              end="")
                        if AbsM:
                            print("\t%.2f" % self.photometry[b]['AbsMag'][i],
                                  end="")
                            if flux:
                                print("\t%.2e" % \
                                      self.photometry[b]['flux'][i],
                                      end="")
                                print("\t%.2e" % \
                                      self.photometry[b]['dflux'][i],
                                      end="")
                        print("")

                else:

                    f.write("#band ", b, "mjd\t \tphase\t \tmag \tdmag")
                    for i in order:
                        f.write("\t%.3f" % self.photometry[b]['mjd'][i],
                                end="")
                        f.write("\t%.3f\t" % (self.photometry[b]['phase'][i]),
                                end="")
                        f.write("\t%.2f" % self.photometry[b]['mag'][i],
                                end="")
                        f.write("\t%.2f" % self.photometry[b]['dmag'][i],
                                end="")
                        if nat:
                            f.write("\t%s" % self.photometry[b]['camsys'][i],
                                    end="")
                        elif AbsM:
                            f.write("\t%.2f" % self.photometry[b]['AbsMag'][i],
                                    end="")
                            if flux:
                                f.write("\t%.2e" % \
                                        self.photometry[b]['flux'][i],
                                        end="")
                                f.write("\t%.2e" % \
                                        self.photometry[b]['dflux'][i],
                                        end="")
                        else:
                            f.write("")

        if color:
            for c in cbands:
                print("\n\n\n", c)
                print("\n\n\ncolors : ", self.colors[c])

                if len(self.colors[c]['mjd']) == 0:
                    continue
                if fout is None:
                    print("#band ", c, "mjd\t \tphase\t \tcolor \tdmag")
                    for i in range(len(self.colors[c]['mjd'])):
                        print("\t%.3f\t" % (self.colors[c]['mjd'][i] + Vmax),
                              end="")
                        print("\t%.3f" % self.colors[c]['mjd'][i],
                              end="")
                        print("\t%.2f" % self.colors[c]['mag'][i],
                              end="")
                        print("\t%.2f" % self.colors[c]['dmag'][i])
                else:
                    f.write("#band ", c, "mjd\t \tphase\t \tcolor \tdmag")
                    for i in range(len(self.colors[c]['mjd'])):
                        f.write("\t%.3f\t" % (self.colors[c]['mjd'][i] + Vmax),
                                end="")
                        f.write("\t%.3f" % self.colors[c]['mjd'][i],
                                end="")
                        f.write("\t%.2f" % self.colors[c]['mag'][i],
                                end="")
                        f.write("\t%.2f" % self.colors[c]['dmag'][i])

        if template:
            for b in self.su.bands:
                print(b, " band: ")
                print("  stretch:  ", self.stats[b].templatefit['stretch'])
                print("  x-stretch:", self.stats[b].templatefit['xstretch'])
                print("  x-offset: ", self.stats[b].templatefit['xoffset'])
                print("  y-offset: ", self.stats[b].templatefit['yoffset'])

        if extended:
            for b in self.su.bands:
                if self.filters[b] == 0:
                    continue
                print(b, " band: ")
                self.stats[b].printstats()
        print("\n##################################################\n\n\n")

    def printsn_fitstable(self, fout=None):
        import pyfits as pf
        print("\n\n\n##############  THIS SUPERNOVA IS: ###############\n")
        print("name: ", self.name)
        print("type: ", self.type)
        print("Vmax date: %.3f" % self.Vmax)
        print("Vmax  mag: %.2f" % self.Vmaxmag)
        print("filters: ", self.filters)
        bands = self.su.bands
        allcamsys = np.array([])
        for b in bands:
            allcamsys = np.concatente(allcamsys, self.photometry[b]['camsys'])
        allcamsys = [a for a in set(allcamsys) if not a == '']
        fitsfmt = {}
        if not fout:
            fout = self.name + ".fits"
        col = [ \
            pf.Column(name='SNname', format='8A', unit='none', array=[self.name]), \
            pf.Column(name='SNtype', format='10A', unit='none', array=[self.type]), \
            pf.Column(name='Vmaxdate', format='D', unit='MJD', array=[self.Vmax]), \
            pf.Column(name='Vmax', format='D', unit='mag', array=[self.Vmaxmag]), \
            pf.Column(name='pipeversion', format='10A', unit='none', array=allcamsys), \
            ]
        for b in bands:
            if b == 'i':
                bb = 'ip'
            elif b == 'u':
                bb = 'up'
            elif b == 'r':
                bb = 'rp'
            else:
                bb = b
            if self.filters[b] == 0:
                continue
            #            fitsfmt[b]=str(self.filters[b])+'D'
            fitsfmt[b] = 'D'
            col = col + [ \
                pf.Column(name=bb + 'pipeversion', format='10A', unit='none',
                          array=[a for a in set(self.photometry[b]['camsys'])]), \
 \
                pf.Column(name=bb + 'epochs', format=fitsfmt[b],
                          unit='MJD', array=self.photometry[b]['mjd']), \
                pf.Column(name=bb, format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['mag']), \
                pf.Column(name='d' + bb, format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['dmag']), \
                pf.Column(name=bb + '_nat', format=fitsfmt[b],
                          unit='mag', array=self.photometry[b]['natmag']), \
                ]
        '''
        col=col+[pf.Column(name='bands',          format='2A',  unit='none', array=[b for b in bands if self.filters[b] > 0])]
        '''
        # create headers
        table_hdu = pf.new_table(col)
        table_hdu.name = "TDC Challenge Light Curves"
        phdu = pf.PrimaryHDU()
        hdulist = pf.HDUList([phdu, table_hdu])

        # write to file
        hdulist.writeto(fout, clobber=True)

        print("\n##################################################\n\n\n")

    def printsn_textable(self, template=False, printlc=False, photometry=False, color=False, extended=False, band=None,
                         cband=None, fout=None):
        print("#name: ", self.name)
        print("#type: ", self.type)
        print("#Vmax date: %.3f" % self.Vmax)
        #        print "Vmax  mag: %.2f"%self.Vmaxmag
        print("#filters: ", self.filters)
        bands = self.su.bands
        if fout:
            fout = fout.replace('.tex', 'opt.tex')
            print(fout)
            fo = open(fout, 'w')
            fout = fout.replace('opt.tex', 'nir.tex')
            print(fout)
            fir = open(fout, 'w')
            print(fo, fir)
        import operator
        print("\n################################################\n")

        maxn = max(self.filters.items(), key=operator.itemgetter(1))[0]
        maxn = self.filters[maxn]
        if self.filters['u'] == 0:
            del self.filters['u']
            myu = 'U'
        elif self.filters['U'] == 0:
            del self.filters['U']
            myu = 'u\''
        if self.filters['r'] == 0:
            del self.filters['r']
            myr = 'R'
        elif self.filters['R'] == 0:
            del self.filters['R']
            myr = 'r\''
        if self.filters['i'] == 0:
            del self.filters['i']
            myi = 'I'
        elif self.filters['I'] == 0:
            del self.filters['I']
            myi = 'i\''
        if not fout is None:
            fo.write('''\\begin{deluxetable*}{ccccccccccccccc}
\\tablecolumns{15}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{''' + self.name.replace('sn', 'SN~') + '''}} Optical Photometry}''')

            fnir.write('''\\begin{deluxetable*}{ccccccccc}
\\tablecolumns{9}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{''' + self.name.replace('sn', 'SN~') + '''}} NIR Photometry}''')

        if fout is None:
            print("mjdtU\tdU\tmjd\tB\tdB\tmjd\tV\tdV\tmjd\t" + myr +
                  "\td" + myr + "\tmjd\t" + myi + "\td" + myi +
                  "\tmjd\tH\tdH\tmjd\tJ\tdJ\tmjd\tK_s\tdK_s")
        else:
            f = fo
            f.write("\\tablehead{\\colhead{MJD}&")
            f.write("\\colhead{$" + myu + "$}&")
            f.write("\\colhead{d$" + myu + "$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$B$}&")
            f.write("\\colhead{d$B$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$V$}&")
            f.write("\\colhead{d$V$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$" + myr + "$}&")
            f.write("\\colhead{d$" + myr + "$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$" + myi + "$}&")
            f.write("\\colhead{d$" + myi + "$}}")
            f.write("\\startdata")
            f = fir
            f.write("\\tablehead{\\colhead{MJD}&")
            f.write("\\colhead{$H$}&")
            f.write("\\colhead{d$H$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$J$}&")
            f.write("\\colhead{d$J$}&")

            f.write("\\colhead{MJD}&")
            f.write("\\colhead{$K_s$}&")
            f.write("\\colhead{d$K_s$}}")
            f.write("\\startdata")

        if fout:
            f = fo
        for i in range(maxn):
            for b in [myu[0], 'V', 'B', myr[0], myi[0]]:
                if i < len(self.photometry[b]['mjd']):
                    if fout is None:
                        print("%.3f\t" % self.photometry[b]['mjd'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['mag'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['dmag'][i],
                              end="")
                    else:
                        f.write("%.3f &" % self.photometry[b]['mjd'][i],
                                end="")
                        f.write("%.2f &" % self.photometry[b]['mag'][i],
                                end="")
                        if myi[0] in b:
                            f.write("%.2f\\\\ " % self.photometry[b]['dmag'][i],
                                    end="")
                        else:
                            f.write("%.2f & " % self.photometry[b]['dmag'][i],
                                    end="")

                else:
                    if fout is None:
                        print("-\t", "-\t", "-\t",
                              end="")
                    else:
                        if b.startswith(myi[0]):
                            f.write("-&", "-&", "-\\\\", end="")
                        else:
                            f.write("-&", "-&", "-&", end="")

            if fout is None:
                print("")
            else:
                f.write("")
        if fout:
            f.write('''\\enddata
\\label{tab:snoptphot}
\\end{deluxetable*}''')

        if fout:
            f = fir
        for i in range(maxn):
            for b in ['H', 'J', 'K']:

                if i < len(self.photometry[b]['mjd']):
                    if fout is None:
                        print("%.3f\t" % self.photometry[b]['mjd'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['mag'][i],
                              end="")
                        print("%.2f\t" % self.photometry[b]['dmag'][i],
                              end="")
                    else:
                        f.write("%.3f &" % self.photometry[b]['mjd'][i],
                                end="")
                        f.write("%.2f &" % self.photometry[b]['mag'][i],
                                end="")
                        if 'K' in b:
                            f.write("%.2f\\\\ " % self.photometry[b]['dmag'][i],
                                    end="")
                        else:
                            f.write("%.2f & " % self.photometry[b]['dmag'][i],
                                    end="")

                else:
                    if fout is None:
                        print("-\t", "-\t", "-\t", end="")
                    else:
                        if 'K' in b:
                            f.write("-&", "-&", "-\\\\",
                                    end="")
                        else:
                            f.write("-&", "-&", "-&",
                                    end="")

            if fout is None:
                print("")
            else:
                f.write("")
        if fout:
            f.write('''\\enddata
\\label{tab:snnirphot}
\\end{deluxetable*}''')

    def colorcolorplot(self, band1='B-V', band2='r-i', fig=None, legends=[], label='', labsize=24, plotred=True):
        # b-v vs v-r
        if len(legends) == 0:
            legends = []
        myfig = fig
        if not myfig:
            myfig = pl.figure()
        print(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'])

        ax = myfig.add_subplot(1, 1, 1)

        print("color-color", self.name, self.type, self.Vmax,
              band1, self.maxcolors[band1]['color'], band2,
              self.maxcolors[band2]['color'],
              self.maxcolors[band1]['dcolor'], band2,
              self.maxcolors[band2]['dcolor'], end="")
        #        print self.su.mysymbols[self.type]
        typekey = self.type
        if typekey not in self.su.mytypecolors.keys():
            typekey = 'other'
            legends = myplot_err(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'],
                                 yerr=self.maxcolors[band1]['dcolor'],
                                 xerr=self.maxcolors[band2]['dcolor'],
                                 symbol=self.su.mytypecolors[typekey] + self.su.mysymbols[typekey],
                                 alpha=1, offset=0, fig=myfig, fcolor=self.su.mytypecolors[typekey], ms=15,
                                 markeredgewidth=2)
        else:
            legends = myplot_err(self.maxcolors[band1]['color'], self.maxcolors[band2]['color'],
                                 yerr=self.maxcolors[band1]['dcolor'],
                                 xerr=self.maxcolors[band2]['dcolor'],
                                 symbol=self.su.mytypecolors[typekey] + self.su.mysymbols[typekey],
                                 alpha=0.5, offset=0, fig=myfig, fcolor=self.su.mytypecolors[typekey], ms=15)

        ax.annotate("", xy=(2.6, 2.6), xycoords='data',
                    xytext=(2.3, 2.3), textcoords='data', ha='center', va='center',
                    arrowprops=dict(arrowstyle="->", color='#b20000'), )
        myplot_setlabel(xlabel=band1, ylabel=band2, title=None, label=label, xy=(0.75, 0.8), labsize=labsize)
        if plotred:
            pl.figtext(0.88, 0.9, "red", fontsize=labsize)

        # pl.plot(self.maxcolors[band1]['color'],self.maxcolors[band2]['color'],c=self.su.mytypecolors[typekey],marker=self.su.mysymbols[typekey], markernsize=8, alpha=0.5)
        return (myfig, legends, (self.maxcolors[band1]['color'], self.maxcolors[band2]['color']))

    def plotsn(self, photometry=False, band='', color=False, c='',
               fig=None, ax=None, show=False,
               verbose=False, save=False, savepng=False, symbol='', title='',
               Vmax=None, plottemplate=False, plotpoly=False, plotspline=False,
               relim=True, xlim=None, ylim=None, offsets=False, ylabel='Mag',
               aspect=1, nir=False, allbands=True, fcolor=None, legendloc=1,
               nbins=None, singleplot=False, noylabel=False, ticklabelsz=16):

        su = setupvars()
        from pylab import rc
        rc('axes', linewidth=2)
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 20
        mpl.rcParams['font.family'] = 'Times New Roman'
        # mpl.rcParams['font.serif'] = 'Times'
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['xtick.labelsize'] = ticklabelsz
        mpl.rcParams['ytick.labelsize'] = ticklabelsz
        if save:
            mpl.rcParams['ytick.major.pad'] = '6'
        offset = 0.0
        boffsets = su.bandoffsets
        # 'J': -3, 'H': -4, 'K': -5, 'w1': 3, 'm2': 4, 'w2': 5}

        print("\n##############  PLOTTING SUPERNOVA : ", self.name, "###############")
        myfig = fig  # None
        if not myfig:
            myfig = pl.figure()  # , figsize=(30,60))
        if photometry:
            print("plotting...")
            if not ax:
                ax1 = myfig.add_subplot(1, 1, 1)
                adjustFigAspect(myfig, aspect=aspect)
            else:
                ax1 = ax

            ax1.minorticks_on()
            majorFormatter = FormatStrFormatter('%d')
            minorLocator = MultipleLocator(0.2)
            #            majorLocator   = MultipleLocator()
            #            ax.yaxis.set_minor_locator(minorLocator)
            ax1.yaxis.set_major_formatter(majorFormatter)

            legends = []
            notused = []
            if band == '':
                if allbands:
                    mybands = self.su.bands
                else:
                    mybands = self.su.bandsnonir
            else:
                mybands = [band]

            #            if self.stats[mybands[0]].maxjd[1]==0.0:
            #                self.getstats(mybands[0])
            scaleAdjust = int(1e-3 *
                              np.nanmin([self.photometry[mb]['mjd'].min() \
                                             if self.filters[mb] > 0 else np.nan \
                                         for mb in mybands])) * 1.0e3
            print([self.photometry[mb]['mjd'].min() \
                       if self.filters[mb] > 0 \
                       else np.nan for mb in mybands])
            print("scaleAdjust, xlim", scaleAdjust, xlim)
            xAdjust = scaleAdjust
            if scaleAdjust < 2400000:
                scaleAdjust += 2400000
                print(scaleAdjust)
            limAdjust = scaleAdjust - 2400000
            if xlim:
                myxlim = (xlim[0] - xAdjust, xlim[1] - xAdjust)
            elif not relim:
                myxlim = ax.get_xlim()
            else:

                if self.Vmax > 10e5:
                    scaleAdjust = int(float(self.Vmax) * 1e-3) * 1.0e3
                    myxlim = (float(self.Vmax) - 10 - scaleAdjust,
                              float(self.Vmax) + 10 - scaleAdjust)
                    limAdjust = scaleAdjust - 2400000.0
                elif self.Vmax <= 10e5:
                    scaleAdjust = int(float(self.Vmax) * 1e-3) * 1.0e3
                    myxlim = (float(self.Vmax) - 10 - scaleAdjust,
                              float(self.Vmax) + 10 - scaleAdjust)
                    limAdjust = scaleAdjust
                else:
                    scaleAdjust = 0
                    myxlim = (min([self.photometry[mb]['mjd'].min() \
                                       if self.filters[mb] > 0 else np.nan \
                                   for mb in mybands]),
                              max([self.photometry[mb]['mjd'].min() \
                                       if self.filters[mb] > 0 else np.nan \
                                   for mb in mybands]))
                    limAdjust = 0

                if not self.stats[mybands[0]].maxjd == [0.0, 0.0]:
                    xlim = [float(self.stats[mybands[0]].maxjd[0]) - limAdjust - 10,
                            float(self.stats[mybands[0]].maxjd[0]) + 10 - limAdjust]
                else:
                    xlim = [self.Vmax - scaleAdjust - 10.5,
                            self.Vmax - scaleAdjust + 10]

                    # print xlim
                    # print self.photometry
            if ylim:
                myylim = ylim
                if 1:  # verbose:
                    print("ylim ", myylim)
            elif not self.stats[mybands[0]].maxjd[0] == 0.0:
                myylim = (0, 0)

            else:
                bandind = 0
                while len(self.photometry[mybands[bandind]]['mjd']) == 0:
                    bandind += 1
                    if bandind > len(mybands):
                        pass
                #                print mybands[bandind],self.photometry[mybands[bandind]]['mjd']
                myxlim = (min(self.photometry[mybands[bandind]]['mjd']) - 20 - limAdjust,
                          max(self.photometry[mybands[bandind]]['mjd']) + 20 - limAdjust)
                myylim = (0, 0)

            if int(str(int(myxlim[1]))[-1]) < 5:
                myxlim = (myxlim[0], myxlim[1] + 5)

            #            majorLocator   = MultipleLocator(int((myxlim[1]-myxlim[0])/6))
            majorFormatter = FormatStrFormatter('%d')
            bandswdata = []
            # if title=='':
            #    title =self.name
            ylabel = ylabel.replace('+0', '')
            print(self.type)
            if self.name[-2].isdigit():
                label = self.name.upper()
            else:
                label = self.name
            label = label.replace('sn', 'SN ') + "\n" + self.type

            # ax.locator_params(tight=True, nbins=4)
            majorLocator = MultipleLocator(5)
            try:
                if (ylim[0] - ylim[1]) < 10:
                    majorLocator = MultipleLocator(2)
            except:
                pass
            ax1.yaxis.set_major_locator(majorLocator)

            if noylabel:
                myplot_setlabel(xlabel='JD - %.1f' % scaleAdjust, title=title,
                                label=label, ax=ax1, ylabel="  ",
                                rightticks=True, labsize=21)
            else:
                myplot_setlabel(xlabel='JD - %.1f' % scaleAdjust,
                                ylabel=ylabel, title=title,
                                label=label, ax=ax1, labsize=18)

            for b in mybands:
                if offsets:
                    offset = boffsets[b]
                else:
                    offset = 0.0
                if self.filters[b] == 0:
                    if verbose:
                        print("nothing to plot for ", b)
                        notused.append(b)
                    continue
                bandswdata.append(b.replace('r', 'r\'').replace('i', 'i\''))

                if verbose:
                    print("plotting band ", b, " for ", self.name)
                if not relim or ylim:
                    ylim = ax1.get_ylim()
                if relim:
                    xlim = [min(myxlim[0], min(self.photometry[b]['mjd']) - 10 - limAdjust),
                            max(myxlim[1], max(self.photometry[b]['mjd']) + 10 - limAdjust)]
                    myxlim = xlim
                    if myylim == (0, 0):
                        ylim = (20, 0)
                        myylim = (max(ylim[1], max(self.photometry[b]['mag']) + 1 + offset),
                                  min(ylim[0], min(self.photometry[b]['mag']) - 1 + offset))
                        if verbose:
                            print("this is the new myylim", myylim)
                    elif myylim is None:
                        myylim = (max(ylim[1], (max(myylim[0], max(self.photometry[b]['mag']) + 1 + offset))),
                                  min(ylim[0], min(myylim[1], min(self.photometry[b]['mag']) - 1 + offset)))

                if 'J' in b or 'H' in b or 'K' in b:
                    fcolor = 'None'
                else:
                    fcolor = self.su.mycolors[b]
                if symbol == '':
                    symbol = '%s%s' % (self.su.mycolors[b], self.su.myshapes[b])

                try:
                    l, = myplot_err(np.asarray(self.photometry[b]['mjd']) - xAdjust,
                                    np.asarray(self.photometry[b]['mag']),
                                    yerr=np.asarray(self.photometry[b]['dmag']),
                                    xlim=myxlim,
                                    ylim=myylim, symbol=symbol, offset=offset,
                                    fcolor=fcolor, fig=myfig, ax=ax1,
                                    litsn=np.array(self.photometry[b]['camsys']) == 'lit')
                    # print self.photometry[b]['mjd']-53000.0,
                except:
                    l, = myplot_err(np.asarray(self.photometry[b]['mjd']) - xAdjust,
                                    np.asarray(self.photometry[b]['mag']),
                                    xlim=myxlim,
                                    ylim=myylim, symbol=symbol, offset=offset,
                                    fcolor='None',
                                    # fcolor,
                                    fig=myfig, ax=ax1,
                                    litsn=np.array(self.photometry[b]['camsys']) == 'lit')

                legends.append(l)
                ax1.legend(legends, bandswdata, loc=legendloc, ncol=1,
                           prop={'size': 12}, numpoints=1, framealpha=0.5)

                symbol = ''
                if plotpoly or plottemplate:
                    if self.Vmax:
                        fullxrange = np.arange(float(self.Vmax) - scaleAdjust - 10.,
                                               float(self.Vmax) - scaleAdjust + 40.0, 0.1)
                    else:
                        try:
                            fullxrange = np.arange(self.stats['V'].maxjd[0] - 10.,
                                                   self.stats['V'].maxjd[0] + 40., 0.1)
                        except:
                            continue
                    if plotpoly and self.solution[b]['sol']:
                        myplot_err(fullxrange, self.solution[b]['sol'](fullxrange),
                                   symbol='%s-' % self.su.mycolors[b], offset=offset)
                    if plottemplate and not self.stats[b].templrchisq == 0:
                        myplot_err(fullxrange, self.templsol[b](fullxrange, [self.stats[b].templatefit['stretch'],
                                                                             self.stats[b].templatefit['xoffset'],
                                                                             self.stats[b].templatefit['yoffset'],
                                                                             self.stats[b].templatefit['xstretch']], b),
                                   symbol='%s--' % self.su.mycolors[b], offset=offset)
                if plottemplate:
                    if savepng:
                        pl.savefig(self.name + "_" + b + ".template.png", bbox_inches='tight')
                    if save:
                        thisname = self.name + "_" + b + ".template.pdf"
                        try:
                            pl.savefig(thisname)
                        except RuntimeError:
                            pl.show()
                        thisdir = os.environ['SESNPATH']
                        os.system("perl %s/pdfcrop.pl %s" % (thisdir, thisname))
                if plotspline:
                    if verbose:
                        print("plotting spline")
                    x = self.photometry[b]['mjd'].astype(np.float64)
                    #                         print "here", x, self.snspline[b]
                    fullxrange = np.arange(min(x), max(x), 0.1)
                    a = self.snspline[b](fullxrange)
                    smoothed = smooth(a, window_len=5)
                    #            smoothed=sp.signal.filter.medfilter(a,5)
                    #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                    #                    results = zip([x[0] for x in results], smoothed)
                    # print "fr",fullxrange,smoothed
                    myplot_err(fullxrange - limAdjust, smoothed, symbol='%s.' % self.su.mycolors[b], offset=offset,
                               settopx=False, fig=myfig)
                    pl.show()

            if Vmax and not self.flagmissmax:
                if self.Vmax:
                    try:
                        myplotarrow(float(self.Vmax) - scaleAdjust,
                                    min(self.photometry['V']['mag']) - 0.5,
                                    label="V max")
                    except:
                        pass
            #                else:
            #                    try:
            #                        myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
            #                    except:
            #                        pass
            print("Vmax:", self.Vmax, self.flagmissmax)
            print(self.photometry[b]['phase'])
            if verbose:
                print("Vmax:", self.Vmax, self.flagmissmax)
            ax1.tick_params('both', length=10, width=1, which='major')
            ax1.tick_params('both', length=5, width=1, which='minor')
            pl.setp(ax1.get_xticklabels(), fontsize=ticklabelsz,
                    rotation=20)

            Vmax4plot = 0
            if not self.flagmissmax:
                ax2 = ax1.twiny()
                ax2.tick_params('both', length=10, width=1, which='major')
                ax2.tick_params('both', length=5, width=1, which='minor')
                ax2.set_xlabel("phase (days)")
                pl.setp(ax2.get_xticklabels(), fontsize=ticklabelsz)
                if verbose:
                    print("putting second axis")
                Vmax4plot = self.Vmax
                if Vmax4plot > 2400000:
                    Vmax4plot -= scaleAdjust
                if Vmax4plot > 53000:
                    Vmax4plot -= limAdjust
                print(Vmax4plot)
                # print "here", (myxlim)

                ax2.set_xlim((myxlim[0] - Vmax4plot, myxlim[1] - Vmax4plot))
                if (myxlim[1] - myxlim[0]) < 100:
                    ax2.xaxis.set_major_locator(MultipleLocator(20))
                ax2.xaxis.set_minor_locator(MultipleLocator(10))

                if myxlim[0] - Vmax4plot < 0 and myxlim[1] - Vmax4plot > 0:
                    ax2.plot([0, 0], [ax1.get_ylim()[0],
                                      ax1.get_ylim()[1]], 'k-', alpha=0.3)
                    ax2.fill_between([0 - self.dVmax, 0 + self.dVmax],
                                     [ax1.get_ylim()[0], ax1.get_ylim()[0]],
                                     [ax1.get_ylim()[1], ax1.get_ylim()[1]],
                                     color='k', alpha=0.2)

            for i in notused:
                mybands.remove(i)
            if savepng:
                pl.savefig(self.name + "_" + ''.join(mybands) + '.png', bbox_inches='tight')
            if save:
                thisname = self.name + "_" + ''.join(mybands) + '.pdf'
                try:
                    pl.savefig(thisname)
                except RuntimeError:
                    self.printsn(photometry=True)
                    pl.show()

                print("running pdfcrop.pl")

                os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))

            if nir:
                legends = []
                myfig_nir = pl.figure()
                ax = myfig_nir.add_subplot(1, 1, 1)
                if band == '':
                    mybands = self.su.bandsnir
                else:
                    mybands = [band]
                if xlim:
                    myxlim = xlim - limAdjust

                elif not relim:
                    xlim = pl.xlim()
                else:
                    xlim = [float(self.stats[mybands[0]].maxjd[0]) - limAdjust - 10,
                            float(self.stats[mybands[0]].maxjd[0]) - limAdjust + 10]
                if ylim:
                    myylim = ylim
                elif not self.stats[mybands[0]].maxjd[0] == 0.0:
                    myxlim = (min(xlim[0], float(self.stats[mybands[0]].maxjd[0]) - 10 - 53000),
                              max(xlim[1], float(self.stats[mybands[0]].maxjd[0]) + 10 - 53000))
                elif self.Vmax:
                    if self.Vmax > 10e5:
                        myxlim = (float(self.Vmax) - 10 - 2400000 - limAdjust,
                                  float(self.Vmax) + 10 - 2400000 - limAdjust)
                    else:
                        myxlim = (float(self.Vmax) - 10 - limAdjust,
                                  float(self.Vmax) + 10 - limAdjust)
                else:
                    myxlim = (min(self.photometry[mybands[0]]['mjd']) - 20 - limAdjust,
                              max(self.photometry[mybands[0]]['mjd']) + 20 - limAdjust)
                bandswdata = []
                for b in mybands:
                    if offsets:
                        offset = boffsets[b]
                    else:
                        offset = 0.0
                    if verbose:
                        print("band here", b)
                    if self.filters[b] == 0:
                        if verbose:
                            print("nothing to plot for ", b)
                        notused.append(b)
                        continue
                    bandswdata.append(b)

                    if verbose:
                        print("plotting band ", b, " for ", self.name)
                    if not relim or ylim:
                        ylim = pl.ylim()
                    elif relim:
                        ylim = [float(self.stats[mybands[0]].maxjd[1]) + 10,
                                float(self.stats[mybands[0]].maxjd[1]) - 1 - 5]
                        if myylim == (0, 0):
                            myylim = (max(ylim[1], max(self.photometry[b]['mag']) + 1),
                                      min(ylim[0], min(self.photometry[b]['mag']) - 1))
                        else:
                            myylim = (max(ylim[1], (max(myylim[0], max(self.photometry[b]['mag']) + 1))),
                                      min(ylim[0], min(myylim[1], min(self.photometry[b]['mag']) - 1)))
                    if verbose:
                        print("myylim", myylim)
                    #                    if title=='':
                    #                        title =self.name
                    #                    myplot_setlabel(xlabel='JD - 2453000.00',ylabel=ylabel,title=title, ax=ax, label=label)
                    if symbol == '':
                        symbol = '%s%s' % (self.su.mycolors[b], self.su.myshapes[b])

                    legends.append(myplot_err(self.photometry[b]['mjd'] - limAdjust,
                                              self.photometry[b]['mag'],
                                              yerr=self.photometry[b]['dmag'],
                                              xlim=myxlim,
                                              ylim=myylim, symbol=symbol, offset=offset,
                                              fig=my_nir))

                    symbol = ''
                    if verbose:
                        print("vmax", self.Vmax)
                    if plotpoly or plottemplate:
                        if self.Vmax:
                            fullxrange = np.arange(float(self.Vmax) - scaleAdjust - 10.,
                                                   float(self.Vmax) - scaleAdjust + 40.0, 0.1)
                        else:
                            try:
                                fullxrange = np.arange(self.stats['V'].maxjd[0] - 10., self.stats['V'].maxjd[0] + 40.,
                                                       0.1)
                            except:
                                continue

                        if plotpoly and self.solution[b]['sol']:
                            myplot_err(fullxrange, self.solution[b]['sol'](fullxrange),
                                       symbol='%s-' % self.su.mycolors[b], offset=offset)
                        if plottemplate and not self.stats[b].templrchisq == 0:
                            myplot_err(fullxrange, self.templsol[b](fullxrange, [self.stats[b].templatefit['stretch'],
                                                                                 self.stats[b].templatefit['xoffset'],
                                                                                 self.stats[b].templatefit['yoffset'],
                                                                                 self.stats[b].templatefit['xstretch']],
                                                                    b), symbol='%s--' % self.su.mycolors[b],
                                       offset=offset)
                    if savepng:
                        pl.savefig(self.name + "_" + b + ".template.png", bbox_inches='tight')
                    if save:
                        thisname = self.name + "_" + b + ".template.pdf"
                        pl.savefig(thisname)
                        os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))
                    if plotspline:
                        if versbose:
                            print("plotting spline")
                        x = self.photometry[b]['mjd'].astype(np.float64)
                        fullxrange = np.arange(min(x), max(x), 0.1)
                        a = self.snspline[b](fullxrange)
                        smoothed = smooth(a, window_len=5)
                        #            smoothed=sp.signal.filter.medfilter(a,5)
                        #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                        #                    results = zip([x[0] for x in results], smoothed)

                        myplot_err(fullxrange, smoothed, symbol='%s-' % self.su.mycolors[b], offset=offset)
                        #                    print smoothed

                if Vmax:
                    if self.Vmax:
                        myplotarrow(float(self.Vmax) - scaleAdjust, min(self.photometry['V']['mag']) - 0.5,
                                    label="V max")
                #                    else:
                #                        try:
                #                            myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
                #                        except:
                #                            pass

                pl.legend(legends[::-1], bandswdata[::-1], loc=1, ncol=1, prop={'size': 12}, numpoints=1,
                          framealpha=0.5)
                for i in notused:
                    mybands.remove(i)
                if savepng:
                    bnds = "UBVRIriHJK.png"
                    pl.savefig(self.name + "_" + bnds, bbox_inches='tight')
                if save:
                    bnds = "UBVRIriHJK.pdf"
                    thisname = self.name + "_" + bnds
                    pl.savefig(thisname)
                    os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))

        if color:
            rc('axes', linewidth=1)
            if photometry:
                print("need new fig number")

            myfig = fig
            if not myfig:
                myfig = pl.figure()
            if not singleplot:
                ax = myfig.add_subplot(2, 1, 1)
            else:
                adjustFigAspect(myfig, aspect=2)
                ax = myfig.add_subplot(1, 1, 1)

            ax.minorticks_on()

            majorFormatter = FormatStrFormatter('%.1f')
            minorLocator = MultipleLocator(0.2)
            majorLocator = MultipleLocator(1.0)
            if '06jc' in self.name:
                majorLocator = MultipleLocator(2.0)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_major_formatter(majorFormatter)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)
                # specify integer or one of preset strings, e.g.
                # tick.label.set_fontsize('x-small')

                #            adjustFigAspect(myfig)#,aspect=aspect)
            legends = []
            notused = []
            if c == '':
                mybands = [k for k in self.su.cs.keys()]
                if mybands.index("B-i"):
                    del mybands[mybands.index("B-i")]
                # print mybands
            else:
                mybands = [c]
            myylim = (0, 0)
            myxlim = (-15, 85)
            workingbands = []
            for b in mybands:
                if len(self.colors[b]['mjd']) == 0:
                    if verbose:
                        print("nothing to plot for ", b)
                    notused.append(b)
                    continue
                if verbose:
                    print("plotting band ", b, " for ", self.name)
                    print(self.colors[b]['mjd'], self.colors[b]['mag'])

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                myylim = (float(min(myylim[0], min(self.colors[b]['mag']) - 0.5)),
                          float(max(myylim[1], max(self.colors[b]['mag']) + 0.5)))
                if self.name[-2].isdigit():
                    thename = self.name[:-1] + self.name[-1].upper()
                else:
                    thename = self.name
                myplot_setlabel(xlabel='', ylabel='color (mag)', label=thename, ax=ax, labsize=15)
                if '06jc' in self.name:
                    myxlim = (0, 85)
                    myylim = (-2, 7.5)

                l, = myplot_err(self.colors[b]['mjd'],  # )-53000.0,
                                self.colors[b]['mag'],
                                yerr=self.colors[b]['dmag'],
                                xlim=myxlim, ylim=myylim, symbol='%so' % self.su.mycolorcolors[b], offset=offset,
                                alpha=0.5)
                workingbands.append(b)
                legends.append(l)

            if '06jc' in self.name:
                loc = 2
                ncol = 4
                pl.xlim(pl.xlim()[0] - 10, pl.xlim()[1])
            else:
                loc = 1
                ncol = 1

            #            sort2vectors(v1,v2)
            pl.legend(legends, workingbands, loc=loc, ncol=ncol, prop={'size': 12}, numpoints=1, framealpha=0.2)
            if singleplot:
                ax.set_xlabel("phase (days)")
            else:
                ax2 = myfig.add_subplot(2, 1, 2, sharex=ax)
                ax2.minorticks_on()
                majorFormatter = FormatStrFormatter('%.1f')
                minorLocator = MultipleLocator(0.2)
                ax2.yaxis.set_minor_locator(minorLocator)
                ax2.yaxis.set_major_formatter(majorFormatter)

                for b in mybands:
                    if len(self.colors[b]['mjd']) == 0:
                        if verbose:
                            print("nothing to plot for ", b)
                        notused.append(b)
                        continue
                    if verbose:
                        print("plotting band ", b, " for ", self.name)
                        print(self.colors[b]['mjd'], self.colors[b]['mag'])

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                    myylim = (float(min(myylim[0], min(self.colors[b]['mag']) - 0.5)),
                              float(max(myylim[1], max(self.colors[b]['mag']) + 0.5)))

                    myplot_setlabel(xlabel='phase (days)', ylabel='color', ax=ax2)

                    if '06jc' in self.name:

                        ax2.annotate("red", xy=(-10, 2.05), xycoords='data',
                                     xytext=(-10, 1.4), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#b20000'), )
                        ax2.annotate("blue", xy=(-10, -0.65), xycoords='data',
                                     xytext=(-10, -0.2), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#0066cc'), )
                    else:
                        ax2.annotate("red", xy=(80, 2.05), xycoords='data',
                                     xytext=(80, 1.4), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#b20000'), )
                        ax2.annotate("blue", xy=(80, -0.65), xycoords='data',
                                     xytext=(80, -0.2), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->", color='#0066cc'), )
                    myplot_hist(self.colors[b]['mjd'],  # )-53000.0,
                                self.colors[b]['mag'],
                                xlim=myxlim, ylim=(-0.75, 2.2), symbol='%so' % self.su.mycolorcolors[b], offset=offset,
                                ax=ax2, nbins=nbins)  #

                    # print b,b[0],self.su.mycolors[b[0]]
                    #                legends.append(l)
            for i in notused:
                try:
                    mybands.remove(i)
                except:
                    pass
            if savepng:
                pl.savefig(self.name + "_color" + '.png', bbox_inches='tight', dpi=150)
            if save:
                thisname = self.name + "_color" + '.pdf'
                pl.savefig(thisname)
                os.system("perl %s/pdfcrop.pl %s" % (os.environ['SESNPATH'], thisname))
        if show:
            pl.show()

        return myfig

    def cleanphot(self):
        self.Vmaxmag = 0.0
        self.filters = {}
        for b in self.su.bands:
            self.filters[b] = 0
        self.polysol = {}
        self.snspline = {}
        self.templsol = {}
        self.solution = {}
        self.photometry = {}
        self.stats = {}
        self.colors = {}
        self.maxcolors = {}
        self.maxmags = {}

        # self.flagmissmax = True
        #        self.lc={}
        for b in self.su.bands:
            self.photometry[b] = {'mjd': np.zeros((0), float), 'mag': np.zeros((0), float),
                                  'dmag': np.zeros((0), float), 'extmag': np.zeros((0), float),
                                  'natmag': np.zeros((0), float), 'flux': np.zeros((0), float),
                                  'phases': np.zeros((0), float), 'camsys': ['']
                                  }
            self.stats[b] = snstats()
            self.polysol[b] = None
            self.snspline[b] = None
            self.templsol[b] = None
            self.solution[b] = {'sol': None, 'deg': None, 'pars': None, 'resid': None}
            self.maxmags[b] = {'epoch': 0.0, 'mag': float('NaN'), 'dmag': float('NaN')}

        for c in self.su.cs:
            self.maxcolors[c] = {'epoch': 0.0, 'color': float('NaN'), 'dcolor': float('NaN')}
            self.colors[c] = {'mjd': [], 'mag': [],
                              'dmag': []}  # np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}

        self.polyfit = None
        self.setphot()

    def getphot(self, ebmv=0, RIri=False, verbose=False, quiet=False):
        self.snnameshort = self.name.replace('sn19', '').replace('sn20', '').strip()
        if verbose:
            print(self.snnameshort, "E(B-V)", ebmv)
            print(self.filters)
        for b in self.su.bands:

            ##############################setting up band#########################
            if self.filters[b] <= 0:
                continue
            litindx = []

            indx = np.array(np.where(self.lc['photcode'] == self.su.photcodes[b][0].encode('utf-8'))[0])
            if not self.su.photcodes[b][1] == self.su.photcodes[b][0]:
                if len(indx) == 0:
                    indx = np.array(np.where(self.lc['photcode'] == self.su.photcodes[b][1].encode('utf-8'))[0])
                else:
                    phot1indx = np.where(self.lc['photcode'] == self.su.photcodes[b][1].encode('utf-8'))[0]
                    if len(phot1indx) > 0:
                        newindx = np.concatenate([phot1indx, indx])
                        indx = newindx
            if not b in ['w1', 'w2', 'm2']:
                if sum(self.lc['photcode'] == self.su.photcodes[b][2][:2].encode('utf-8')) > 0:
                    litindx = np.where(self.lc['photcode'] == self.su.photcodes[b][2][:2].encode('utf-8'))[0]
                    newindx = np.concatenate([litindx, indx])
                    indx = newindx


            self.getphotband(indx, b, litindx)
            if verbose:
                print(self.snnameshort, b, self.photometry[b]['mjd'])
            try:
                self.stats[b].tlim = (min(self.photometry[b]['mjd']), max(self.photometry[b]['mjd']))
                self.stats[b].maglim = (min(self.photometry[b]['mag']), max(self.photometry[b]['mag']))

            except:
                pass
            self.filters[b] = len(self.photometry[b]['mjd'])
        if not ebmv == 0:
            self.extcorrect(ebmv, verbose=verbose)
        try:
            self.Vmax = float(self.Vmax)
            if self.Vmax < 2400000 and not self.Vmax == 0:
                self.Vmax += 2453000.5
        except:
            if self.Vmax.startswith("<0") and len(self.photometry["V"]['mjd']) > 0:
                self.Vmax = "<24" + str(self.photometry["V"]['mjd'][0])
            if self.Vmax.startswith("<"):
                try:
                    self.Vmax = self.Vmax.replace("<", "")

                    self.Vmax = float(self.Vmax)
                #                    if self.Vmax=float(self.Vmax)
                except:
                    pass
            self.nomaxdate = True

        if self.filters['R'] == 0 and self.filters['I'] == 0 and self.filters['r'] > 0 and self.filters[
            'i'] > 0 and RIri == True:

            self.getonecolor('r-i', quiet=quiet)
            #             print "right here",self.filters
            #             self.printsn(color=True,cband='r-i')
            tmptimeline = np.array(self.colors['r-i']['mjd']) + self.Vmax

            if tmptimeline[0] > 2300000:
                tmptimeline -= 2400000.5
            tmplin = interp1d(tmptimeline, self.colors['r-i']['mag'], kind='linear', bounds_error=False)
            tmplinerr = interp1d(tmptimeline, self.colors['r-i']['dmag'], kind='linear', bounds_error=False)
            self.photometry['R']['mjd'] = self.photometry['r']['mjd']
            self.photometry['R']['mag'] = self.photometry['r']['mag'] - 0.153 * tmplin(
                self.photometry['R']['mjd']) - 0.117
            self.photometry['R']['dmag'] = np.sqrt(
                self.photometry['r']['dmag'] ** 2 + tmplinerr(self.photometry['R']['mjd']) ** 2 + 0.00043 ** 2)

            self.photometry['I']['mjd'] = self.photometry['R']['mjd']
            self.photometry['I']['mag'] = self.photometry['R']['mag'] - 0.930 * tmplin(
                self.photometry['I']['mjd']) - 0.259
            self.photometry['I']['dmag'] = np.sqrt(
                self.photometry['R']['dmag'] ** 2 + tmplinerr(self.photometry['I']['mjd']) ** 2 + 0.00055 ** 2)
            self.filters['R'] = self.filters['r']
            self.filters['I'] = self.filters['R']

        if self.filters['r'] == 0 and self.filters['i'] == 0 and self.filters['B'] > 0 and self.filters[
            'V'] > 0 and RIri == True:
            ##Jester et al. (2005) transformations via
            ##https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
            self.getonecolor('B-V', quiet=quiet)
            # print self.Vmax, np.array(self.colors['B-V']['mjd'])+self.Vmax
            tmptimeline = np.array(self.colors['B-V']['mjd']) + self.Vmax
            if tmptimeline[0] > 2300000:
                tmptimeline -= 2400000.5
            tmplin = interp1d(tmptimeline, self.colors['B-V']['mag'], kind='linear', bounds_error=False)
            tmplinerr = interp1d(tmptimeline, self.colors['B-V']['dmag'], kind='linear', bounds_error=False)
            self.photometry['r']['mjd'] = self.photometry['V']['mjd']
            self.photometry['r']['mag'] = self.photometry['V']['mag'] - 0.42 * tmplin(
                self.photometry['r']['mjd']) + 0.11
            self.photometry['r']['dmag'] = np.sqrt(
                self.photometry['V']['dmag'] ** 2 + tmplinerr(self.photometry['r']['mjd']) ** 2 + 0.03 ** 2)
            self.filters['r'] = self.filters['V']

            self.getonecolor('R-I', quiet=quiet)
            tmptimeline = np.array(self.colors['R-I']['mjd']) + self.Vmax
            if len(tmptimeline) > 0:
                if tmptimeline[0] > 2300000:
                    tmptimeline -= 2400000.5
                tmplin = interp1d(tmptimeline, self.colors['R-I']['mag'], kind='linear', bounds_error=False)
                tmplinerr = interp1d(tmptimeline, self.colors['R-I']['dmag'], kind='linear', bounds_error=False)
                self.photometry['i']['mjd'] = self.photometry['r']['mjd']
                self.photometry['i']['mag'] = self.photometry['r']['mag'] - 0.91 * tmplin(
                    self.photometry['i']['mjd']) + 0.20
                self.photometry['i']['dmag'] = np.sqrt(
                    self.photometry['r']['dmag'] ** 2 + tmplinerr(self.photometry['i']['mjd']) ** 2 + 0.03 ** 2)
                self.filters['i'] = self.filters['r']

    def setphase(self, verbose=False):
        for b in self.su.bands:
            if self.filters[b] == 0:
                continue
            # print self.photometry[b]['mjd'][0], self.Vmax, self.photometry[b]['mjd'][0]<2400000, self.photometry[b]['mjd'][0]>4e4, 4e4
            if self.photometry[b]['mjd'][0] < 2400000 and \
                    self.photometry[b]['mjd'][0] > 1e4:
                if verbose:
                    print("case 1")
                self.photometry[b]['phase'] = self.photometry[b]['mjd'] - \
                                              self.Vmax + 2400000.5
            elif self.photometry[b]['mjd'][0] > 2400000:
                if verbose:
                    print("case 2")
                self.photometry[b]['phase'] = self.photometry[b]['mjd'] - \
                                              self.Vmax

            else:
                if verbose:
                    print("case 3")
                self.photometry[b]['phase'] = self.photometry[b]['mjd'] * \
                                              float('NaN')
            if verbose:
                print(self.photometry[b]['phase'][0])

    def setAbsmag(self, Dl=None, verbose=False):
        if Dl:
            self.Dl = Dl
        if self.Dl == 0:
            print("must set Dl first")
            return -1
        for b in self.su.bands:
            self.photometry[b]['AbsMag'] = self.photometry[b]['mag'] - \
                                           5.0 * (np.log10(self.Dl) - 1)

            self.photometry[b]['flux'] = 10 ** (23. - \
                                                (self.photometry[b]['AbsMag'] + 48.6) / 2.5)
            self.photometry[b]['dflux'] = self.photometry[b]['dmag'] * \
                                          self.photometry[b]['flux'] / 2.5 * np.log(10.)

    def getphotband(self, indx, b, litindx):
        # if 1:
        try:
            self.photometry[b] = {'mjd': self.lc['mjd'][indx],
                                  'phase': np.empty_like(self.lc['mjd']),
                                  'mag': self.lc['ccmag'][indx],
                                  'dmag': self.lc['dmag'][indx], \
                                  'natmag': self.lc['mag'][indx],
                                  'extmag': self.lc['mag'][indx] * np.nan,
                                  'flux': self.lc['mag'][indx] * np.nan,
                                  'camsys': np.array([survey[0] if i < survey_dates[0] else survey[1] \
                                      if i < survey_dates[1] \
                                      else survey[2][self.pipeline] \
                                      if i < survey_dates[2] else survey[3] \
                                                      for i in self.lc['mjd'][indx]])
                                  }
            for i in range(len(indx)):
                if indx[i] in litindx:
                    self.photometry[b]['camsys'][i] = 'lit'
        except:
            print("#############\n\n\n failed to get photometry \n\n\n##############")
            pass
            # self.printsn(photometry=True)

    #            self.photometry[b]={'mjd':self.lc['mjd'][indx],'mag':self.lc['mag'][indx],'dmag':self.lc['dmag'][indx], 'camsys':[survey[0] if i<survey_dates[0] else survey[1] if i<survey_dates[1] else survey[2][self.pipeline]  if i<survey_dates[2] else survey[3]   for i in self.lc['mjd'][indx]]}

    def extcorrect(self, ebmv, verbose=False):
        if verbose:
            print("ebmv", ebmv)
        for b in self.su.bands:
            R = self.su.AonEBmV[b] * ebmv
            self.photometry[b]['mag'] -= R

    def getmaxcolors(self, band, tol=5, quiet=False):
        (self.maxcolors[band]['epoch'], self.maxcolors[band]['color'],
         self.maxcolors[band]['dcolor']) = self.getepochcolors(band,
                                                               tol=tol,
                                                               quiet=False)

        #
        print(self.name, self.maxcolors[band]['epoch'],
              self.maxcolors[band]['color'],
              self.maxcolors[band]['dcolor'])  # #

    #
    #        if len(self.colors[band]['mjd'])<=1:
    #            "print no data"
    #            return -1
    #        else:
    #            indx=np.where(abs(np.array(self.colors[band]['mjd']))==min(abs(np.array(self.colors[band]['mjd']))))[0]
    #            if len(indx)>0:
    #                indx=indx[0]###
    #
    #            if abs(self.colors[band]['mjd'][indx]) > tol:
    #                print "#########\n\n\n no data within 5 days ",self.name,self.type,"\n\n\n############"
    #                self.maxcolors[band]['epoch'] = float('NaN')
    #                self.maxcolors[band]['color'] =float('NaN')
    #                self.maxcolors[band]['dcolor'] =float('NaN')
    #            else:
    #                print "#########\n\n\n YES data within 5 days ",self.name,self.type,"\n\n\n############"
    #                self.maxcolors[band]['epoch'] = self.colors[band]['mjd'][indx]
    #                self.maxcolors[band]['color'] = self.colors[band]['mag'][indx]
    #                self.maxcolors[band]['dcolor'] =self.colors[band]['dmag'][indx]
    #            print self.maxcolors[band]['epoch'],self.maxcolors[band]['color'],self.maxcolors[band]['dcolor']

    def getepochcolors(self, band, epoch=0.0, tol=5,
                       interpolate=False, quiet=False):
        if len(self.colors[band]['mjd']) < 1:
            print("no data")
            return (float('NaN'), float('NaN'), float('NaN'))
        else:
            if not interpolate:
                indx, = np.where(abs(np.array(self.colors[band]['mjd']) - epoch) == min(
                    abs(np.array(self.colors[band]['mjd']) - epoch)))
                #            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
                #            print indx
                if len(indx) > 0:
                    indx = indx[0]
                if abs(self.colors[band]['mjd'][indx] - epoch) > tol:
                    return (float('NaN'), float('NaN'), float('NaN'))
                return (self.colors[band]['mjd'][indx], self.colors[band]['mag'][indx], self.colors[band]['dmag'][indx])
            else:
                ddate = self.colors[band]['mjd'] - epoch
                try:
                    indlft, = np.arange(ddate.size)[ddate == ddate[ddate < 0].max()]
                    indrgt, = np.arange(ddate.size)[ddate == ddate[ddate > 0].min()]
                except ValueError:
                    if not quiet:
                        print("the datapoints don't surround epoch, " +
                              "im at the edge ", epoch)
                    return (float('NaN'), float('NaN'), float('NaN'))

                mag = interp1d(self.colors[band]['mjd'][indlft:indrgt + 1], self.colors[band]['mag'][indlft:indrgt + 1],
                               kind='linear')(epoch)
                err = np.sqrt(self.colors[band]['dmag'][indlft] ** 2 + self.colors[band]['dmag'][indrgt] ** 2)
                return (epoch, float(mag), err)

    def getmagmax(self, band, tol=2, forceredo=False, verbose=False, quiet=False):
        if is_empty(self.metadata, verbose=verbose):
            return -1
        if verbose:
            print(self.name, 'cfa' + band.upper() + 'max',
                  self.metadata['cfa' + band.upper() + 'max'])
        nomax = True
        if not forceredo:
            try:
                self.maxmags[band]['epoch'] = float(self.metadata['CfA ' + band.upper() + 'JD bootstrap'])
                self.maxmags[band]['mag'] = float(self.metadata['cfa' + band.upper() + 'max'])
                self.maxmags[band]['dmag'] = float(self.metadata['cfa' + band.upper() + 'maxerr'])
                if verbose:
                    print("we have max's", self.maxmags[band]['epoch'],
                          self.maxmags[band]['mag'],
                          self.maxmags[band]['dmag'])
                nomax = False
            except:
                if verbose:
                    print("no max's")
        if nomax or forceredo:
            if self.Vmax:
                if verbose:
                    print("self.Vmax:", self.Vmax)
                if not type(self.Vmax) == float:
                    pass
                if float(self.Vmax) > 2000000:
                    self.Vmax = float(self.Vmax) - 2400000.5
                if verbose:
                    print("Vm:", self.Vmax, Vmax)
                self.maxmags[band]['epoch'], \
                self.maxmags[band]['mag'], \
                self.maxmags[band]['dmag'] = self.getepochmags(band,
                                                                 epoch=self.Vmax + coffset[band],
                                                                 verbose=verbose, tol=tol,
                                                                 quiet=quiet)

        # self.getepochmags(band,epoch=Vmax+coffset)))[band],tol=tol)

    #                  print self.maxmags[band]['epoch'],self.maxmags[band]['mag'],self.maxmags[band]['dmag'],"after getepochmags"

    def getmagmax_band(self, band, epoch=None, tol=5, verbose=False, quiet=True):
        if not epoch:
            epoch = self.Vmax + coffset[band]
            if self.Vmax or not type(self.Vmax) == float:
                if verbose:
                    print("self.Vmax:", self.Vmax)
                if float(self.Vmax) > 2000000:
                    Vmax = float(self.Vmax) - 2400000.5
                    if verbose:
                        print("Vm in getmagmax_band:", self.Vmax, Vmax)
                else:
                    return (0, 0, 0, 0)
        indx, = np.where((self.photometry[band]['mjd'] < epoch + 15) & (
                self.photometry[band]['mjd'] > self.photometry[band]['mjd'] - 8))
        if verbose:
            print(indx, self.photometry[band]['mjd'], epoch)
        x = self.photometry[band]['mjd'][indx]
        y = self.photometry[band]['mag'][indx]
        e = self.photometry[band]['dmag'][indx]
        if verbose:
            print(x, y)
        #        pl.figure()
        #        pl.plot(x-self.Vmax+2400000.5,y)
        #        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
        #        try:
        try:
            nodes = splrep(x, y, w=1.0 / (self.photometry[band]['dmag'][indx]) ** 2, k=2)
            newx = np.arange(x[0], x[-1], 0.1)
            splx = splev(newx, nodes)
            mymax = min(splx)
            if verbose:
                print(mymax, end="")
            epmax = newx[np.where(splx == mymax)][0]
        #            print epmax
        #            pl.plot(newx-self.Vmax+2400000.5,splx)
        #            return (epmax,mymax)
        except:
            # if not quiet: print "splining to find max mag failed for band ",band
            return (0, 0, 0)

        #        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
        #
        #        pl.ylim(pl.ylim()[1],pl.ylim()[0])
        #        pl.show()
        #        accept=int(raw_input("is this spline reasonable? 1 yes 0 no"))
        # if accept>0:
        #        print epmax,mymax, np.sqrt((e[np.where(x<epmax)[0][-1]])**2+(e[np.where(x>epmax)[0][0]])**2)
        try:
            return (epmax, mymax, np.sqrt((e[np.where(x < epmax)[0][-1]]) ** 2 + (e[np.where(x > epmax)[0][0]]) ** 2))
        except IndexError:
            return (epmax, mymax, e[np.where(x == epmax)[0][0]])

    #        elif accept<0:
    #             sys.exit()
    #        else:
    #             return(0,0,0)

    #        (self.maxmags[band]['epoch'], self.maxmags[band]['mag'], self.maxmags[band]['dmag'])=self.getepochmags(band, tol=tol, epoch=bandepoch)

    def getepochmags(self, band, phase=None, epoch=None, tol=2, interpolate=False, verbose=False, plot=False,
                     quiet=False):
        if verbose:
            print("getting band", band, "magnitudes")
        if self.filters[band] == 0:
            if verbose:
                print("no data in filter ", band)
            return (float('NaN'), float('NaN'), float('NaN'))
        if not epoch:
            if not phase:
                epoch = self.Vmax - 2400000.5
            if phase:
                epoch = self.Vmax - 2400000.5 + phase
        myc = band.lower()
        if myc == 'i':
            myc = 'b'
        if myc == 'j':
            myc = 'm'
        if myc == 'h':
            myc = 'c'
        if myc == 'k':
            myc = 'm'
        # if plot:
        #     if phase:
        #         pl.plot(self.photometry[band]['mjd'] - self.Vmax + 2400000.5, self.photometry[band]['mag'], '%s-' % myc)
        #         pl.errorbar(self.photometry[band]['mjd'] - self.Vmax + 2400000.5, self.photometry[band]['mag'],
        #                     self.photometry[band]['dmag'], fmt='k.')
        #     else:
        #         pl.plot(self.photometry[band]['mjd'], self.photometry[band]['mag'], '%s-' % myc)
        #         pl.errorbar(self.photometry[band]['mjd'], self.photometry[band]['mag'], self.photometry[band]['dmag'],
        #                     fmt='k.')
        #     pl.title(self.name)

            #            pl.draw()
        try:
            ddate = abs(np.array(self.photometry[band]['mjd']) - epoch)
            indx, = np.where(ddate == min(ddate))
        except ValueError:
            print("no min ")
            indx = []
        #            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
        if verbose:
            print(indx, self.photometry[band]['mjd'][indx],
                  abs(self.photometry[band]['mjd'][indx] - epoch))

        if len(indx) > 0:
            indx = indx[0]
        #            print  abs(self.colors[band]['mjd'][indx]-epoch)

        if min(ddate) == 0:
            if verbose:
                print("observaions at same exact epoch!")
            # print self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx]
            return (self.photometry[band]['mjd'][indx],
                    self.photometry[band]['mag'][indx],
                    self.photometry[band]['dmag'][indx])

        if min(ddate) > tol and not interpolate:
            if verbose:
                print("nodata within ", tol, "days of ", epoch)
            return (epoch, float('NaN'), float('NaN'))
        if verbose:
            print(self.photometry[band]['mjd'][indx],
                  self.photometry[band]['mag'][indx],
                  self.photometry[band]['dmag'][indx])
        if not interpolate:
            if plot:
                if phase:
                    pl.errorbar(self.photometry[band]['mjd'][indx] - \
                                self.Vmax + 2400000.5,
                                self.photometry[band]['mag'][indx],
                                yerr=self.photometry[band]['dmag'][indx],
                                fmt='r')
                else:
                    pl.errorbar(self.photometry[band]['mjd'][indx],
                                self.photometry[band]['mag'][indx],
                                yerr=self.photometry[band]['dmag'][indx],
                                fmt='r')
            return (self.photometry[band]['mjd'][indx],
                    self.photometry[band]['mag'][indx],
                    self.photometry[band]['dmag'][indx])

        if interpolate:
            if verbose:
                print("interpolating")
            ddate = self.photometry[band]['mjd'] - epoch
            try:
                indlft, = np.arange(ddate.size)[ddate == ddate[ddate < 0].max()]
                indrgt, = np.arange(ddate.size)[ddate == ddate[ddate > 0].min()]
            except ValueError:
                if not quiet:
                    print(band, ": the datapoint dont surround epoch, " +
                          "im at the edge ", epoch, ddate)
                    # ,self.photometry[band]['mjd'])
                return (self.photometry[band]['mjd'][indx],
                        self.photometry[band]['mag'][indx],
                        self.photometry[band]['dmag'][indx])

            mag = interp1d(self.photometry[band]['mjd'][indlft:indrgt + 1],
                           self.photometry[band]['mag'][indlft:indrgt + 1],
                           kind='linear')(epoch)
            err = np.sqrt(self.photometry[band]['dmag'][indlft] ** 2 +
                          self.photometry[band]['dmag'][indrgt] ** 2)
            if plot:
                if phase:
                    pl.errorbar(epoch - self.Vmax + 2400000.5, float(mag),
                                yerr=err, fmt='r')
                else:
                    pl.errorbar(epoch, float(mag), yerr=err, fmt='r')
            return (epoch, float(mag), err)



    def getcolors(self, BmI=False, Bmi=False, verbose=False, quiet=False):
        ###setup B-I for bolometric correction as per Lyman 2014
        for ckey in self.su.cs.keys():
            ##################iterate over the color keys to get the colors cor each object
            ##############THIS IS LAME AND I MUST FIND A BETTER WAY TO DO IT!!###########
            if verbose:
                print(ckey)
            self.getonecolor(ckey, verbose=verbose, quiet=quiet)
            if len(self.colors['r-i']['mjd']) == 0:
                self.getonecolor('r-i', quiet=quiet)
            if BmI:
                if self.filters['I'] == 0 and (self.filters['i'] > 0 and \
                                               self.filters['r'] > 0):
                    if verbose:
                        print(self.photometry['r'], self.photometry['i'])

                    tmpmjd = []
                    tmpI = []
                    tmpIerr = []
                    for k, mjd in enumerate(self.photometry['r']['mjd']):
                        timediff = np.abs(np.array(self.colors['r-i']['mjd']) + \
                                          self.Vmax - 2400000.5 - mjd)
                        if verbose:
                            print("timediff", timediff)
                        if min(timediff) < 1.5:
                            mjdind = np.where(timediff == min(timediff))[0]
                            if len(mjdind) > 1:
                                mjdind = mjdind[0]
                            tmpmjd.append(np.mean([self.colors['r-i']['mjd'][mjdind] + self.Vmax - 2400000.5, mjd]))
                            tmpI.append(
                                self.photometry['r']['mag'][k] - 1.2444 * (self.colors['r-i']['mag'][mjdind]) - 0.3820)
                            tmpIerr.append(np.sqrt(self.photometry['r']['dmag'][k] ** 2 + self.colors['r-i']['dmag'][
                                mjdind] ** 2 + 0.0078 ** 2))
                    self.photometry['I']['mjd'] = np.array(tmpmjd)
                    self.photometry['I']['mag'] = np.array(tmpI)
                    self.photometry['I']['dmag'] = np.array(tmpIerr)
                    self.filters['I'] = len(tmpmjd)
                    #        self.printsn(photometry=True)
                    self.getonecolor('B-I', verbose=verbose, quiet=quiet)
                    #        self.printsn(color=True)

            if Bmi:
                if self.filters['i'] == 0 and (self.filters['I'] > 0 and self.filters['R'] > 0):
                    if verbose:
                        print(self.photometry['R'], self.photometry['I'])

                    tmpmjd = []
                    tmpi = []
                    tmpierr = []
                    for k, mjd in enumerate(self.photometry['I']['mjd']):
                        timediff = np.abs(np.array(self.colors['R-I']['mjd']) + self.Vmax - 2400000.5 - mjd)
                        if verbose:
                            print("timediff", timediff)
                        if min(timediff) < 1.5:
                            mjdind = np.where(timediff == min(timediff))[0]
                            if len(mjdind) > 1:
                                mjdind = mjdind[0]
                            tmpmjd.append(np.mean([self.colors['R-I']['mjd'][mjdind] + self.Vmax - 2400000.5, mjd]))
                            tmpi.append(
                                self.photometry['I']['mag'][k] + 0.247 * self.colors['R-I']['mag'][mjdind] + 0.329)

                            tmpierr.append(np.sqrt(self.photometry['I']['dmag'][k] ** 2 + \
                                                   self.colors['R-I']['dmag'][mjdind] ** 2 \
                                                   + 0.003 ** 2))
                    self.photometry['i']['mjd'] = np.array(tmpmjd)
                    self.photometry['i']['mag'] = np.array(tmpi)
                    self.photometry['i']['dmag'] = np.array(tmpierr)
                    self.filters['i'] = len(tmpmjd)
                    #        self.printsn(photometry=True)
                    self.getonecolor('B-i', verbose=verbose, quiet=quiet)
                    #        self.printsn(color=True)

    def getonecolor(self, ckey, verbose=False, quiet=False):
        if verbose:
            print("ckey", ckey)
            print(self.colors[ckey])
        if not self.colors[ckey]['mjd'] == [] and not len(self.colors[ckey]['mjd']) == 0:
            if not ckey == 'r-i' and not ckey == 'B-i':
                if not quiet:
                    print("color", ckey,
                          "is already there. clean it first " +
                          "if you want me to redo it")
        else:
            if isinstance(self.colors[ckey]['mjd'], (np.ndarray, np.generic)):
                self.colors[ckey]['mjd'] = []
                self.colors[ckey]['mag'] = []
                self.colors[ckey]['dmag'] = []
            indx = np.nan
            for k, mjd in enumerate(self.photometry[ckey[0]]['mjd']):
                # check vmax:
                if not type(self.Vmax) == float or float(self.Vmax) < 200000:
                    self.Vmax = float(self.photometry[ckey[0]]['mjd'][0]) + 2453000.5
                mjd = float(mjd)
                try:
                    timediff = min(abs(self.photometry[ckey[2]]['mjd'] - mjd))
                    if verbose:
                        print("timediff ", timediff)
                except:
                    continue
                if timediff < 1.5:
                    indx = np.where(abs(self.photometry[ckey[2]]['mjd'] - mjd) == timediff)[0]
                    indx = indx[0]
                    if verbose:
                        print("mags ", mjd,
                              self.photometry[ckey[2]]['mag'][indx])

            if ~np.isnan(indx):
                self.colors[ckey]['mjd'].append(mjd - float(self.Vmax) + 2400000.5)

                self.colors[ckey]['mag'].append(self.photometry[ckey[0]]['mag'][k] - \
                                                self.photometry[ckey[2]]['mag'][indx])
                self.colors[ckey]['dmag'].append( \
                    np.sqrt(self.photometry[ckey[0]]['dmag'][k] ** 2 + \
                            self.photometry[ckey[2]]['dmag'][indx] ** 2))
            self.colors[ckey]['mjd'] = np.array(self.colors[ckey]['mjd']).flatten()
            self.colors[ckey]['mag'] = np.array(self.colors[ckey]['mag']).flatten()
            self.colors[ckey]['dmag'] = np.array(self.colors[ckey]['dmag']).flatten()
            if verbose:
                print(self.colors[ckey])

    def savecolors(self, band=None):
        if band is None:
            mybands = [k for k in self.su.cs.keys()]
        else:
            mybands = [band]

        for c in mybands:
            fout = open(self.name + "_" + c + ".dat", "w")
            if len(self.colors[c]['mjd']) > 0:
                for i, mjd in enumerate(self.colors[c]['mjd']):
                    fout.write(self.colors[c]['mjd'][i],
                               self.colors[c]['mag'][i],
                               self.colors[c]['dmag'][i])

    def formatlitsn(self, lit_lc, verbose=True):
        nir = False

        #        print lit_lc.dtype
        #        sys.exit()
        thissnkeys = lit_lc.dtype.names
        # print(thissnkeys)
        if not 'mjd' in thissnkeys:
            if not 'JD' in thissnkeys:
                print("need 'mjd' or 'JD' in the literature lcv")
                return -1
            import numpy.lib.recfunctions as rf
            # print(lit_lc['JD'])
            lit_lc = rf.append_fields(lit_lc, 'mjd', lit_lc['JD'] - 2400000.5,
                                      dtypes=lit_lc['JD'].dtype, usemask=False,
                                      asrecarray=True)
            # print lit_lc

        if not self.snnameshort:
            self.snnameshort = self.name.replace('sn19', '') \
                .replace('sn20', '').strip()

        fileout = open(os.environ['SESNPATH'] + \
                       "literaturedata/slc.sn" + \
                       self.snnameshort + '.f', 'w')

        for b in self.su.bands:
            if verbose:
                print(b, b in thissnkeys)
            if b == 'J' or b == 'K' or b == 'H':
                nir = True
                # print "NIR:", nir
                continue
            if not b in thissnkeys:
                continue
            if not 'd' + b in thissnkeys:
                print("missing mag errors, include them in literature lcv " +
                      "as a vectore named 'd<b>' for every filter b " +
                      "for which you have magnitudes (e.g. V -> dV)")
            if not b in self.su.photcodes.keys():
                continue
            for i, dp in enumerate(lit_lc[b]):
                # print dp
                if np.isnan(dp):
                    continue
                if verbose:
                    print(self.su.photcodes[b][2], lit_lc['mjd'][i],
                          'nan', 'nan', lit_lc['d' + b][i], dp)
                fileout.write('{0} {1} {2} {3} {4} {5}\n'.format(self.su.photcodes[b][2],
                                                                 lit_lc['mjd'][i],
                                                                 'nan', 'nan', lit_lc['d' + b][i], dp))
        if verbose:
            print("file out:", fileout)
            print("NIR", nir)
        if nir:
            # if len(self.name) == 7:
            fileout = open(os.environ['SESNPATH'] + "literaturedata/nirphot/" + self.snnameshort+ '.dat', 'w') #self.name[:-1] + self.name[-1].upper() + '.dat', 'w')
            # else:
            #     fileout = open(os.environ['SESNPATH'] + "literaturedata/nirphot/" + self.snnameshort + '.dat', 'w')

            for b in ['K', 'J', 'H']:
                if not b in thissnkeys:
                    continue
                if not 'd' + b in thissnkeys:
                    print("missing mag errors, include them in literature " +
                          "lcv as a vectore named 'd<b>' for every filter " +
                          "b for which you have magnitudes (e.g. V -> dV)")
                if not b in self.su.photcodes.keys():
                    continue
                for i, dp in enumerate(lit_lc[b]):
                    # print dp
                    if np.isnan(dp):
                        continue
                    if verbose:
                        print(b + 'l', lit_lc['mjd'][i], dp,
                              lit_lc['d' + b][i])
                    fileout.write('{0} {1} {2} {3}\n'.format(b + 'l', lit_lc['mjd'][i], dp,
                                                            lit_lc['d' + b][i]))
            if verbose:
                print("file out:", fileout)

    def loadCfA3(self, f, superverbose=False):
        try:
            self.lc = np.loadtxt(f, usecols=(0, 1, 6, 7, 8,),
                                 dtype={'names': ('photcode', 'mjd', 'mag', 'dmag',
                                                  'ccmag'), \
                                        'formats': ('S2', 'f', 'f', 'f', 'f')})
            self.lc['photcode'] = ['%02d' % int(p) for p in self.lc['photcode']]
            flux = 10 ** (-self.lc['ccmag'] / 2.5) * 5e10
            dflux = flux * self.lc['dmag'] / LN10x2p5
            if superverbose:
                print(self.lc['mjd'])
                print(self.lc['ccmag'])
                print(self.lc['photcode'])

        except:
            return None, None
        return flux, dflux

    def loadCfA4(self, f, verbose=False):
        try:
            lc = np.loadtxt(f, usecols=(0, 1, 5, 3, 7), \
                            dtype={'names': ('photcode', 'mjd', \
                                             'mag', 'dmag', 'ccmag'), \
                                   'formats': ('S2', 'f', 'f', 'f', 'f')})

            flux = 10 ** (-self.lc['ccmag'] / 2.5) * 5e10
            dflux = flux * self.lc['dmag'] / LN10x2p5
        except:
            if verbose:
                print("trying again", f)
            try:
                lc = np.loadtxt(f, usecols=(0, 1, 5, 4, 5), \
                                dtype={'names': ('photcode', 'mjd', \
                                                 'mag', 'dmag', 'ccmag'), \
                                       'formats': ('S2', 'f', 'f', 'f', 'f')})
                #                    self.lc['ccmag'] = self.lc['mag']
                flux = 10 ** (-lc['mag'] / 2.5) * 5e10

                dflux = flux * lc['dmag'] / LN10x2p5

            except:
                if verbose:
                    print("failed at loading CfA4")
                return None, None, None
        # print  lc, flux, dflux
        return lc, flux, dflux

    def loadNIR(self, f, verbose=False):
        # loading NIR data
        try:
            if verbose:
                print(f)
            self.nirlc = np.loadtxt(f, usecols=(0, 1, 2, 3), \
                                    dtype={'names': ('photcode', 'mjd',
                                                     'mag', 'dmag'), \
                                           'formats': ('S1', 'f', \
                                                       'f', 'f')})
            print ((self.nirlc['mjd']).size)

            nirflux = 10 ** (-self.nirlc['mag'] / 2.5) * 5e10
            nirdflux = nirflux * self.nirlc['dmag'] / LN10x2p5
            if (self.nirlc['mjd']).size > 0:
                self.nir = True
                lc = {}
                lc['photcode'] = np.concatenate([self.lc['photcode'],
                                                 self.nirlc['photcode']], axis=0)
                lc['mjd'] = np.concatenate([self.lc['mjd'], self.nirlc['mjd']])
                lc['ccmag'] = np.concatenate([self.lc['ccmag'], self.nirlc['mag']])
                lc['mag'] = np.concatenate([self.lc['mag'], self.nirlc['mag']])
                lc['dmag'] = np.concatenate([self.lc['dmag'], self.nirlc['dmag']])
                self.lc = lc
                flux = 10 ** (-lc['mag'] / 2.5) * 5e10

                dflux = flux * lc['dmag'] / LN10x2p5
            else:
                flux, dflux = None, None
        except ValueError:
            if verbose:
                print("passing Value Error in loadsn nir, " +
                      "no nir data presusmibly")
            return None, None

        return flux, dflux

    def loadlit(self, f):
        print("loading lit data")

        lc, flux, dflux = self.loadCfA4(f, verbose=True)
        newlc = {}
        newlc['photcode'] = np.concatenate([self.lc['photcode'], lc['photcode']], axis=0)
        newlc['mjd'] = np.concatenate([self.lc['mjd'], lc['mjd']])
        newlc['ccmag'] = np.concatenate([self.lc['ccmag'], lc['mag']])
        newlc['mag'] = np.concatenate([self.lc['mag'], lc['mag']])
        newlc['dmag'] = np.concatenate([self.lc['dmag'], lc['dmag']])

        self.lc = newlc
        return flux, dflux

#TODO: unCommented by Somayeh
    def loadsn(self, f, fnir=None,
               lit=False, verbose=False, superverbose=False, addlit=False):
        if f.split('/')[-1].startswith('slc'):
            self.pipeline = 'CfA4'
            if verbose:
                print("lightcurve type CfA4 ", f)
            self.lc, flux, dflux = self.loadCfA4(f)
            if flux is None and dflux is None:
                print("file ", f, " failed. wrong file format? moving on ")
                return 0, 0, 0, 0
        elif f.split('/')[-1].startswith('sn'):
            self.pipeline = 'CfA3'
            print("lightcurve type CfA3")
            # sys.exit()
            if verbose:
                print("lightcurve type CfA3")
            flux, dflux = self.loadCfA3(f, superverbose=superverbose)
            if flux is None and dflux is None:
                print("file ", f, " failed. wrong file format? moving on ")
                return 0, 0, 0, 0
        else:
            if verbose:
                print("what kind of file is this??? " + f)
            return 0, 0, 0, 0
        allf = self.optfiles

        if self.lit or self.addlit:
            for f in allf[1:]:
                reflux, redflux = self.loadlit(f)
                flux = np.concatenate([flux, reflux])
                dflux = np.concatenate([dflux, redflux])

        if verbose:
            print("fnir?:", fnir, self.fnir)
        if fnir and self.fnir:
            if verbose:
                print("doing nir", self.fnir)
            nirflux, nirdflux = np.array([]), np.array([])
            if isinstance(self.fnir, basestring):
                nirflux, nirdflux = self.loadNIR(self.fnir,
                                                 verbose=verbose)
            else:
                for f in self.fnir:
                    nirreflux, nirredflux = self.loadNIR(f, verbose=verbose)
                    # print nirreflux, nirflux
                    if not nirreflux is None:
                        # print "now ", f, nirreflux
                        nirflux = np.concatenate([nirflux, nirreflux])
                        nirdflux = np.concatenate([nirdflux, nirredflux])

            if nirflux is None and nirdflux is None:
                pass
            else:
                return self.lc, flux, flux, self.name

        return self.lc, flux, dflux, self.name

    def loadsn2(self, verbose=False, superverbose=False, D11=False):
        # print("here", self.optfiles)
        if not D11:
            for f in self.optfiles:
                if f.split("/")[-1].startswith('D11'):
                    self.optfiles.remove(f)
        # print(self.optfiles)
        if len(self.optfiles) > 0:
            # first file
            if self.optfiles[0].split('/')[-1].startswith('D11'):
                f = self.optfiles[1]
            else:
                f = self.optfiles[0]
            if f.split('/')[-1].startswith('slc'):
                self.pipeline = 'CfA4'
                if verbose:
                    print("lightcurve type CfA4 ", f)
                    print("loading", f.split('/')[-1])
                self.lc, flux, dflux = self.loadCfA4(f)
                # self.printsn(photometry=True)
                if flux is None and dflux is None:
                    print("file ", f, " failed. wrong file format? moving on ")
                    return 0, 0, 0, 0
            elif f.split('/')[-1].startswith('sn'):
                self.pipeline = 'CfA3'
                if verbose:
                    print("lightcurve type CfA3")
                    print("loading", f.split('/')[-1])
                flux, dflux = self.loadCfA3(f, superverbose=superverbose)
                if flux is None and dflux is None:
                    print("file ", f,
                          " failed. wrong file format? moving on ")
                    return 0, 0, 0, 0


            else:
                if verbose:
                    print("what kind of file is this??? " + f)
                return 0, 0, 0, 0

            for f in self.optfiles[1:]:
                if verbose:
                    print("loading", f)
                reflux, redflux = self.loadlit(f)
                flux = np.concatenate([flux, reflux])
                dflux = np.concatenate([dflux, redflux])

        else:
            flux = []
            reflux = []
            dflux = []
            redflux = []

        if len(self.fnir) > 0:
            # print(self.fnir)
            if verbose:
                print("doing nir", self.fnir)
            nirflux, nirdflux = np.array([]), np.array([])
            if isinstance(self.fnir, basestring):
                nirflux, nirdflux = self.loadNIR(self.fnir,
                                                 verbose=verbose)
            else:
                for f in self.fnir:
                    nirreflux, nirredflux = self.loadNIR(f, verbose=verbose)
                    if not nirreflux is None:
                        nirflux = np.concatenate([nirflux, nirreflux])
                        nirdflux = np.concatenate([nirdflux, nirredflux])

            # if nirflux is None and nirdflux is None:
            #    pass
            # else:
            # return self.lc, flux, flux, self.name
        return self.lc, flux, dflux, self.name

    def sortlc(self):
        for b in self.su.bands:
            if self.filters[b] == 0:
                continue
            sortindx = np.argsort(self.photometry[b]['mjd'])
            self.photometry[b]['mjd'] = self.photometry[b]['mjd'][sortindx]
            self.photometry[b]['phase'] = self.photometry[b]['phase'][sortindx]
            self.photometry[b]['AbsMag'] = self.photometry[b]['mag'][sortindx]
            self.photometry[b]['mag'] = self.photometry[b]['mag'][sortindx]
            self.photometry[b]['dmag'] = self.photometry[b]['dmag'][sortindx]
            self.photometry[b]['camsys'] = self.photometry[b]['camsys'][sortindx]
            self.photometry[b]['extmag'] = self.photometry[b]['extmag'][sortindx]
            self.photometry[b]['natmag'] = self.photometry[b]['natmag'][sortindx]
            self.photometry[b]['flux'] = self.photometry[b]['flux'][sortindx]
            self.photometry[b]['dflux'] = self.photometry[b]['flux'][sortindx]

    def getstats(self, b, verbose=False):
        # find max day and dm15
        xp = np.linspace(min(self.photometry[b]['mjd']), max(self.photometry[b]['mjd']), 10000)
        if self.solution[b]['pars']:
            print(self.solution[b]['pars'])
            maxjd = float(polyroots(self.solution[b]['pars'][::-1])[0].real)
            #                       print root,xp[np.where(solution['sol'](xp)==
            #                                               min(solution['sol'](xp)))[0]]

            if maxjd > min(self.photometry[b]['mjd']) and maxjd < max(self.photometry[b]['mjd']):
                if verbose:
                    print("root found is within data range")
                self.stats[b].maxjd = [maxjd, self.solution[b]['sol'](maxjd)]
                self.stats[b].dm15 = self.stats[b].maxjd[1] - self.solution[b]['sol'](maxjd + 15.0)

            else:
                print("root NOT found is within data range")
                mjdindex = np.where(self.solution[b]['sol'](xp) ==
                                    min(self.solution[b]['sol'](xp)))[0]
            if len(mjdindex) > 1:
                mjdindex = [mjdindex[1]]
            try:
                self.stats[b].maxjd[0] = xp[mjdindex]
                self.stats[b].maxjd[1] = self.solution[b]['sol'](self.stats[b].maxjd[0])[0]
                self.stats[b].dm15 = self.stats[b].maxjd[1] - self.solution[b]['sol'](self.stats[b].maxjd[0] + 15.0)

                try:
                    if len(self.stats[b].maxjd[0]) > 1:
                        self.stats[b].maxjd[0] = np.mean(self.stats[b].maxjd[0])
                        print("WARNING data has multiple points at same epoch")
                    if len(self.stats[b].maxjd[1]) > 1:
                        self.stats[b].maxjd[1] = np.mean(self.stats[b].maxjd[1])
                        print("WARNING data has multiple points at same epoch")
                except:
                    pass
            except:
                self.stats[b].maxjd[0] = -1000
                self.stats[b].dm15 = -1000
                print("WARNING data does not constraint the max mag")
                self.stats[b].flagmissmax = 1

        try:
            self.stats[b].maxjd[0] = self.stats[b].maxjd[0][0]
        except:
            pass

        if b == 'V':
            self.Vmax = self.stats[b].maxjd[0] + 2453000
            try:
                #                print "\n\n\nVmaxmag"
                tmp = float(self.metadata['MaxVMag'])
                print("\n\n\nVmaxmag", tmp)
                self.Vmaxmag = tmp
                print(self.Vmaxmag)
            except:
                self.Vmaxmag = self.stats[b].maxjd[1]

        if 1:
            #        try:
            self.stats[b].m15data[0] = self.stats[b].maxjd[0] + 15.
            # print self.stats[b].m15data[0]
            try:
                if len(self.stats[b].m15data[0]) > 1:
                    self.stats[b].m15data[0] = np.mean(self.stats[b].m15data[0])
                    print("WARNING data has multiple points at same epoch")
            except:
                pass
            self.stats[b].m15data[1] = [self.photometry[b]['mag'][np.where(
                self.photometry[b]['mjd'] == min([tmp for tmp in self.photometry[b]['mjd']
                                                  if tmp > self.stats[b].m15data[0]]))[0]],
                                        self.photometry[b]['mag'][np.where(
                                            self.photometry[b]['mjd'] == max([tmp for tmp in self.photometry[b]['mjd']
                                                                              if tmp < self.stats[b].m15data[0]]))[0]]]

            try:
                if len(self.stats[b].m15data[1]) > 1:
                    self.stats[b].m15data[1] = np.mean(self.stats[b].m15data[1])
            except:
                pass
            try:
                if len(self.stats[b].dm15) > 1:
                    self.stats[b].dm15 = np.mean(self.stats[b].dm15)
            except:
                pass

    #        except:
    #            print "WARNING data does not constraint mag at 15 days"
    #            self.stats[b].flagmiss15=2
    ##                        print bandlist[bandcounter]
    #        self.stats[b].success=1

    def printlog(self, b, inst, logoutput):
        logoutput.write("%-30s %s %-10s %-10s %02d " % (self.name, self.type, b, inst, self.filters[b]), end="")
        try:
            logoutput.write("%5.3f %02d %5.3f  %-10s %d %-10s %d %-10s %5.3f " % \
                            (abs(self.stats[b].polyrchisq),
                             self.stats[b].polydeg,
                             np.median(np.abs(self.stats[b].polyresid)) / \
                             0.6745, " ", self.stats[b].flagmissmax,
                             " ", self.stats[b].flagmiss15, " ",
                             self.stats[b].dm15), end="")
        except:
            print("could not print the log to logoutput")
            print(b, self.stats[b].polyrchisq, self.stats[b].polydeg,
                  np.median(np.abs(self.stats[b].polyresid)) / 0.6745,
                  " ", self.stats[b].flagmissmax, " ",
                  self.stats[b].flagmiss15, " ", self.stats[b].dm15)

        try:
            l = len(self.stats[b].maxjd[1])
            if l > 1:
                self.stats[b].maxjd[1] = np.mean(self.stats[b].maxjd[1])
                try:
                    self.stats[b].maxjd[0] = np.mean(self.stats[b].maxjd[0])
                except:
                    pass
                print("WARNING: maxjd is an array: " +
                      "something is very wrong with the fit!")
                logoutput.write("-1000 -1000")
                self.stats[b].flagmissmax, self.stats[b].flagbadfit = 0.0, 4
                pl.savefig("%s.%s.%s.png" % (self.name, b, inst),
                           bbox_inches='tight')
                return -1
            else:
                logoutput.write("%5.3f %5.3f " % (self.stats[b].dm15lin,
                                                  self.stats[b].maxjd[0] + \
                                                  2453000.0))
        except:
            try:
                logoutput.write("%5.3f %5.3f " % (self.stats[b].dm15lin, self.stats[b].maxjd[0] + 2453000.0))
            except:
                print("could not output to log")
                print(self.stats[b].dm15lin,
                      self.stats[b].maxjd[0] + 2453000.0)

        return 0
