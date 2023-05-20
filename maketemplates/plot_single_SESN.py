import os
import pylab as pl
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
from matplotlib.ticker import (MultipleLocator)
# s = json.load( open(str(os.getenv ('PUI2015'))+"/fbb_matplotlibrc.json") )
# pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import matplotlib as mpl


mpl.use('agg')


pl.rcParams['figure.figsize']=(10,10)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

bands = ['U','B','V', 'g', 'R', 'I', 'rp','ip','up','J','H','K','m2','w1','w2']
color_bands = {'U':'k','up':'k','B':'#0066cc','V':'#47b56c','R':'#b20000','I':'m',
                         'rp':'#b20000','ip':'m', 'g': '#317e4b',
                         'J':'#4F088A','H':'#FFB700','K':'#A4A4A4',
                         'm2':'#708090', 'w2':'#a9b2bc', 'w1':'#434d56'}
plot_vmax = False
allbands = False
allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                             "/SESNessentials.csv")['SNname'].values


if __name__ == '__main__':
    # uncomment for all lcvs to be read in
    if len(sys.argv) > 1:

        for arg in sys.argv:
            if arg.startswith('name='):
                allsne = arg.split('=')[1].split(',')
            elif arg.startswith('band='):
                bands = arg.split('=')[1].split(',')
            elif arg.startswith('vmax'):
                plot_vmax = True
            elif arg.startswith('allbands'):
                allbands = True


    for sn in allsne:
        for b in bands:

            bb = b
            if b == 'ip':
                bb = 'i'
            if b == 'up':
                bb = 'u'
            if b == 'rp':
                bb = 'r'

            try:
                thissn = snstuff.mysn(sn, addlit=True)
            except AttributeError:
                continue

            if len(thissn.optfiles) + len(thissn.fnir) == 0:
                print("bad sn")

            # read metadata for SN
            thissn.readinfofileall(verbose=False, earliest=False, loose=True)
            thissn.printsn()

            # check SN is ok and load data
            if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
                print("bad sn")
            print(" starting loading ")
            # print (os.environ['SESNPATH'] + "/finalphot/*" + \
            #        thissn.snnameshort.upper() + ".*[cf]")
            # print (os.environ['SESNPATH'] + "/finalphot/*" + \
            #        thissn.snnameshort.lower() + ".*[cf]")

            # print( glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
            #                  thissn.snnameshort.upper() + ".*[cf]") + \
            #        glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
            #                  thissn.snnameshort.lower() + ".*[cf]") )

            lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
            thissn.setphot()
            thissn.getphot()
            thissn.setphase()
            thissn.sortlc()
            # thissn.printsn()

            # check that it is k
            if np.array([n for n in thissn.filters.values()]).sum() == 0:
                print("bad sn")

            if len(thissn.photometry[bb]['mjd']) == 0:
                print('No photometry for '+ sn+ ' in band '+b)
                continue

            xmin = thissn.photometry[bb]['mjd'].min()

            if xmin - thissn.Vmax < -1000:
                x = thissn.photometry[bb]['mjd'] - 55000.5#- thissn.Vmax + 2400000.5
                x2 = thissn.photometry[bb]['mjd'] - thissn.Vmax + 2400000.5
                # vmax = thissn.Vmax - 55000.5

            elif xmin - thissn.Vmax > 1000:
                x = thissn.photometry[bb]['mjd'] - 2455000.5 #- thissn.Vmax - 2400000.5
                x2 = thissn.photometry[bb]['mjd'] - thissn.Vmax - 2400000.5
                # vmax = thissn.Vmax - 2455000.5
            else:
                x = thissn.photometry[bb]['mjd'] #- thissn.Vmax
                x2 = thissn.photometry[bb]['mjd'] - thissn.Vmax
                # vmax = thissn.Vmax

            if thissn.Vmax > 2400000:
                vmax = thissn.Vmax - 2455000.5
            else:
                vmax = thissn.Vmax - 55000.5

            dvmax = thissn.dVmax

            y = thissn.photometry[bb]['mag']
            # x2 = thissn.photometry[bb]['mjd'] -

            # y = y.min() - y

            yerr = thissn.photometry[bb]['dmag']

            fig = plt.figure(figsize=(14, 14))

            ax = plt.gca()
            ax2 = ax.twiny()

            # plt.errorbar(x, y, yerr = yerr, color = color_bands[b],fmt = '.', ls = '-', linewidth = 1)
            plt.errorbar(x, y, yerr=yerr, color=color_bands[b], fmt='^', linewidth=1, markersize=20, label=bb)

            ax.yaxis.get_ticklocs(minor=True)
            ax.minorticks_on()
            ax.invert_yaxis()

            ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=10, labelsize=40,
                           width=2)
            ax2.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=10, labelsize=40,
                           width=2)

            ax.tick_params(axis="both", direction="in", which="minor", right=True, left=True,
                           bottom=True, top=True, size=6, width=1)
            ax2.tick_params(axis="both", direction="in", which="minor", right=True, left=True,
                           bottom=True, top=True, size=6, width=1)

            ax.xaxis.set_minor_locator(MultipleLocator(5))

            ax2.set_xticks([vmax-10, vmax, vmax+10, vmax+20, vmax+30, vmax+40])
            # ax2.set_xbound(ax.get_xbound())
            ax2.set_xticklabels([-10, 0, 10, 20, 30, 40])
            ax2.set_xlabel('Phase (days)', size=50)

            plt.legend(loc='upper right', ncol=2, prop={'size': 35})
            plt.axvline(vmax, color='grey', linewidth=5)
            # ax.axvline(vmax - dvmax, color='grey')
            # ax.axvline(vmax + dvmax, color='grey')
            ax.set_xlabel('JD - 2455000.5 (days)', size=50)
            ax.set_ylabel('Magnitude', size=50)

            plt.savefig("outputs/Plot_lc_%s_%s.png" % (sn, b))
