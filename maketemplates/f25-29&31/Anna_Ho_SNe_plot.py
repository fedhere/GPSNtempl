# Generate a single plot to show Anna Ho's light curves with our templates
# Arguments: This is plotted automatically
# List of the SNe:
# Anna Ho's Ib's: Ib SN=SN2019dge,SN2018ghd b=g,r,i
# Anna Ho's Ic's: Ic SN=SN2020oi b=g,r,i
# Anna Ho's Ic-bl's: Ic-bl SN=SN2018gep b=g,r,i
# Anna Ho's IIb's: IIb SN=SN2020ano,SN2020rsc,SN2019rta,SN2018gjx,SN2020ikq,SN2020xlt b=g,r,i
import os
import pickle as pkl
import sys
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import colorcet as cc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

import itertools

try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import snclasses as snstuff
from myastrotools import *
import templutils as templutils

su = templutils.setupvars()
coffset = su.coffset

# We are plotting the atypical plots in these four bands:
bands = ['g', 'r', 'i']
atypicals = ['SN2018ghd', 'SN2019dge','SN2020oi', 'SN2018gep', 
             'SN2020ano', 'SN2020ikq', 'SN2020rsc', 
             'SN2018gjx', 'SN2019rta', 'SN2020xlt']
tmpl = {}
row_subtypes = ['IIb', 'Ib', 'Ic', 'Ic-bl']

# Reading the GP templates in each of the given bands
for bb in bands:
    tmpl[bb] = {}
    for SNTYPE in row_subtypes:

        
        tmpl[bb][SNTYPE] = {}

        # if bb in ['u', 'i', 'r']:
        #     bb = bb + 'p'

        path = os.getenv("SESNPATH") + "maketemplates/outputs/GP_template_files/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
        tmpl_ = pkl.load(open(path, "rb"))

        if np.nansum(tmpl_['rollingMedian']) == 0:
            print(bb, SNTYPE)
            continue

        tmpl[bb][SNTYPE] = tmpl_



sns.reset_orig()  # get default matplotlib styles back
colors_atypicals = [
                    # "#e51f00",
                    # "#00c2f2",
                    # "#589e7e", #light green
                    # "#cf4343", #medium red
                    # "#ffaa00", 
                    # "#59b359", 
                    # "#c48747", #dirty  orange
                    # "#99ba4e", #very bright green
                    # "#cf4343", #medium red
                    # "#dcb643", #dark yellow
                    # "#c77a9d", #light purple
                    # "#117442", #dark green
                    # "#5990da", #very light blue,
                    # "#6f34a9", #violet
                    # "#e51f00", 
                    # "#ffaa00", 
                    # "#59b359", 
                    # "#00c2f2", 
                    # "#f2b6de",
                    # "#502db3"
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#ff7f00",
                    "#dedd17", 
                    ]
# [ '#f7790a', '#36ff17','#0a4bff',
                     # '#f02244', '#755405', '#07b368', 
                     # '#ff24e2', '#fff024', '#14aae0', '#660944']

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

atypicals_colors = {} # A set of colors for plotting the atypicals.
handles = []
for i, n in enumerate(atypicals):
    atypicals_colors[n] = colors_atypicals[i]


atypicals_phot = {} # All atypical photometry will be saved here

for b in bands:
    atypicals_phot[b] = {'x': [], 'y': [], 'yerr': [], 'name': [], 'subtype': []}

for i, sn in enumerate(atypicals):
    print(sn)

    #     meansmooth = lambda x : -tmpl['spl_med'](x) + tmpl['spl_med'](0)
    # Each atypical photometry is read here
    thissn = snstuff.mysn(sn, addlit=True)
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    thissn.printsn()
    lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
    thissn.setphot()
    thissn.getphot()
    thissn.setphase()
    thissn.sortlc()
    if np.isnan(thissn.Vmax):
        # If there are no known Vmax for that SN, setVmaxFromFile function will look at peaks in other 
        # bands in a given input file and converts them to Vmax.
        input_file = pd.read_csv(os.getenv("SESNCFAlib") + \
                                 "/SESNessentials_large_table.csv", encoding="ISO-8859-1")
        thissn.setVmaxFromFile(input_file, verbose=False, earliest=False,
                               loose=True, D11=False,
                               bigfile=False, quiet=False)

    # Manually giving the handles color
    # if len(thissn.type)>5:
    #     # print('yess',thissn.type)
    #     handles.append(mpatches.Patch(color=colors_atypicals[i], label=sn + ', ' + thissn.type))
    # else:
    # handles.append(mpatches.Patch(color=colors_atypicals[i], label=sn+ ', ' + thissn.type))


    for b in bands:

        # thissn.getmagmax(band=b, forceredo=True)

        if thissn.filters[b] != 0:


            # If time is not in MJD, fix it!
            xmin = thissn.photometry[b]['mjd'].min()
            t_max = thissn.Vmax

            if xmin - t_max < -1000:
                x = thissn.photometry[b]['mjd'] - t_max + 2400000.5

            elif xmin - t_max > 1000:
                x = thissn.photometry[b]['mjd'] - t_max - 2400000.5
            else:
                x = thissn.photometry[b]['mjd'] - t_max

            y = thissn.photometry[b]['mag']

            if np.isnan(t_max):
                # If no Vmax was found, read in the GP fit of that light curve
                y = -y
                pklf = os.getenv("SESNPATH") + "./../GPSNtempl_output/outputs/GPs_2022/GPfit%s_%s.pkl" % (
                    sn, b + 'p' if b in ['u', 'r', 'i']
                    else b)
                if not os.path.isfile(pklf):
                    # If the SN has no GP fit:
                        # First take the minimum y as the peak
                        # If t_ymin>5, it means the peak is not covered, so skip that band.
                        # If t_ymin<-5, it might be a pre-shock peak chosen, look for a peak
                        # in [-5,5] interval
                    print("missing GP file ", pklf)
                    min_y = np.min(y)
                    if x[y == min_y][0] > 5:
                        print('5 Removed sn', sn)
                        continue
                    elif x[y == min_y][0] < -5:
                        min_y = np.min(y[(x > -5) & (x < 5)])
                    y = min_y - y
                    # continue
                    pass

                # If the SN has GP fit, take the peak of that as the peak if 
                ygp, gp, tmplm = pkl.load(open(pklf, "rb"))
                if x[0] > 0:
                    delta_y = y[0] + tmplm(x)[0]
                    y = y - delta_y
                else:
                    # Fixing the vertical alignment issue for those SNe with a pre-shock
                    # if min(np.abs(x)) < 2:
                    #     y_min = y[np.argmin(np.abs(x))]
                    #     y = y - y_min
                    # else:
                    y_min = y[np.argmin(np.abs(x))] + tmplm(x)[np.argmin(np.abs(x))]
                    y = y - y_min
            else:
                if not isinstance(x[y == min(y)], float):
                    xmin = x[y == np.nanmin(y)][0]
                else:
                    xmin = x[y == np.nanmin(y)]
                if xmin > 5:
                    continue
                if xmin < -5:
                    # y = y[np.abs(x) == np.min(np.abs(x))] -y
                    try:
                        min_y = np.min(y[(x > -5) & (x < 5)])
                    except:
                        print(sn, b, 'no data between -5 and 5 days')
                        continue
                else:
                    min_y = np.min(y)

                y = min_y - y

            if sn == 'SN2020rsc':
                y = 20.517 - thissn.photometry[b]['mag']
                x = thissn.photometry[b]['mjd'] - 2459100.35
                print('SN2020rsc', x, y)

            yerr = thissn.photometry[b]['dmag']

            y = y[x < 100]
            yerr = yerr[x < 100]
            x = x[x < 100]
            atypicals_phot[b]['x'].append(x)
            atypicals_phot[b]['y'].append(y)
            atypicals_phot[b]['yerr'].append(yerr)
            atypicals_phot[b]['name'].append(sn)
            atypicals_phot[b]['subtype'].append(thissn.type)


fig, axs = plt.subplots(4, 3, figsize=(17*3, 15*4),
                        sharey=True, sharex=True)


legsize = 50
labelsize = 70
plt.subplots_adjust(hspace=.04, wspace=0.04, bottom=0.08)

sn_temp = 0

n_panels = len(axs.flatten())
# handles, labels = [[]]*len(atypicals), [[]]*len(atypicals)

for k in range(4):
    handles = []
    labels = []
    hl = [[], []]
    SNTYPE = row_subtypes[k]
    for i, b in enumerate(bands):    


        for j in range(len(atypicals_phot[b]['x'])):
            if not atypicals_phot[b]['subtype'][j] == row_subtypes[k]:
                continue
            

            label_temp = str(atypicals_phot[b]['name'][j])

            axs[k, i].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                      atypicals_phot[b]['yerr'][j], fmt='.', ls='-',
                                      color=atypicals_colors[atypicals_phot[b]['name'][j]], linewidth=3)
            axs[k, i].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                      atypicals_phot[b]['yerr'][j], fmt='o', linewidth=5,
                                      color=atypicals_colors[atypicals_phot[b]['name'][j]],
                                      label=label_temp)


        axs[k, i].plot(tmpl[b][SNTYPE]['t'], tmpl[b][SNTYPE]['rollingMedian'],
                              color='k', linewidth=6)
        axs[k, i].fill_between(tmpl[b][SNTYPE]['t'], tmpl[b][SNTYPE]['rollingPc75'], tmpl[b][SNTYPE]['rollingPc25'],
                                      color='grey', alpha=0.5)
        axs[k, i].set_xlim(-25, 105)
        axs[k, i].set_ylim(-4.5, .5)
        if k == 3:
            axs[k, i].set_ylim(-4.5, 1)
        axs[k, i].text(0.8, 0.9, b + ', ' + SNTYPE, transform=axs[k, i].transAxes, size=60)

        
    for i in range(len(axs[k, :].flatten())):
        handles += (axs[k, :].flatten()[i]).get_legend_handles_labels()[0]

    
    for i in range(len(axs[k, :].flatten())):
        labels += (axs[k, :].flatten()[i]).get_legend_handles_labels()[1]


    dups = duplicates(labels)
    dups_indx = duplicates_indices(labels)
    for i, label in enumerate(labels):
        if label in dups:
            if not np.isnan(dups_indx[label][0]):
                hl[0].append(label)
                hl[1].append(handles[i])
                dups_indx[label][0] = np.nan
            else:
                continue
        else:
            hl[0].append(label)
            hl[1].append(handles[i])

    legend_col_num = 2

    axs[k, -1].legend(hl[1], hl[0], 
                      loc='lower center', 
                      ncol=legend_col_num, 
                      prop={'size': legsize})

frame1 = fig.text(0.08, 0.5, 'Relative magnitude', va='center', rotation='vertical', size=labelsize)
frame2 = fig.text(0.45, 0.05, 'Phase (days)', va='center', size=labelsize)


for ax in axs.flatten():
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=12, labelsize=labelsize, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=7, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.subplots_adjust(hspace=0, wspace=0, top=0.94)
plt.savefig(
    os.getenv("SESNPATH") + 'maketemplates/outputs/output_plots/all_Anna_Ho.pdf',
    bbox_inches='tight')
