# Generate plots of the Ibc template versus typical and / atypical SNe

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
bands = ['B', 'V', 'R', 'I']


templates = {}

# Reading the GP templates in each of the given bands
for b in bands:

    templates[b] = {}

    path = os.getenv("SESNPATH") + "maketemplates/outputs/Ibc_template_files/UberTemplate_%s.pkl" %\
                                    (b + 'p' if b in ['u', 'r', 'i'] else b)
    templates_ = pkl.load(open(path, "rb"))

    templates[b] = templates_

    if np.nansum(templates_['med_smoothed']) == 0:
        print(b + ' band has no Ibc templates.')
        continue

#'#f7790a', '#36ff17','#0a4bff', '#f02244',
sns.reset_orig()  # get default matplotlib styles back
colors_atypicals = ["#3588d1", "#38485e", "#589e7e", "#c86949", 
                    "#881448", "#e8250c", "#621da6", "#cf80dd", "#0b522e", "#fe5cde"]
#sns.color_palette(cc.glasbey, n_colors=NUM_atypicals)
# su.allsne_colormaps(NUM_atypicals)
# colors_atypicals = su.allsne_colors

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

atypicals = ['SN1994I', 'SN2005kl', 'LSQ13ccw', 'SN1998bw', 'SN1993J']

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

            yerr = thissn.photometry[b]['dmag']

            y = y[x < 100]
            yerr = yerr[x < 100]
            x = x[x < 100]
            atypicals_phot[b]['x'].append(x)
            atypicals_phot[b]['y'].append(y)
            atypicals_phot[b]['yerr'].append(yerr)
            atypicals_phot[b]['name'].append(sn)
            atypicals_phot[b]['subtype'].append(thissn.type)


fig, axs = plt.subplots(2, 2, figsize=(35, 30),
                        sharey=True, sharex=True)

legsize = 55
labelsize = 65
plt.subplots_adjust(hspace=.04, wspace=0.04, bottom=0.08)

sn_temp = 0

n_panels = len(axs.flatten())
# handles, labels = [[]]*len(atypicals), [[]]*len(atypicals)

for i, b in enumerate(bands):
    index = i

    for j in range(len(atypicals_phot[b]['x'])):

        label_temp = str(atypicals_phot[b]['name'][j])+ ', ' +\
                                         atypicals_phot[b]['subtype'][j]

        axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                  atypicals_phot[b]['yerr'][j], fmt='.', ls='-',
                                  color=atypicals_colors[atypicals_phot[b]['name'][j]], linewidth=3)
        axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                  atypicals_phot[b]['yerr'][j], fmt='o', linewidth=5,
                                  color=atypicals_colors[atypicals_phot[b]['name'][j]],
                                  label=label_temp)

            


    axs.flatten()[index].plot(templates[b]['phs'], -1*templates[b]['med_smoothed'],
                          color='k', linewidth=6)
    axs.flatten()[index].fill_between(templates[b]['phs'], 
                                      -1*templates[b]['pc25_smoothed'], 
                                      -1*templates[b]['pc75_smoothed'],
                                      color='grey', alpha=0.5)
    axs.flatten()[index].set_xlim(-25, 105)
    axs.flatten()[index].set_ylim(-4.5, .5)
    axs.flatten()[index].text(0.8, 0.9, b, transform=axs.flatten()[index].transAxes, size=60)

handles = []
for i in range(len(axs.flatten())):
    handles += (axs.flatten()[i]).get_legend_handles_labels()[0]

labels = []
for i in range(len(axs.flatten())):
    labels += (axs.flatten()[i]).get_legend_handles_labels()[1]
if len(bands) == 7:
    fig.delaxes(axs.flatten()[1])
hl = [[], []]

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

if len(labels) < 6:
    legend_col_num = len(labels)
else:
    if len(labels) % 5 == 0:
        legend_col_num = 5
    elif len(labels) % 4 == 0:
        legend_col_num = 4
    elif len(labels) % 3 == 0:
        legend_col_num = 3
    else:
        legend_col_num = 3

fig.legend(hl[1], hl[0], loc='upper center', ncol=legend_col_num, prop={'size': legsize})

frame1 = fig.text(0.06, 0.5, 'Relative magnitude', va='center', rotation='vertical', size=labelsize)
frame2 = fig.text(0.45, 0.02, 'Phase (days)', va='center', size=labelsize)
# Artist.set_visible(frame1, False)
# Artist.set_visible(frame2, False)


for ax in axs.flatten():
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=12, labelsize=labelsize, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=7, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
plt.subplots_adjust(hspace=0, wspace=0, top=0.89)
plt.savefig(
    os.getenv("SESNPATH") + 'maketemplates/outputs/output_plots/atypicals_Ibc_in_%s_bands.pdf' % (len(bands)),
    bbox_inches='tight')
