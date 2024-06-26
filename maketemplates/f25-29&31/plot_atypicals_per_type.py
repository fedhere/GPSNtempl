# Generate plots per SESN subtype to show the GP template for that subtype versus individual SESN
# Arguments: subtype, list of individual SESN separated by comma
# Examples:
# Ib SN=SN2015ap,SN2007uy,SN2009er,SN2007ke,PTF11kmb,SN2010et b=B,V,R
# Ibn SN=SN2011hw,SN2015U,LSQ13ccw,OGLE-2012-SN-006,OGLE-2014-SN-131,PS1-12sk,sn2010al b=B,g,V,r,R,i,I
# IIb SN=SN2013df,SN2010as,SN2011fu,SN2011hs
# Ic SN=SN2013ge,iPTF15dtg,LSQ14efd,SN2017ein,SN2012hn,PTF12gzk,SN2003id,SN1994I
# Ic-bl SN=SN2013cq,SN2010bh,SN2009bb,SN2006aj
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
from matplotlib.ticker import MaxNLocator

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

if __name__ == '__main__':

    if len(sys.argv) > 1:
        # It takes as an argument the subtype. If none given, subtype Ib is plotted
        if sys.argv[1] in ['Ib', 'IIb', 'Ic', 'Ic-bl', 'Ibn']:
            SNTYPE = sys.argv[1]
        else:
            SNTYPE = 'Ib'
        if len(sys.argv) > 2:
            # Here, individual atypical SN names are given. This is REQUIRED.
            if sys.argv[2].startswith('SN='):
                atypicals = sys.argv[2].split('=')[1].split(',')
                NUM_atypicals = len(atypicals)
            if sys.argv[3].startswith('b='):
                bands = sys.argv[3].split('=')[1].split(',')
                next_arg = 4
            else:
                next_arg = 3

            if sys.argv[next_arg].startswith('top='):
                top_num = sys.argv[next_arg].split('=')[1]
                top_num = float(top_num)
            if sys.argv[next_arg+1].startswith('name_save='):
                name_save = sys.argv[next_arg+1].split('=')[1]
                # name_save = str(name_save)
            else:
                print('Select individual SESN to plot by typing: SN=SN1,SN2,SN3,...')
                sys.exit()

tmpl = {}
# Reading the GP templates in each of the given bands
for bb in bands:

    tmpl[bb] = {}
    tmpl[bb][SNTYPE] = {}

    # if bb in ['u', 'i', 'r']:
    #     bb = bb + 'p'

    path = os.getenv("SESNPATH") + "maketemplates/outputs/GP_template_files/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
    tmpl_ = pkl.load(open(path, "rb"))

    if np.nansum(tmpl_['rollingMedian']) == 0:
        print(bb, SNTYPE)
        continue

    tmpl[bb][SNTYPE] = tmpl_
#'#f7790a', 'DarkGreen','SteelBlue', '#f02244',
sns.reset_orig()  # get default matplotlib styles back
colors_atypicals = [
                    # "#6f34a9", #violet
                    # "#0d6bbe", #blue
                    # "#c48747", #dirty  orange
                    # "#cf4343", #medium red
                    # "#99ba4e", #very bright green
                    # "#dcb643", #dark yellow
                    # "#c77a9d", #light purple
                    # "#117442", #dark green
                    # "#91b0da", #very light blue,
                    # "#589e7e", #light green
                    # "#e41a1c",
                    # "#377eb8",
                    # "#4daf4a",
                    # "#984ea3",
                    "#e41a1c",
                    "#377eb8",
                    "#4daf4a",
                    "#984ea3",
                    "#ff7f00",
                    "#dedd17", 
                    "#a65628",
                    "#f781bf"
                    ]
# ['#f7790a', 'DarkGreen','SteelBlue', '#f02244', 
                    # '#755405', '#07b368', '#ff24e2', '#fff024', '#14aae0', '#660944']
# 
# ['#ff0505', '#038041', '#ff8585', '#bf6204', '#0582ff',
#                     '#bfbf63', '#7f4242', '#8205ff','#222280', '#ffff05']
#sns.color_palette(cc.glasbey, n_colors=NUM_atypicals)
# su.allsne_colormaps(NUM_atypicals)
# colors_atypicals = su.allsne_colors

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

        if thissn.filters[b] >3:


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
                # if xmin > 5:

                #     continue
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
                    

            if sn == 'OGLE-2012-SN-006' and (b == 'B' or b == 'R'):
                continue

            if sn == 'SN2007ke' and (b == 'V'):
                continue
            if sn == 'SN2017ein' and (b == 'I'):
                continue
            if sn == 'SN2007ke' and (b == 'B'):
                continue

            if sn == 'OGLE-2012-SN-006' and (b == 'I'):
                df = pd.read_csv(os.getenv(
                    "SESNPATH") + '/maketemplates/data/Lightcurve_modifications/OGLE-2012-SN-006/' + sn + '_modified.csv')

                for l in [2, 7]:
                    x = df.t[(df.b == 'I') & (df.s == l)].values
                    y = df.m[(df.b == 'I') & (df.s == l)].values
                    yerr = df.e_m[(df.b == 'I') & (df.s == l)].values

                    ymin = df.m[(df.b == 'I') & (df.s == 2)].values.min()
                    xmin = df.m[(df.b == 'I') & (df.s == 2)].values[np.argmin(df.m[(df.b == 'I') & (df.s == 2)].values)]

                    y = y.min() - y
                    x = x - x[np.argmax(y)]

                    y = y[x < 100]
                    yerr = yerr[x < 100]
                    x = x[x < 100]

                    atypicals_phot[b]['x'].append(x)
                    atypicals_phot[b]['y'].append(y)
                    atypicals_phot[b]['yerr'].append(yerr)
                    atypicals_phot[b]['name'].append(sn)
                    atypicals_phot[b]['subtype'].append('Ibn')

            else:
                y = y[x < 100]
                yerr = yerr[x < 100]
                x = x[x < 100]
                atypicals_phot[b]['x'].append(x)
                atypicals_phot[b]['y'].append(y)
                atypicals_phot[b]['yerr'].append(yerr)
                atypicals_phot[b]['name'].append(sn)
                atypicals_phot[b]['subtype'].append(thissn.type)
            

if len(bands) == 1:
    fig, axs = plt.figure(15, 10)
    legsize = 55
    labelsize = 70
    plt.subplots_adjust(hspace=.04, wspace=0.04)
elif len(bands) < 4:
    fig, axs = plt.subplots(1, len(bands), figsize=(60, 20), sharey=True, sharex=True)
    legsize = 60
    labelsize = 70
    plt.subplots_adjust(hspace=.04, wspace=0.04)
else:
    fig, axs = plt.subplots(int(round(len(bands) / 2, 0)), 2, figsize=(50, 15 * int(round(len(bands) / 2, 0))),
                            sharey=True, sharex=True)

    subplot_indexes = {'B': 0, 'u': 1,
                       'g': 2, 'V': 3,
                       'r': 4, 'R': 5,
                       'i': 6, 'I': 7}
    legsize = 55
    labelsize = 65
    plt.subplots_adjust(hspace=.04, wspace=0.04, bottom=0.08)

sn_temp = 0

n_panels = len(axs.flatten())
# handles, labels = [[]]*len(atypicals), [[]]*len(atypicals)

for i, b in enumerate(bands):
    if len(bands) == 7:
        index = subplot_indexes[b]
    else:
        index = i

    for j in range(len(atypicals_phot[b]['x'])):

        if (b == 'I') and (atypicals_phot[b]['name'][j] == 'OGLE-2012-SN-006'):

            if sn_temp == 0:

                if len(atypicals_phot[b]['subtype'][j])>5:
                    label_temp = str(atypicals_phot[b]['name'][j])+ ', ' +\
                                             atypicals_phot[b]['subtype'][j]
                else:
                    label_temp = atypicals_phot[b]['name'][j]

                axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                          atypicals_phot[b]['yerr'][j], fmt='.', ls='-',
                                          color=atypicals_colors[atypicals_phot[b]['name'][j]], linewidth=5)
                axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                          atypicals_phot[b]['yerr'][j], fmt='o', linewidth=5,
                                          color=atypicals_colors[atypicals_phot[b]['name'][j]],
                                          label=label_temp)
                sn_temp = sn_temp + 1
            else:

                axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                          atypicals_phot[b]['yerr'][j], fmt='.', ls='-',
                                          color=atypicals_colors[atypicals_phot[b]['name'][j]], linewidth=5)
                axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                          atypicals_phot[b]['yerr'][j], fmt='o', linewidth=5,
                                          color=atypicals_colors[atypicals_phot[b]['name'][j]])
        else:

            if len(atypicals_phot[b]['subtype'][j])>5:
                label_temp = str(atypicals_phot[b]['name'][j])+ ', ' +\
                                             atypicals_phot[b]['subtype'][j]
            else:
                label_temp = str(atypicals_phot[b]['name'][j])

            axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                      atypicals_phot[b]['yerr'][j], fmt='.', ls='-',
                                      color=atypicals_colors[atypicals_phot[b]['name'][j]], linewidth=5)
            axs.flatten()[index].errorbar(atypicals_phot[b]['x'][j], atypicals_phot[b]['y'][j],
                                      atypicals_phot[b]['yerr'][j], fmt='o', linewidth=5,
                                      color=atypicals_colors[atypicals_phot[b]['name'][j]],
                                      label=label_temp)


    axs.flatten()[index].plot(tmpl[b][SNTYPE]['t'], tmpl[b][SNTYPE]['rollingMedian'],
                          color='k', linewidth=8)
    axs.flatten()[index].fill_between(tmpl[b][SNTYPE]['t'], tmpl[b][SNTYPE]['rollingPc75'], tmpl[b][SNTYPE]['rollingPc25'],
                                  color='grey', alpha=0.5)
    axs.flatten()[index].set_xlim(-25, 105)
    axs.flatten()[index].set_ylim(-3, .5)
    axs.flatten()[index].text(0.8, 0.9, b + ', ' + SNTYPE, transform=axs.flatten()[index].transAxes, size=60)

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
        legend_col_num = 4
    else:
        legend_col_num = 3

fig.legend(hl[1], hl[0], loc='upper center', ncol=legend_col_num, prop={'size': legsize})

frame1 = fig.text(0.07, 0.5, 'Relative magnitude', va='center', rotation='vertical', size=labelsize)
frame2 = fig.text(0.45, 0.02, 'Phase (days)', va='center', size=labelsize)
# Artist.set_visible(frame1, False)
# Artist.set_visible(frame2, False)


for ax in axs.flatten():
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=12, labelsize=labelsize, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=7, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(prune='both'))

plt.subplots_adjust(hspace=0, wspace=0, top=top_num)
plt.savefig(
    os.getenv("SESNPATH") + 'maketemplates/outputs/output_plots/atypicals_GP_%s_in_%s_bands_%s.pdf' % (SNTYPE, len(bands), name_save),
    bbox_inches='tight')
