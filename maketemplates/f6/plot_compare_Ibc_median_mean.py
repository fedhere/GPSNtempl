#############################################################################################################################
# A three-panel plot for each band.
# Panel 1: Plotting all lc labeled by type along with median template
# Panel 2: Plotting all lc labeled by type along with average template
# Panel 3: Plotting the median and average template
# The peak of the average templates is manually set to zero to align the templates vertically.

import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.ticker import AutoMinorLocator
from bokeh.models import ColumnDataSource

try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNPATH") + "/maketemplates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from ubertemplates import preplcvs, plotme2

bands = ['u', 'U', 'B', 'g', 'r', 'i', 'V', 'R', 'I', 'J', 'H', 'K', 'w2', 'm2', 'w1']
med_color = '#35618f'
mean_color = '#5c0d47'

colorTypes = {'IIb': 'FireBrick',
              'Ib': 'SteelBlue',
              'Ic': 'DarkGreen',
              'Ic-bl': 'DarkOrange',
              'Ibn': 'purple',
              'other': 'brown'}

maxy = 100
miny = -30

def doall(bands=bands):
    templates = {}
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") +
                   "/SESNessentials.csv", encoding="ISO-8859-1")['SNname'].values  # [:5]

    if os.path.isfile(os.getenv("SESNPATH") +'maketemplates/input/sncolors.pkl'):
        print('reading sncolors')
        with open(os.getenv("SESNPATH") +'maketemplates/input/sncolors.pkl', 'rb') as f:
            sncolors = pkl.load(f, encoding="latin")
        # sncolors =  pkl.load(open('maketemplates/input/sncolors.pkl'))

        if not len(sncolors) == len(inputSNe):
            print("redoing SNcolors")

            sncolors = setcolors(inputSNe)
    else:
        sncolors = setcolors(inputSNe)
    
        
    for b in bands:
        workBands = b


        allSNe = pkl.load(open(os.getenv("SESNPATH") +
                            'maketemplates/input/allSNe.pkl',
                            'rb'))

        fig, axs = plt.subplots(3, 1, figsize=(22, 20))

        source = ColumnDataSource(data={})
        source.data, axs = plotme2(allSNe[b], b, axs, sncolors)
        maxy = np.max(source.data['y'])
        miny = np.min(source.data['y'])



        templates[b] = {}

        path = os.getenv("SESNPATH") +\
               "maketemplates/outputs/Ibc_template_files/UberTemplate_%s.pkl" %\
                (b + 'p' if b in ['u', 'r', 'i'] else b)
        templates_ = pkl.load(open(path, "rb"))

        templates[b] = templates_

        if np.nansum(templates_['med_smoothed']) == 0:
            print(b + ' band has no Ibc templates.')
            continue


        phs = templates[b]['phs']
        med_ = templates[b]['med_smoothed']
        pc25_ = templates[b]['pc25_smoothed']
        pc75_ = templates[b]['pc75_smoothed']
        wmu = templates[b]['mu']
        wgstd = templates[b]['wgstd']
        std = templates[b]['std']
        med = templates[b]['med']
        pc75 = templates[b]['pc75']
        pc25 = templates[b]['pc25']

        

        axs[0].plot(phs, med_, color = med_color, lw=4, label=' Median')
        axs[0].fill_between(phs,
                            pc25_,
                            pc75_,
                            color = med_color, alpha=0.5)

        axs[1].plot(phs[std > 0], wmu[std > 0], color = mean_color, lw=4, label='Weighted average')
        axs[1].fill_between(phs[std > 0], wmu[std > 0] - wgstd[std > 0],
                            wmu[std > 0] + wgstd[std > 0],
                            color = mean_color, alpha=0.5)

        axs[2].plot(phs[std > 0], wmu[std > 0], color = mean_color, lw=4, label='Weighted average')

        axs[2].fill_between(phs[std > 0], wmu[std > 0] - wgstd[std > 0],
                            wmu[std > 0] + wgstd[std > 0],
                            color = mean_color, alpha=0.5)
        axs[2].plot(phs, med_, color = med_color, lw=4, label='Median')
        axs[2].fill_between(phs,
                            pc25_,
                            pc75_,
                            color=med_color, alpha=0.5)


        handles0, labels0 = axs[0].get_legend_handles_labels()
        handles1, labels1 = axs[1].get_legend_handles_labels()
        artistIIb = plt.Line2D((0, 1), (0, 0), color=colorTypes['IIb'],
                              marker='o', linestyle='')
        artistIb = plt.Line2D((0, 1), (0, 0), color=colorTypes['Ib'],
                             marker='o', linestyle='')
        artistIc = plt.Line2D((0, 1), (0, 0), color=colorTypes['Ic'],
                             marker='o', linestyle='')
        artistIcbl = plt.Line2D((0, 1), (0, 0), color=colorTypes['Ic-bl'],
                               marker='o', linestyle='')
        artistIbn = plt.Line2D((0, 1), (0, 0), color=colorTypes['Ibn'],
                              marker='o', linestyle='')
        artistother = plt.Line2D((0, 1), (0, 0), color=colorTypes['other'],
                                marker='o', linestyle='')


        handles = [artistIIb, artistIb, artistIc, artistIcbl, artistIbn, artistother] +\
                      [handle for handle in handles0]+[handle for handle in handles1]
        labels = ['IIb', 'Ib', 'Ic', 'Ic-bl', 'Ibn', 'other'] +\
                      [label for label in labels0]+[label for label in labels1]

        fig.legend(handles, labels, loc='upper center', ncol=4, prop={'size': 40})

        max_axs0 = [maxy, np.max(med[std > 0] + pc75[std > 0])]
        max_axs1 = [maxy, np.max(wmu[std > 0] + wgstd[std > 0])]
        max_axs2 = [np.max(wmu[std > 0] + wgstd[std > 0]), np.max(med[std > 0] + pc75[std > 0])]

        min_axs0 = [miny, np.min(med[std > 0] - pc25[std > 0])]
        min_axs1 = [miny, np.min(wmu[std > 0] - wgstd[std > 0])]
        min_axs2 = [np.min(wmu[std > 0] - wgstd[std > 0]), np.min(med[std > 0] - pc25[std > 0])]

        axs[0].set_xlim(-30, 100)
        axs[1].set_xlim(-30, 100)
        axs[2].set_xlim(-30, 100)
        axs[0].set_ylim(4, -1)
        axs[1].set_ylim(4, -1)
        axs[2].set_ylim(4, -1)
        axs[2].set_xlabel("phase (days since Vmax)", fontsize=50)
        axs[0].text(90, 0, b, size=50)
        fig.text(0.06, 0.5, 'Relative magnitude', va='center', rotation='vertical', size=50)

        for ax in (axs):
            ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=45,
                           width=2)
            ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=4, width=2)
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.set_yticks([ -1, 0, 1, 2, 3])
            ax.set_yticklabels([ '-1', '0', '1', '2', '3'])
            ax.set_xticks([-20, 0, 20, 40, 60, 80, 100])
            ax.set_xticklabels(['-20', '0', '20', '40', '60', '80', ''])
            ax.axvline(0, color = 'grey', alpha = 0.5)

        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        fig.savefig(os.getenv("SESNPATH") +"maketemplates/outputs/output_plots/UberTemplate_%s_types.pdf" % \
                    (b + 'p' if b in ['u', 'r', 'i']
                     else b), bbox_inches='tight')



if __name__ == '__main__':
    if len(sys.argv) > 1:
        doall(bands=[sys.argv[1]])
    else:
        doall()