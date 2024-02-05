import os
import pickle as pkl
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    os.environ['SESNPATH']
except KeyError:
    print("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

SNTYPES = ['IIb', 'Ib', 'Ibn', 'Ic', 'Ic-bl']
bands = ['up', 'U', 'B', 'g', 'rp', 'ip', 'V', 'R', 'I', 'J', 'H', 'K', 'w2', 'm2', 'w1']

colorTypes = {'IIb': 'FireBrick',
              'Ib': 'SteelBlue',
              'Ic': 'DarkGreen',
              'Ic-bl': 'DarkOrange',
              'Ibn': 'purple'}

# plt.rc('font', **font)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['font.weight'] = 'normal'

tmpl = {}


def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-dir', action='store', default=None,
                        help='output_directory where the figures should be saved.')
    arguments = parser.parse_args()
    return arguments


args = parse_options()

# Define your output directory
if not args.dir is None:
    output_directory = args.dir
else:
    output_directory = './../../GPSNtempl_output/'

for bb in bands:

    tmpl[bb] = {}

    for SNTYPE in SNTYPES:

        tmpl[bb][SNTYPE] = {}

        try:
            path = os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
            tmpl_ = pkl.load(open(path, "rb"))
        except:
            print(SNTYPE, bb)
            continue
        tmpl[bb][SNTYPE] = tmpl_

bb = ['u', 'U', 'B', 'g', 'r', 'i', 'V', 'R', 'I']

up_lim = 100

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(30, 25))

for i, b in enumerate(bb):
    b_ = b
    for tp in SNTYPES:

        if b == 'i' or b == 'r' or b == 'u':
            # b_ = b+ str("'")
            b = b + str('p')

        if len(tmpl[b][tp].keys()) == 0:
            print("No data for %s in band %s" % (tp, b))
            continue

        np.concatenate(ax)[i].plot(tmpl[b][tp]['t'][tmpl[b][tp]['t'] < up_lim],
                                   tmpl[b][tp]['rollingMedian'][tmpl[b][tp]['t'] < up_lim],
                                   color=colorTypes[tp], label=tp, linewidth=4)
        np.concatenate(ax)[i].fill_between(tmpl[b][tp]['t'][tmpl[b][tp]['t'] < up_lim],
                                           tmpl[b][tp]['rollingPc25'][tmpl[b][tp]['t'] < up_lim],
                                           tmpl[b][tp]['rollingPc75'][tmpl[b][tp]['t'] < up_lim],
                                           color=colorTypes[tp], alpha=0.2)
        np.concatenate(ax)[i].tick_params(axis="both", direction="in", which="major",
                                          right=True, top=True, size=7, labelsize=35, width=2)

    # np.concatenate(ax)[i].legend(ncol=2,loc = 'lower left', prop={'size': 30})
    np.concatenate(ax)[i].grid()
    np.concatenate(ax)[i].text(0.9, 0.9, b_, transform=np.concatenate(ax)[i].transAxes, size=40)

    np.concatenate(ax)[i].tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7,
                                      labelsize=40, width=2)
fig.text(0.5, 0.04, 'Phase (Days)', ha='center', size=50)
fig.text(0.04, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=50)

handlesDic = {}
for ax_ in np.concatenate(ax):
    for item in np.array(ax_.get_legend_handles_labels()).T:
        handlesDic[item[1]] = item[0]
handles = []
labels = []
for i, tp in enumerate(SNTYPES):
    if tp not in list(handlesDic.keys()):
        continue
    handles.append(handlesDic[tp])
    labels.append(tp)
fig.legend(handles, labels, loc='upper center', ncol=5, prop={'size': 55})

plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=0.91, wspace=0.04, hspace=0.04)

fig.savefig(os.getenv("SESNPATH") + "maketemplates/outputs/GP_tmpl_per_band_UVRIugri.pdf", bbox_inches='tight')
fig.savefig(output_directory + "GP_tmpl_per_band_UVRIugri.pdf", bbox_inches='tight')

bb = {'w2': 1,
      'm2': 1,
      'w1': 1,
      'J': 1,
      'H': 1,
      'K': 1}
up_lim = 100
bc = len(bb.keys())
bb_selected = []
for i, b in enumerate(bb.keys()):
    c = 0
    for tp in SNTYPES:
        if len(tmpl[b][tp].keys()) == 0:
            print("No data for %s in band %s" % (tp, b))
            continue
        c += 1
    if c == 0:
        bb[b] = 0
        bc -= 1
    else:
        bb_selected.append(b)
n_subplots = bc
fig, ax = plt.subplots(1, n_subplots, sharex=True, sharey=True, figsize=(40, 12))

for i, b in enumerate(bb_selected):
    b_ = b
    for tp in SNTYPES:
        if len(tmpl[b][tp].keys()) == 0:
            print("No data for %s in band %s" % (tp, b))
            continue

        ax[i].plot(tmpl[b][tp]['t'][tmpl[b][tp]['t'] < up_lim],
                   tmpl[b][tp]['rollingMedian'][tmpl[b][tp]['t'] < up_lim],
                   color=colorTypes[tp], label=tp, linewidth=4)
        ax[i].fill_between(tmpl[b][tp]['t'][tmpl[b][tp]['t'] < up_lim],
                           tmpl[b][tp]['rollingPc25'][tmpl[b][tp]['t'] < up_lim],
                           tmpl[b][tp]['rollingPc75'][tmpl[b][tp]['t'] < up_lim],
                           color=colorTypes[tp], alpha=0.2)

    ax[i].grid()
    ax[i].text(0.9, 0.9, b_, transform=ax[i].transAxes, size=50)
    ax[i].tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7,
                      labelsize=45, width=2)
    ax[i].set_ylim([-3.5, 1])

handlesDic = {}
for ax_ in ax:
    for item in np.array(ax_.get_legend_handles_labels()).T:
        handlesDic[item[1]] = item[0]
handles = []
labels = []
for i, tp in enumerate(SNTYPES):
    if tp not in list(handlesDic.keys()):
        continue
    handles.append(handlesDic[tp])
    labels.append(tp)


fig.legend(handles, labels, loc='upper center', ncol=5, prop={'size': 55})

fig.text(0.5, 0.04, 'Phase (Days)', ha='center', size=55)
fig.text(0.04, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=55)

plt.subplots_adjust(left=0.08, bottom=0.15, right=None, top=0.8, wspace=0.04, hspace=0.04)

fig.savefig(os.getenv("SESNPATH") + "maketemplates/outputs/GP_tmpl_per_band_JKHw1w2m2.pdf", bbox_inches='tight')
fig.savefig(output_directory + "GP_tmpl_per_band_JKHw1w2m2.pdf", bbox_inches='tight')
