import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

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


from matplotlib.ticker import AutoMinorLocator

# plt.rc('font', **font)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['font.weight'] = 'normal'

templates = {}


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

bands = ['u', 'U', 'B', 'g', 'r', 'i', 'V', 'R', 'I', 'J', 'H', 'K', 'w2', 'm2', 'w1']

for b in bands:

    templates[b] = {}

    path = os.getenv("SESNPATH") + "maketemplates/outputs/Ibc_template_files/UberTemplate_%s.pkl" %\
                                    (b + 'p' if b in ['u', 'r', 'i'] else b)
    templates_ = pkl.load(open(path, "rb"))

    templates[b] = templates_

    if np.nansum(templates_['med_smoothed']) == 0:
        print(b + ' band has no Ibc templates.')
        continue

b_c = 'V'
fig, axs = plt.subplots(5, 3, figsize=(25, 20), sharex=True, sharey=True)
colors = ['k', 'k', '#2166ac',
          '#1b7837', '#b2182b', '#542788',
          '#1b7837', '#b2182b', '#542788',
          '#c51b7d', '#f46d43', '#35978f',
          '#8c510a', '#bf812d', '#cc4c02']

for i, b in enumerate(templates.keys()):

    if i < 3:
        axs[0, i].plot(templates[b]['phs'], templates[b]['med_smoothed'], '-', color=colors[i])
        axs[0, i].plot(templates[b_c]['phs'], templates[b_c]['med_smoothed'], '--', color='g')
        axs[0, i].fill_between(templates[b]['phs'],
                               templates[b]['pc25_smoothed'],
                               templates[b]['pc75_smoothed'],
                               color=colors[i], alpha=0.3)
        axs[0, i].invert_yaxis()
        axs[0, i].text(98, 0.5, b, weight='bold', size=30)
    elif (i > 2) and (i < 6):
        axs[1, i - 3].plot(templates[b]['phs'], templates[b]['med_smoothed'], '-', color=colors[i])
        axs[1, i - 3].plot(templates[b_c]['phs'], templates[b_c]['med_smoothed'], '--', color='g')
        axs[1, i - 3].fill_between(templates[b]['phs'],
                                   templates[b]['pc25_smoothed'],
                                   templates[b]['pc75_smoothed'],
                                   color=colors[i], alpha=0.3)
        axs[1, i - 3].invert_yaxis()
        axs[1, i - 3].text(98, 0.5, b, weight='bold', size=30)

    elif (i > 5) and (i < 9):
        axs[2, i - 6].plot(templates[b]['phs'], templates[b]['med_smoothed'], '-', color=colors[i])
        axs[2, i - 6].plot(templates[b_c]['phs'], templates[b_c]['med_smoothed'], '--', color='g')
        axs[2, i - 6].fill_between(templates[b]['phs'],
                                   templates[b]['pc25_smoothed'],
                                   templates[b]['pc75_smoothed'],
                                   color=colors[i], alpha=0.3)
        axs[2, i - 6].invert_yaxis()
        axs[2, i - 6].text(98, 0.5, b, weight='bold', size=30)
    elif (i > 8) and (i < 12):
        axs[3, i - 9].plot(templates[b]['phs'], templates[b]['med_smoothed'], '-', color=colors[i])
        axs[3, i - 9].plot(templates[b_c]['phs'], templates['V']['med_smoothed'], '--', color='g')
        axs[3, i - 9].fill_between(templates[b]['phs'],
                                   templates[b]['pc25_smoothed'],
                                   templates[b]['pc75_smoothed'],
                                   color=colors[i], alpha=0.3)
        axs[3, i - 9].invert_yaxis()
        axs[3, i - 9].text(98, 0.5, b, weight='bold', size=30)

    elif (i > 11) and (i < 15):
        axs[4, i - 12].plot(templates[b]['phs'], templates[b]['med_smoothed'], '-', color=colors[i])
        axs[4, i - 12].plot(templates[b_c]['phs'], templates[b_c]['med_smoothed'], '--', color='g')
        axs[4, i - 12].fill_between(templates[b]['phs'],
                                    templates[b]['pc25_smoothed'],
                                    templates[b]['pc75_smoothed'],
                                    color=colors[i], alpha=0.3)
        axs[4, i - 12].invert_yaxis()
        axs[4, i - 12].text(94, 0.5, b, weight='bold', size=30)
plt.ylim(4, -1)

for i, ax in enumerate(np.concatenate(axs)):
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=30, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=4, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    if i % 3 == 0:
        ax.set_yticks([-1, 0, 1, 2, 3, 4])
        ax.set_yticklabels(['', '0', '1', '2', '3', ''])


axs[4, 0].set_xticks([0, 50, 100])
axs[4, 0].set_xticklabels(['0', '50', '100'])


plt.subplots_adjust(hspace=0, wspace=0)

fig.text(0.5, 0.04, 'Phase (days)', ha='center', size=40)
fig.text(0.07, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=40)
fig.savefig(os.getenv("SESNPATH") + "maketemplates/outputs/output_plots/allbands_Ibc.pdf", bbox_inches='tight')
plt.savefig(output_directory + 'allbands_Ibc.pdf', bbox_inches='tight')

# Compare Ibc templates to D11 and T15

D11_V = np.loadtxt('lit_templates/D11_V_template.txt', skiprows=3)
D11_R = np.loadtxt('lit_templates/D11_R_template.txt', skiprows=3)
T15 = pd.read_csv('lit_templates/T15_templates.csv')

fig, axs = plt.subplots(2, 1, figsize=(22, 15), sharex=True)

V_max_ind = np.nanargmin(templates['V']['med_smoothed'][
                             (templates['V']['phs'] < max(D11_V[:, 0])) & (templates['V']['phs'] > min(D11_V[:, 0]))])
V_max_t = \
    templates['V']['phs'][(templates['V']['phs'] < max(D11_V[:, 0])) & (templates['V']['phs'] > min(D11_V[:, 0]))][
        V_max_ind]

axs[0].plot(D11_V[:, 0], D11_V[:, 1], '-', color='black', label='D11 V', linewidth=3)
axs[0].plot(templates['V']['phs'] - V_max_t,
            templates['V']['med_smoothed'],
            '-', color='#1b7837', linewidth=3, label='Ibc template')
axs[0].fill_between(templates['V']['phs'] - V_max_t,
                    templates['V']['pc25_smoothed'],
                    templates['V']['pc75_smoothed'],
                    color='#1b7837', alpha=0.3)
axs[0].invert_yaxis()
axs[0].legend(loc=1, prop={'size': 40})

R_max_ind = np.nanargmin(templates['R']['med_smoothed'][
                             (templates['R']['phs'] < max(D11_R[:, 0])) & (templates['R']['phs'] > min(D11_R[:, 0]))])
R_max_t = templates['R']['phs'][(templates['R']['phs'] < max(D11_R[:, 0])) &
                                (templates['R']['phs'] > min(D11_R[:, 0]))][R_max_ind]

axs[1].plot(D11_R[:, 0], D11_R[:, 1], '-', color='black', label='D11 R', linewidth=3)
axs[1].plot(templates['R']['phs'],
            templates['R']['med_smoothed'],
            '-', color='#b2182b', linewidth=3, label='Ibc template')
axs[1].fill_between(templates['R']['phs'],
                    templates['R']['pc25_smoothed'],
                    templates['R']['pc75_smoothed'],
                    color='#b2182b', alpha=0.3)

axs[1].legend(loc=1, prop={'size': 40})
axs[1].invert_yaxis()

for ax in (axs):
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=30, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=4, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['0', '1', '2', '3'])

axs[0].set_ylim(4, -0.5)
axs[1].set_ylim(4, -0.5)

fig.text(0.5, 0.0, 'Phase (days)', ha='center', size=40)
fig.text(0.05, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=40)
plt.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.getenv("SESNPATH") + "maketemplates/outputs/output_plots/compare_with_D11_4.pdf", bbox_inches='tight')
plt.savefig(output_directory + 'compare_with_D11_4.pdf', bbox_inches='tight')

T15['Mu'] = -2.5 * np.log10(T15['Fu'] / T15['Fu'][0])
T15['Mr'] = -2.5 * np.log10(T15['Fr'] / T15['Fr'][0])
T15['Mi'] = -2.5 * np.log10(T15['Fi'] / T15['Fi'][0])

fig, axs = plt.subplots(3, 1, figsize=(22, 20), sharex=True)

u_max_ind = np.nanargmin(templates['u']['med_smoothed'])
u_max_t = templates['u']['phs'][u_max_ind]
axs[0].plot(T15['Ep.u'], T15['Mu'] - min(T15['Mu']), '-', color='#2166ac', linewidth=3, label="T15 u")
axs[0].plot(templates['u']['phs'] - u_max_t,
            templates['u']['med_smoothed'],
            '-', color='k', linewidth=3, label='Ibc template')
axs[0].fill_between(templates['u']['phs'] - u_max_t,
                    templates['u']['pc25_smoothed'],
                    templates['u']['pc75_smoothed'],
                    color='k', alpha=0.3)

axs[0].invert_yaxis()
axs[0].legend(loc=1, prop={'size': 40})

r_max_ind = np.nanargmin(templates['r']['med_smoothed'])
r_max_t = templates['r']['phs'][r_max_ind]

axs[1].plot(T15['Ep.r'], T15['Mr'] - min(T15['Mr']), '-', color='#2166ac', linewidth=3, label="T15 r")
axs[1].plot(templates['r']['phs'] - r_max_t,
            templates['r']['med_smoothed'],
            '-', color='#b2182b', linewidth=3, label='Ibc template')
axs[1].fill_between(templates['r']['phs'] - r_max_t,
                    templates['r']['pc25_smoothed'],
                    templates['r']['pc75_smoothed'],
                    color='#b2182b', alpha=0.3)

axs[1].legend(loc=1, prop={'size': 40})
axs[1].invert_yaxis()

i_max_ind = np.nanargmin(templates['i']['med_smoothed'])
i_max_t = templates['i']['phs'][i_max_ind]

axs[2].plot(T15['Ep.i'], T15['Mi'] - min(T15['Mi']), '-', color='#2166ac', linewidth=3, label="T15 i")
axs[2].plot(templates['i']['phs'][:len(templates['i']['med_smoothed'])] - i_max_t, templates['i']['med_smoothed'], '-',
            color='#542788', linewidth=3,
            label='Ibc template')
# axs[2].plot(templates['i']['phs'][templates['i']['std']>0], templates['i']['wmu'][templates['i']['std']>0],'-', color='orange', label='Average')
axs[2].fill_between(templates['i']['phs'][:len(templates['i']['med_smoothed'])] - i_max_t,
                    templates['i']['pc25_smoothed'][:len(templates['i']['med_smoothed'])],
                    templates['i']['pc75_smoothed'][:len(templates['i']['med_smoothed'])],
                    color='#542788', alpha=0.3)
axs[2].legend(loc=1, prop={'size': 40})
axs[2].invert_yaxis()

for ax in (axs):
    ax.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=30, width=2)
    ax.tick_params(axis="both", direction="in", which="minor", right=True, top=True, size=4, width=2)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['0', '1', '2', '3'])
axs[0].set_ylim(4, -0.5)
axs[1].set_ylim(4, -0.5)
axs[2].set_ylim(4, -0.5)
fig.text(0.5, 0.05, 'Phase (days)', ha='center', size=40)
fig.text(0.07, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=40)
plt.subplots_adjust(hspace=0, wspace=0)
fig.savefig(os.getenv("SESNPATH") + "maketemplates/outputs/output_plots/compare_with_T15.pdf", bbox_inches='tight')
plt.savefig(output_directory + 'compare_with_T15.pdf', bbox_inches='tight')
