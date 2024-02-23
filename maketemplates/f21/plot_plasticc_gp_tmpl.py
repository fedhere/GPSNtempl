import os
import pickle as pkl
import sys
import matplotlib.pyplot as plt
from select_lc import *
from Functions import *
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


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

from snclasses import *
import templutils as templutils



SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']

bands = ['R','V','r','g','U','u','J','B','H','I','i','K','m2','w1','w2']
colorTypes = {'IIb':'FireBrick',
             'Ib':'SteelBlue',
             'Ic':'DarkGreen',
             'Ic-bl':'DarkOrange',
             'Ibn':'purple'}
colors = ['SteelBlue', 'g',  'red','purple']

lsst_bands = {'0':'u',
              '1':'g',
              '2': 'r',
              '3': 'i',
              '4': 'z',
              '5': 'y'}

clrs =  {'0':'b',
              '1':'g',
              '2': 'r',
              '3': 'purple',
              '4': 'cyan',
              '5': 'k'}
su = templutils.setupvars()
coffset = su.coffset

ref = coffset['r']
for b in coffset.keys():
    coffset[b] = coffset[b] - ref

# Reading in the GP templates

tmpl = {}

for bb in bands:

    tmpl[bb] = {}

    for SNTYPE in SNTYPES:

        tmpl[bb][SNTYPE] = {}

        try:
            path = os.getenv("SESNPATH") + "maketemplates/outputs/GP_template_files/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
            tmpl_ = pkl.load(open(path, "rb"))
        except:
            continue

        if np.nansum(tmpl_['rollingMedian']) == 0:
            print(bb, SNTYPE)
            continue

        tmpl[bb][SNTYPE] = tmpl_


# Read in Plasticc train set
df1 = pd.read_csv(os.getenv("SESNPATH") + 'maketemplates/Plasticc/plasticc_train_lightcurves.csv')
df2 = pd.read_csv(os.getenv("SESNPATH") + 'maketemplates/Plasticc/plasticc_train_metadata.csv')


# Select SESNe (code 62) and redshifts<0.2

SN_Ibc_id = df2.object_id[(df2.true_target == 62) & (df2.true_z <= 0.2)].values
df1 = df1[df1['object_id'].isin(SN_Ibc_id)]
df1 = df1.reset_index(drop = True)

df2 = df2[(df2.true_target == 62) & (df2.true_z <= 0.2)]

# Select light curves where the minimum SNR is 10 in r band
bn = 2
high_SN = df1.object_id[df1.passband == bn]\
                       [df1.flux[df1.passband == bn]/df1.flux_err[df1.passband == bn] > 10].unique()
len(high_SN)

# Here we check how the total number of selected lc change when time_around_peak_limit changes

ID_selected = []
bn = 2
tot = len(high_SN)
time_around_peak_limit = 10

for i, ID in enumerate(high_SN):

    df_tmp = df1[(df1.passband == bn) & (df1.object_id == ID)]

    df_tmp = df_tmp[df_tmp.flux > 0]

    t = df_tmp.mjd.values
    f = df_tmp.flux.values
    ferr = df_tmp.flux_err.values

    t_peak = df2.true_peakmjd[df2.object_id == ID].values

    if len(t[(t > t_peak - time_around_peak_limit) & (t < t_peak + time_around_peak_limit)]) < 2:
        tot -= 1
        continue

    low_lim = -50
    up_lim = 100

    ind = (t < up_lim + t_peak) & (t > low_lim + t_peak)

    y = f[ind]
    yerr = ferr[ind]
    x = t[ind]

    m = 27.5 - 2.5 * np.log10(y)
    merr = 2.5 / np.log(10) * yerr / y

    tt = np.linspace(x.min(), x.max(), 1000)

    if len(y) < 4:
        tot -= 1
        continue

    interpld = interp1d(x, y, kind='nearest')(tt)
    #     t_new, func, p0 = lc_fit(np.row_stack((x, f[ind], ferr[ind])), x_peak = x_peak)
    m_func = 27.5 - 2.5 * np.log10(interpld)

    ind_ymin = (tt - t_peak < time_around_peak_limit) & (tt - t_peak > -time_around_peak_limit)

    if len(m_func[ind_ymin]) == 0:
        tot -= 1
        continue
    ID_selected.append(ID)

    ymin = np.min(m_func[ind_ymin])


print('Total number of good lc is ', tot)

# ugrizy = 012345
bb = ['u', 'g', 'r', 'i']
ID = 26842116
b_ = [0, 1, 2, 3]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


# u, g, r, i

max_u = [[], []]
max_g = [[], []]
max_r = [[], []]
max_i = [[], []]

# plt.rcParams['text.usetex'] = True
# plt.rcParams[
#     'text.latex.preamble'] = r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'

# print(band_sntypes)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(25, 25))

diff_u, diff_g, diff_r, diff_i = [], [], [], []

for b in b_:
    band_sntypes = [*tmpl[bb[b]]]

    for i, ID in enumerate(ID_selected):

        df_new = df1[(df1.object_id == ID) & (df1.passband == b)]
        #         df_r = df1[(df1.object_id == ID) & (df1.passband == 2)].reset_index(drop=True)

        df_new = df_new[df_new.flux > 0]

        t = df_new.mjd.values
        f = df_new.flux.values
        ferr = df_new.flux_err.values

        t_peak = df2.true_peakmjd[df2.object_id == ID].values

        low_lim = -50
        up_lim = 100

        ind = (t < up_lim + t_peak) & (t > low_lim + t_peak)

        y = f[ind]
        yerr = ferr[ind]
        x = t[ind]

        m = 27.5 - 2.5 * np.log10(y)
        merr = 2.5 / np.log(10) * yerr / y

        if len(y) < 4:
            if bb[b] == 'r':

                print(ID, bb[b])
            continue

        tt = np.linspace(x.min(), x.max(), 1000)

        interpld = interp1d(x, y, kind='cubic')(tt)
        #     t_new, func, p0 = lc_fit(np.row_stack((x, f[ind], ferr[ind])), x_peak = x_peak)
        m_func = 27.5 - 2.5 * np.log10(interpld)

        ind_ymin = (tt - t_peak < time_around_peak_limit) & (tt - t_peak > -time_around_peak_limit)

        if len(m_func[ind_ymin]) == 0:
            tot -= 1
            continue

        ymin = np.nanmin(m_func[ind_ymin])

        #         print(ID, bb[b])
        #         print(np.min(ymin - m), np.max(ymin - m))
        #         print(np.min(x - t_peak), np.max(x - t_peak))
        #         print('\n')

        if i == 0:

            np.concatenate(ax)[b].errorbar(x - t_peak, ymin - m, yerr=merr, fmt='o', \
                                           linewidth=3, color='k', alpha = 0.1,
                                           label='PLAsTiCC light curves')
        #             np.concatenate(ax)[b].plot(t_new - new_t_peak,\
        #                 new_y_peak - m_func,\
        #                 '-', linewidth = 0.1, color = 'r', alpha = 0.5, label = 'VL fit')

        else:
            np.concatenate(ax)[b].errorbar(x - t_peak, ymin - m, yerr=merr, fmt='o', \
                                           linewidth=3, color='k', alpha = 0.1
                                           )
    #             np.concatenate(ax)[b].plot(t_new - new_t_peak,\
    #                 new_y_peak - m_func,\
    #                 '-', linewidth = 0.1, color = 'r', alpha = 0.5)

    for tp in band_sntypes:
        if b == 2:
            try:
                if tp == 'Ib':
                    up_lim = 55
                else:
                    up_lim = 55

                np.concatenate(ax)[b].plot(tmpl[bb[b]][tp]['t'][tmpl[bb[b]][tp]['t'] < up_lim], \
                                           tmpl[bb[b]][tp]['rollingMedian'][tmpl[bb[b]][tp]['t'] < up_lim], \
                                           '-', color=colorTypes[tp], linewidth=5, label=tp + ' template')
                np.concatenate(ax)[b].fill_between(tmpl[bb[b]][tp]['t'][tmpl[bb[b]][tp]['t'] < up_lim], \
                                                   tmpl[bb[b]][tp]['rollingPc25'][tmpl[bb[b]][tp]['t'] < up_lim], \
                                                   tmpl[bb[b]][tp]['rollingPc75'][tmpl[bb[b]][tp]['t'] < up_lim], \
                                                   alpha=0.3, color=colorTypes[tp])
            except:
                pass
        else:

            try:
                if b == 1 and tp == 'Ic':

                    np.concatenate(ax)[b].plot(tmpl[bb[b]][tp]['t'][tmpl[bb[b]][tp]['t'] < 55], \
                                               tmpl[bb[b]][tp]['rollingMedian'][tmpl[bb[b]][tp]['t'] < 55], \
                                               '-', color=colorTypes[tp], linewidth=5)
                    np.concatenate(ax)[b].fill_between(tmpl[bb[b]][tp]['t'][tmpl[bb[b]][tp]['t'] < 55], \
                                                       tmpl[bb[b]][tp]['rollingPc25'][tmpl[bb[b]][tp]['t'] < 55], \
                                                       tmpl[bb[b]][tp]['rollingPc75'][tmpl[bb[b]][tp]['t'] < 55], \
                                                       alpha=0.3, color=colorTypes[tp])

                else:
                    np.concatenate(ax)[b].plot(tmpl[bb[b]][tp]['t'], \
                                               tmpl[bb[b]][tp]['rollingMedian'], \
                                               '-', color=colorTypes[tp], linewidth=5)
                    np.concatenate(ax)[b].fill_between(tmpl[bb[b]][tp]['t'], \
                                                       tmpl[bb[b]][tp]['rollingPc25'], \
                                                       tmpl[bb[b]][tp]['rollingPc75'], \
                                                       alpha=0.3, color=colorTypes[tp])
            except:
                pass

                #     plt.title('Plasticc I light curves compared with GP templates in ' + str(bb) + ' band', size=30)

        #         plt.legend(loc = 'lower left', ncol=3, prop={'size':30})
        np.concatenate(ax)[b].tick_params(axis="both", direction="in", which="major", \
                                          right=True, top=True, size=7, labelsize=25, width=2)

    np.concatenate(ax)[b].text(0.93, 0.92, bb[b], transform=np.concatenate(ax)[b].transAxes, \
                               size=50, color='k')

handles, labels = np.concatenate(ax)[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, prop={'size': 35})
np.concatenate(ax)[0].set_yticklabels(['', '-3','-2', '-1', '0'], size=30)
np.concatenate(ax)[2].set_yticklabels(['', '-3','-2', '-1', '0'], size=30)

np.concatenate(ax)[2].set_xticklabels(['', '-20', '-10','0', '10', '20','30', '40', '50', ''], size=30)
np.concatenate(ax)[3].set_xticklabels(['', '-20','-10','0', '10', '20', '30', '40', '50', ''], size=30)


np.concatenate(ax)[0].set_xlim(-25, 55)
np.concatenate(ax)[0].set_ylim(-4, 0.9)
plt.subplots_adjust(hspace=.03, wspace=0.03, top=0.91, left=0.1, bottom=0.1)

fig.text(0.5, 0.04, 'Phase (days)', ha='center', size=40)
fig.text(0.04, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=40)

plt.savefig(os.getenv("SESNPATH") + 'maketemplates/outputs/output_plots/plasticc_gp_tmpl_ugri_high_SN_peak_covered.pdf', bbox_inches='tight')

