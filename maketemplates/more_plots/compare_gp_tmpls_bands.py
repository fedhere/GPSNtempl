import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm
import glob
import warnings
warnings.filterwarnings("ignore")


try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)





SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']
bands = ['R','V','r','g','U','u','J','B','H','I','i','K','m2','w1','w2']
colorBands = {'U':'k','up':'k','B':'#0066cc','V':'#47b56c','R':'#b20000','I':'m',
                         'rp':'#b20000','ip':'m', 'g': '#317e4b',
                         'J':'#4F088A','H':'#FFB700','K':'#A4A4A4',
                         'm2':'DarkOrange', 'w1':'FireBrick', 'w2':'purple'}


tmpl = {}

directory = glob.glob(os.getenv("SESNPATH") +"maketemplates/outputs/GPs_2022/*")

for SNTYPE in SNTYPES:

    

    tmpl[SNTYPE] = {}

    for bb in bands:
        # bb = b
        if bb == 'i':
            bb = 'ip'
        if bb == 'u':
            bb = 'up'
        if bb == 'r':
            bb = 'rp'
        
        



        path = os.getenv("SESNPATH") +"maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
        if not path in directory:
            continue
        tmpl_ = pkl.load(open(path, "rb"))
        
#         print(tmpl_['rollingMedian'])

        if not np.nansum(tmpl_['rollingMedian']) == 0:

            tmpl[SNTYPE][bb] = {}
            tmpl[SNTYPE][bb] = tmpl_
            d = {'t': tmpl_['t'][~np.isnan(tmpl_['rollingMedian'])],\
                 'rollMed': tmpl_['rollingMedian'][~np.isnan(tmpl_['rollingMedian'])],\
                 'perc25':tmpl_['rollingPc25'][~np.isnan(tmpl_['rollingMedian'])],\
                 'perc75':tmpl_['rollingPc75'][~np.isnan(tmpl_['rollingMedian'])]}
            df = pd.DataFrame(data=d)
            df.to_csv("outputs/GPs_2022/compare_gp_tmpls/d3_plot/data/tmpl_%s_%s.csv"%(SNTYPE,bb))

for SNTYPE in tqdm(SNTYPES):
    # print (bb)

    bb = SNTYPE

    ax = {}
    a0 = {}
    a1 = {}
    figs = []

    # if bb not in 

    sntypes = [*tmpl[SNTYPE]]
    n = len([*tmpl[SNTYPE]])

    # print (n)


    for i in range(int((n*(n-1))/2)):


        f, (a0[i],a1[i]) = plt.subplots(2,1, gridspec_kw = {'height_ratios': [4, 1]}, sharex = True, figsize = (20,20))
        f.subplots_adjust(hspace=0)
        figs.append(f)


    for i, b in enumerate(list(combinations([*tmpl[bb]], 2))):

        b1 = b[0]
        b2 = b[1]

        if b1 == 'i':
            b1 = 'ip'
        if b1 == 'u':
            b1 = 'up'
        if b1 == 'r':
            b1 = 'rp'

        if b2 == 'i':
            b2 = 'ip'
        if b2 == 'u':
            b2 = 'up'
        if b2 == 'r':
            b2 = 'rp'


        a0[i].plot(tmpl[bb][b1]['t'], tmpl[bb][b1]['rollingMedian'], '-',\
                   color = colorBands[b1], linewidth=4, label = b1)
        a0[i].fill_between(tmpl[bb][b1]['t'],
                           tmpl[bb][b1]['rollingPc25'],
                           tmpl[bb][b1]['rollingPc75'],
                           color= colorBands[b1], alpha=0.3)

        # x1, y1 = [min(tmpl[bb][b1]['t_lc_per_window']), max(tmpl[bb][b1]['t_lc_per_window'])], [-3.25,-3.25]
        # a0[i].plot(tmpl[bb][b1]['t_lc_per_window'], tmpl[bb][b1]['lc_per_window'])
        # a0[i].plot(x1, y1, color=colorBands[b1], marker = '|')
        t_new = [[],[], []]
        t_new[0].append(tmpl[bb][b1]['t_lc_per_window'][0])
        t_new[1].append(tmpl[bb][b1]['lc_per_window'][0])
        t_new[2].append(tmpl[bb][b1]['windows'][0])
        for j,item in enumerate(tmpl[bb][b1]['lc_per_window'][:-1]):
            # x1 = [tmpl[bb][b1]['t_lc_per_window'][j]-(tmpl[bb][b1]['windows'][j]/2.), tmpl[bb][b1]['t_lc_per_window'][j]+(tmpl[bb][b1]['windows'][j]/2.)]
            # y1 = [-3.25,-3.25]
            # a0[i].plot(x1, y1, color=colorBands[b1],linewidth = tmpl[bb][b1]['lc_per_window'][j])

            if not (tmpl[bb][b1]['lc_per_window'][j+1]-tmpl[bb][b1]['lc_per_window'][j])==0:
                # a0[i].plot(tmpl[bb][b1]['t_lc_per_window'][j],-3.25, color=colorBands[b1],marker='|', markersize=15)
                t_new[0].append(tmpl[bb][b1]['t_lc_per_window'][j])
                t_new[1].append(tmpl[bb][b1]['lc_per_window'][j])
                t_new[2].append(tmpl[bb][b1]['windows'][j])
        t_new[0].append(tmpl[bb][b1]['t_lc_per_window'][-1])
        t_new[1].append(tmpl[bb][b1]['lc_per_window'][-1])
        t_new[2].append(tmpl[bb][b1]['windows'][-1])
        # a0[i].plot(tmpl[bb][b1]['t_lc_per_window'][0],-3.25, color=colorBands[b1],marker='|', markersize=15)
        # a0[i].plot(tmpl[bb][b1]['t_lc_per_window'][-1],-3.25, color=colorBands[b1],marker='|', markersize=15)

        for j in range(1, len(t_new[0])): 
            window = t_new[2][j]/24.
            a1[i].plot([t_new[0][j-1]+window/2,t_new[0][j]-window/2],[-3.5,-3.5], linewidth = t_new[1][j], color=colorBands[b1])
            # if t_new[1][j]== min(t_new[1]):
            #     a1[i].axvline(t_new[0][j], color=colorBands[b1])
            #     a1[i].axvline(t_new[0][j-1], color=colorBands[b1])
            if t_new[1][j]== min(t_new[1]) or t_new[1][j]== max(t_new[1]):
                a1[i].text(t_new[0][j-1]+((t_new[0][j]-t_new[0][j-1])/3), -3.45,str(t_new[1][j]), color=colorBands[b1], size = 20)


        a0[i].plot(tmpl[bb][b2]['t'], tmpl[bb][b2]['rollingMedian'], '-',\
                   color = colorBands[b2], linewidth=4,label = b2)
        a0[i].fill_between(tmpl[bb][b2]['t'],
                           tmpl[bb][b2]['rollingPc25'],
                           tmpl[bb][b2]['rollingPc75'],
                           color= colorBands[b2], alpha=0.3)

        # x1, y1 = [min(tmpl[bb][b2]['t_lc_per_window']), max(tmpl[bb][b2]['t_lc_per_window'])], [-3.75,-3.75]
        # a0[i].plot(tmpl[bb][b2]['t_lc_per_window'], tmpl[bb][b2]['lc_per_window'])
        # a0[i].plot(x1, y1, color=colorBands[b2], marker = '|')
        t_new = [[],[], []]
        t_new[0].append(tmpl[bb][b2]['t_lc_per_window'][0])
        t_new[1].append(tmpl[bb][b2]['lc_per_window'][0])
        t_new[2].append(tmpl[bb][b2]['windows'][0])
        for j,item in enumerate(tmpl[bb][b2]['lc_per_window'][:-1]):
            # if j == 0:
            #     x1 = [tmpl[bb][b2]['t_lc_per_window'][j], tmpl[bb][b2]['t_lc_per_window'][j]+(tmpl[bb][b2]['windows'][j]/2.)]
            # elif j == len(tmpl[bb][b2]['lc_per_window'])-1:
            #     x1 = [tmpl[bb][b2]['t_lc_per_window'][j]-(tmpl[bb][b2]['windows'][j]/2.), tmpl[bb][b2]['t_lc_per_window'][j]]
            # else:
            #     x1 = [tmpl[bb][b2]['t_lc_per_window'][j]-(tmpl[bb][b2]['windows'][j]/2.), tmpl[bb][b2]['t_lc_per_window'][j]+(tmpl[bb][b2]['windows'][j]/2.)]
            # y1 = [-3.75,-3.75]
            # a0[i].plot(x1, y1, color=colorBands[b2],linewidth = tmpl[bb][b2]['lc_per_window'][j])

            if not (tmpl[bb][b2]['lc_per_window'][j+1]-tmpl[bb][b2]['lc_per_window'][j])==0:
                # a0[i].plot(tmpl[bb][b2]['t_lc_per_window'][j],-3.75, color=colorBands[b2],marker='|', markersize=15)
                t_new[0].append(tmpl[bb][b2]['t_lc_per_window'][j])
                t_new[1].append(tmpl[bb][b2]['lc_per_window'][j])
                t_new[2].append(tmpl[bb][b2]['windows'][j])
        t_new[0].append(tmpl[bb][b2]['t_lc_per_window'][-1])
        t_new[1].append(tmpl[bb][b2]['lc_per_window'][-1])
        t_new[2].append(tmpl[bb][b2]['windows'][-1])
        # a0[i].plot(tmpl[bb][b2]['t_lc_per_window'][0],-3.75, color=colorBands[b2],marker='|', markersize=15)
        # a0[i].plot(tmpl[bb][b2]['t_lc_per_window'][-1],-3.75, color=colorBands[b2],marker='|', markersize=15)

        for j in range(1, len(t_new[0])): 
            window = t_new[2][j]/24.
            a1[i].plot([t_new[0][j-1]+window/2,t_new[0][j]-window/2],[-3.65, -3.65], linewidth = t_new[1][j], color=colorBands[b2])
            # if t_new[1][j]== min(t_new[1]):
            #     a1[i].axvline(t_new[0][j], color=colorBands[b2])
            #     a1[i].axvline(t_new[0][j-1], color=colorBands[b2])

            if t_new[1][j]== min(t_new[1]) or t_new[1][j]== max(t_new[1]):
                a1[i].text(t_new[0][j-1]+((t_new[0][j]-t_new[0][j-1])/2), -3.6,str(t_new[1][j]), color=colorBands[b2], size = 20)

        a0[i].legend(prop={'size': 45})

        # a1[i].set_xlabel('Phase(days)', size=40)
        # a0[i].set_ylabel('Relative Magnitude', size=40)
        a0[i].set_ylim([-3.85, 0.5])
        a1[i].set_ylim([-3.725, -3.375])
        a0[i].tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=35,
                          width=2)
        a1[i].tick_params(axis="x", direction="in", which="major", top=True, size=7, labelsize=35, width=2)
        a1[i].tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
        a0[i].text(0.1, 0.1, "%s" % (bb), size=60, transform=a0[i].transAxes)


        # ax[i].set_title("Comparing GP templates in bands %s and %s for subtype %s"%( b1, b2, bb), size = 25)
        figs[i].savefig("outputs/GPs_2022/compare_gp_tmpls/GPcompare_%s_%s_%s.pdf"%(bb, b1, b2))

        plt.close(figs[i])
