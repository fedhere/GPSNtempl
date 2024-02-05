import glob
import os
import pickle as pkl
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import combinations
from tqdm import tqdm
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

SNTYPES = ['Ib', 'IIb', 'Ic', 'Ic-bl', 'Ibn']
bands = {'u': [0, 1],
         'U': [1, 1],
         'B': [2, 1],
         'g': [3, 1],
         'r': [4, 1],
         'i': [5, 1],
         'V': [6, 1],
         'R': [7, 1],
         'I': [8, 1],
         'J': [9, 1],
         'H': [10, 1],
         'K': [11, 1],
         'w2': [12, 1],
         'm2': [13, 1],
         'w1': [14, 1]}

colorTypes = {'IIb': 'FireBrick',
              'Ib': 'SteelBlue',
              'Ic': 'DarkGreen',
              'Ic-bl': 'DarkOrange',
              'Ibn': 'purple'}

tmpl = {}

directory = glob.glob(os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/*")

if __name__ == '__main__':

    for i, tp in tqdm(enumerate(list(combinations(SNTYPES, 2)))):

        tp1 = tp[0]
        tp2 = tp[1]

        lim00 = [-3.0, 0.5]
        lim01 = [-2.875, -2.525]

        lim10 = [-2.65, -2.65]
        lim11 = [-2.8, -2.8]

        lim10_text = -2.6
        lim11_text = -2.75

        if tp1 == 'Ibn' or tp2 == 'Ibn':
            shift = 1.1
            lim00 = [-3.0 - shift, 0.5]
            lim01 = [-2.875 - shift, -2.525 - shift]

            lim10 = [-2.65 - shift, -2.65 - shift]
            lim11 = [-2.8 - shift, -2.8 - shift]

            lim10_text = -2.6 - shift
            lim11_text = -2.75 - shift

        band_new = []

        for b in bands:

            bb = b
            if b == 'i':
                bb = 'ip'
            if b == 'u':
                bb = 'up'
            if b == 'r':
                bb = 'rp'

            path1 = os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (tp1, bb)
            path2 = os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (tp2, bb)

            if path1 in directory and path2 in directory:
                pass
            else:
                bands[b][1] = 0
                continue

            band_new.append(b)

        n = len(band_new)

        fig = plt.figure(figsize=(44, 54))

        if n % 3 == 0:
            gs0 = gridspec.GridSpec(int(n / 3), 3, wspace=0.01, hspace=0.01, top=0.99, bottom=0.1)
        else:
            gs0 = gridspec.GridSpec(int(n / 3) + 1, 3, wspace=0.01, hspace=0.01, top=0.99, bottom=0.1)

        marker = 0
        for c, b in enumerate(band_new):
            subplot_num = bands[b][0]
            if subplot_num > n - 1:
                subplot_num = c
                continue


            bb = b
            if b == 'i':
                bb = 'ip'
            if b == 'u':
                bb = 'up'
            if b == 'r':
                bb = 'rp'

            if c == 0:
                gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[4, 1], hspace=0,
                                                        subplot_spec=gs0[subplot_num])
                a00 = fig.add_subplot(gs00[0])
                a11 = fig.add_subplot(gs00[1])
                a0 = fig.add_subplot(gs00[0])
                a1 = fig.add_subplot(gs00[1])
            else:
                gs00 = gridspec.GridSpecFromSubplotSpec(2, 1, height_ratios=[4, 1], hspace=0,
                                                        subplot_spec=gs0[subplot_num])
                a0 = fig.add_subplot(gs00[0], sharey=a00, sharex=a00)
                a1 = fig.add_subplot(gs00[1], sharex=a00, sharey=a11)

            path1 = os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (tp1, bb)
            path2 = os.getenv("SESNPATH") + "maketemplates/outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (tp2, bb)

            tmpl_1 = pkl.load(open(path1, "rb"))
            tmpl_2 = pkl.load(open(path2, "rb"))

            a0.plot(tmpl_1['t'], tmpl_1['rollingMedian'], '-',
                    color=colorTypes[tp1], linewidth=5, label=tp1)
            a0.fill_between(tmpl_1['t'],
                            tmpl_1['rollingPc25'],
                            tmpl_1['rollingPc75'],
                            color=colorTypes[tp1], alpha=0.3)

            t_new = [[], [], []]
            t_new[0].append(tmpl_1['t_lc_per_window'][0])
            t_new[1].append(tmpl_1['lc_per_window'][0])
            t_new[2].append(tmpl_1['windows'][0])
            for j, item in enumerate(tmpl_1['lc_per_window'][:-1]):

                if not (tmpl_1['lc_per_window'][j + 1] - tmpl_1['lc_per_window'][j]) == 0:
                    t_new[0].append(tmpl_1['t_lc_per_window'][j])
                    t_new[1].append(tmpl_1['lc_per_window'][j])
                    t_new[2].append(tmpl_1['windows'][j])
            t_new[0].append(tmpl_1['t_lc_per_window'][-1])
            t_new[1].append(tmpl_1['lc_per_window'][-1])
            t_new[2].append(tmpl_1['windows'][-1])

            for j in range(1, len(t_new[0])):
                window = t_new[2][j] / 24.
                a1.plot([t_new[0][j - 1] + window / 2, t_new[0][j] - window / 2], lim10, linewidth=t_new[1][j],
                        color=colorTypes[tp1])

                if t_new[1][j] == min(t_new[1]) or t_new[1][j] == max(t_new[1]):
                    a1.text(t_new[0][j - 1] + ((t_new[0][j] - t_new[0][j - 1]) / 3), lim10_text, str(t_new[1][j]),
                            color=colorTypes[tp1], size=30)

            a0.plot(tmpl_2['t'], tmpl_2['rollingMedian'], '-',
                    color=colorTypes[tp2], linewidth=5, label=tp2)
            a0.fill_between(tmpl_2['t'],
                            tmpl_2['rollingPc25'],
                            tmpl_2['rollingPc75'],
                            color=colorTypes[tp2], alpha=0.3)

            t_new = [[], [], []]
            t_new[0].append(tmpl_2['t_lc_per_window'][0])
            t_new[1].append(tmpl_2['lc_per_window'][0])
            t_new[2].append(tmpl_2['windows'][0])
            for j, item in enumerate(tmpl_2['lc_per_window'][:-1]):

                if not (tmpl_2['lc_per_window'][j + 1] - tmpl_2['lc_per_window'][j]) == 0:
                    t_new[0].append(tmpl_2['t_lc_per_window'][j])
                    t_new[1].append(tmpl_2['lc_per_window'][j])
                    t_new[2].append(tmpl_2['windows'][j])
            t_new[0].append(tmpl_2['t_lc_per_window'][-1])
            t_new[1].append(tmpl_2['lc_per_window'][-1])
            t_new[2].append(tmpl_2['windows'][-1])

            for j in range(1, len(t_new[0])):
                window = t_new[2][j] / 24.
                a1.plot([t_new[0][j - 1] + window / 2, t_new[0][j] - window / 2], lim11, linewidth=t_new[1][j],
                        color=colorTypes[tp2])

                if t_new[1][j] == min(t_new[1]) or t_new[1][j] == max(t_new[1]):
                    a1.text(t_new[0][j - 1] + ((t_new[0][j] - t_new[0][j - 1]) / 2), lim11_text, str(t_new[1][j]),
                            color=colorTypes[tp2], size=30)

            a0.set_ylim(lim00)
            a1.set_ylim(lim01)
            a0.tick_params(axis="both", direction="in", which="major", right=True, top=True, size=7, labelsize=35,
                           width=2)
            a1.tick_params(axis="x", direction="in", which="major", top=True, size=7, labelsize=35, width=2)
            a1.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
            a0.text(0.1, 0.1, "%s" % (b), size=60, transform=a0.transAxes)

            if (subplot_num + 1) % 3 != 1:
                a0.tick_params(axis='y', which='both', right=True, left=True, labelleft=False)

            if subplot_num == len(band_new) - 1 or subplot_num == len(band_new) - 2 or subplot_num == len(band_new) - 3:
                pass
            else:
                a1.tick_params(axis='x', which='both', top=True, bottom=True, labelbottom=False)

            # a0[i].set_title("Comparing GP templates of subtypes %s and %s in band %s"%( tp1, b2, bb), size = 35)
            # a0[i].set_title("%s"%(bb), size = 4tp)

            if c == 1:
                h, l = a0.get_legend_handles_labels()
        fig.legend(h, l, loc='upper center', ncol=2, prop={'size': 60}, bbox_to_anchor=(0.5, 1.03))

        fig.text(0.07, 0.5, 'Relative magnitude', va='center', rotation='vertical', size=60)
        fig.text(0.45, 0.08, 'Phase (days)', va='center', size=60)

        fig.savefig(output_directory + "GPs_2022/compare_gp_tmpls/GPcompare_subtype_pairs_%s_%s.pdf" %
                    (tp1, tp2), bbox_inches='tight')
