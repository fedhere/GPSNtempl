from Functions import *
import numpy as np
import pickle as pkl
from select_lc import *
import sys
from tqdm import tqdm

from GPSNtempl.maketemplates.elastic_data.Functions import lc_fit

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
import templutils as templutils


dirs = ['ELASTICC_TRAIN_SNIb+HostXT_V19/',
        'ELASTICC_TRAIN_SNIb-Templates/',
        'ELASTICC_TRAIN_SNIc+HostXT_V19/',
        'ELASTICC_TRAIN_SNIc-Templates/',
        'ELASTICC_TRAIN_SNIcBL-+HostXT_V19/']
directory = dirs[4]
First = False
justRead = True

if __name__ == '__main__':
     if len(sys.argv) > 1:

         if 'first' in sys.argv:
             First = True
             justRead = False
         elif 'justRead' in sys.argv:
             justRead = True
         if len(sys.argv) > 2:
             try:
                 if sys.argv[2].split('=')[0] == 'title':
                     title = sys.argv.split('=')[1]
             except:
                 print('Input director title as in "title=..."')
                 sys.exit()

         else:
             title = directory.split('_')[2]


su = templutils.setupvars()
coffset = su.coffset
ref = coffset['r']
for b in coffset.keys():
    coffset[b] = coffset[b] - ref


pklf = directory + 'high_' + title + '_peak_covered_low_redshift.pkl'

if First:
    sne = read_lc(directory)
    pkl.dump(sne, open(directory + 'all_SNe_table_'+ title +'.pkl', "wb"))
    selected_lc = select_lc(sne, max_dist= 10)
    pkl.dump(selected_lc, open(pklf, "wb"))
if justRead:
    selected_lc = pkl.load(open(pklf, "rb"))


bb = ['u', 'g', 'r', 'i']


for ID in tqdm(selected_lc.keys()):
    for b in (bb):
        t = selected_lc[ID][b]['t']
        f = selected_lc[ID][b]['f']
        ferr = selected_lc[ID][b]['ferr']
        selected_lc[ID][b]['t_new'] = np.ones(1000) * np.nan
        selected_lc[ID][b]['f_func'] = np.ones(1000) * np.nan

        t = t[f>0]
        ferr = ferr[f>0]
        f = f[f>0]

        if len(f) == 0:
            print('No data points for ' + ID+ ' in band '+ b)
            continue

        x_peak_ref = selected_lc[ID]['r']['t'][np.argmax(selected_lc[ID]['r']['f'])]

        x_peak = x_peak_ref + coffset[b]

        low_lim = -25
        up_lim = 100

        ind = (t < up_lim + x_peak) & (t > low_lim + x_peak)
        x = t[ind]
        xx = x - x_peak_ref
        f = f[ind]
        ferr = ferr[ind]

        if np.sum(np.abs(xx) < 20) >= 3:
            t_new, func, p0 = lc_fit(np.row_stack((t[ind], f, ferr)), x_peak=x_peak)
            # t_new is the new phase and phase zero is x_peak

            if np.sum(np.isinf(func)) > 0:
                print('Inf found in fits for ID=' , ID, ' in band ', b)
                pass
            else:
                selected_lc[ID][b]['t_new'] = t_new
                selected_lc[ID][b]['f_func'] = func

        else:
            pass

pkl.dump(selected_lc, open(pklf, "wb"))