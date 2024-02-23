from Functions import *
# from savgol import savitzky_golay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from glob import glob
# import gzip
# from astropy.io import fits
# from scipy.interpolate import interp1d
# from scipy.optimize import minimize
import pickle as pkl
# import george
# from george import kernels
# from select_lc import select_lc
# import traceback


direcs = ['ELASTICC_TRAIN_SNIb+HostXT_V19/',
          'ELASTICC_TRAIN_SNIb-Templates/',
          'ELASTICC_TRAIN_SNIc+HostXT_V19/',
          'ELASTICC_TRAIN_SNIc-Templates/',
          'ELASTICC_TRAIN_SNIcBL+HostXT_V19/']

filenames = ['Ib_HostXT',
             'Ib_Templates',
             'Ic_HostXT',
             'Ic_Templates',
             'IcBL_HostXT']

SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']

bands_all = ['R','V','r','g','U','u','J','B','H','I','i','K','m2','w1','w2']

colorTypes = {'IIb':'FireBrick',
             'Ib':'SteelBlue',
             'Ic':'DarkGreen',
             'Ic-bl':'DarkOrange',
             'Ibn':'purple'}

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

tmpl = {}

bands = ['up', 'g', 'rp', 'ip']

for bb in bands:

    tmpl[bb[0]] = {}

    for SNTYPE in SNTYPES:

        tmpl[bb[0]][SNTYPE] = {}

        try:
            path = "./../outputs/GPs_2022/GPalltemplfit_%s_%s_V0.pkl" % (SNTYPE, bb)
            tmpl_ = pkl.load(open(path, "rb"))
        except:
            continue

        #         print(tmpl_['rollingMedian'])

        if np.nansum(tmpl_['rollingMedian']) == 0:
            print(bb, SNTYPE)
            continue

        tmpl[bb[0]][SNTYPE] = tmpl_

for j, d in enumerate(direcs):
    pklf = 'high_SN_'+ filenames[j] +'_peak_covered_low_redshift.pkl'
    selected_lc = pkl.load(open(pklf, "rb"))

    globals()["fits_"+ filenames[j]] = {}

    for b in ['u', 'g', 'r', 'i']:
        globals()["fits_"+ filenames[j]][b] = {}

        for i, ID in enumerate(list(selected_lc.keys())):
            print(ID)

            plt.figure()

            t = selected_lc[ID][b]['t']
            f = selected_lc[ID][b]['f']
            ferr = selected_lc[ID][b]['ferr']

            t = t[f > 0]
            ferr = ferr[f > 0]
            f = f[f > 0]

            if len(f) == 0:
                print(ID)
                continue

            m = 27.5 - 2.5 * np.log10(f)  # -2.5*np.log10(f/(10**(-0.4*27.5)))
            #     median = np.nanmedian(m)
            #     m = m- median
            merr = 2.5 / np.log(10) * ferr / f

            #     x = x - x[np.argmin(y)]
            x_peak = t[np.argmin(m)]
            y_peak = m[np.argmin(m)]

            low_lim = -25
            up_lim = 100

            ind = (t < up_lim + x_peak) & (t > low_lim + x_peak)
            y = m[ind]
            yerr = merr[ind]
            f = f[ind]
            ferr = ferr[ind]
            x = t
            x = x[ind]
            y = y - y[np.argmin(y)]
            # t = np.linspace(x.min(), x.max(), 1000)

            #         chi2  = []
            #         temp_res = [[],[]]

            #         if x[-1]>20:
            #             cutoff = 20
            #         else:
            #             continue

            if len(x[x < 0]) < 2:
                continue

            globals()["fits_"+ filenames[j]][b][ID] = {}

            df = pd.DataFrame({'t': x, 'A': y, 'A_err': yerr})

            #         for w in [1, 0.1, 0.01, 0.001, 0.0001]:
            #     #         p0 = [.05, max(f), w, 20, -0.5]
            #     #         res = minimize(nll_VC, p0,
            #     #                    args=(np.asarray(x), np.asarray(f), np.asarray(ferr)),method = 'Powell')

            #             p1 = [.03, max(f), w]
            #             l = len(x[x<cutoff])
            #             res1 = minimize(nll_VC, p1,
            #                            args=(np.asarray(x[:l+1]), np.asarray(f[:l+1]), np.asarray(ferr[:l+1])),method = 'Powell')

            #             chi2.append(res1['fun'])
            #             temp_res[0].append(res1['x'])

            #         final_res1 = temp_res[0][np.argmin(chi2)]
            #         ll = len(t[t<cutoff])
            #         slope = (f[x>=cutoff][0] - sn_fit(t[:l+1], *final_res1)[-1]) / (x[x>=cutoff][0] - t[l])
            #         p2 = [-0.5, sn_fit(t[:l+1], *final_res1)[-1] - (slope) * cutoff]
            #         res2 = minimize(nll_lin, p2,
            #                            args=(np.asarray(x[x>=cutoff]),
            #                                  np.asarray(f[x>=cutoff]),
            #                                  np.asarray(ferr[x>=cutoff])),
            #                         method = 'Powell')

            #         globals()["fits_"+ filenames[j]][ID]['res1'] = final_res1
            #         globals()["fits_"+ filenames[j]][ID]['res2'] = res2['x']
            #         globals()["fits_"+ filenames[j]][ID]['t'] = t
            #         globals()["fits_"+ filenames[j]][ID]['ypred'] = sn_fit(t, *final_res1)
            #     fits[ID]['Ibc_tmpl_t'] = tmpl_t

            plt.errorbar(x - x_peak, -y, yerr=yerr, fmt='.')
            #     plt.plot(t, sn_fit(t, *final_res))

            #         plt.plot(t[t>=cutoff], linear(t[t>=cutoff], *res2['x']))
            #     ll = len(t[t<cutoff])
            #         y_pred = sn_fit(t[t<x[l+1]], *final_res1)
            #     y_pred[-1] = linear(t[t>=cutoff], *res2['x'])[0]
            #         plt.plot(t[t<x[l+1]], y_pred)

            #         y1 = np.zeros(len(t))
            #         y1[t<x[l]] = sn_fit(t[t<x[l]], *final_res1)
            #         y1[t>=x[l]] = linear(t[[t>=x[l]]], *res2['x'])
            #         y1[(t<x[l]+10) & (t>x[l]-10)] = savitzky_golay(y1[(t<x[l]+10) & (t>x[l]-10)], 51, 2)

            #         plt.plot(t, y1)

            #         globals()["fits_"+ filenames[j]][b][ID]['t'] = t
            #         globals()["fits_"+ filenames[j]][b][ID]['y'] = y1

            plt.plot(x - x_peak, -Chebyhev_fitter(df, 50))

            plt.title(ID)

            plt.savefig(d + '/plots/' + ID + '_' + b + '.png')

    pkl.dump(globals()["fits_"+ filenames[j]], open(d + 'plots/high_SN_'+ filenames[j] +'_peak_covered_low_redshift.pkl', "wb"))


