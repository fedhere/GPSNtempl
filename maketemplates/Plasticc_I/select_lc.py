import numpy as np
from Functions import read_snana_fits
from tqdm import tqdm


def read_lc(directory, num=40):
    sne = []
    for c in tqdm(range(num)):

        n = c + 1

        if n < 10:
            head = directory + 'ELASTICC_TRAIN_NONIaMODEL0-000' + str(n) + '_HEAD.FITS'
            phot = directory + 'ELASTICC_TRAIN_NONIaMODEL0-000' + str(n) + '_PHOT.FITS'
        else:
            head = directory + 'ELASTICC_TRAIN_NONIaMODEL0-00' + str(n) + '_HEAD.FITS'
            phot = directory + 'ELASTICC_TRAIN_NONIaMODEL0-00' + str(n) + '_PHOT.FITS'

        data = read_snana_fits(head, phot, n=None)
        sne.append(data)

    return sne


def select_lc(sne, max_dist=5, high_SN_ratio_threshold=10, least_num_high_SN=5, redshift_threshold=0.2):
    selected_lc = {}
    bb = ['u ', 'g ', 'r ', 'i ']

    count = {'u': 0,
             'g': 0,
             'r': 0,
             'i': 0}

    c= 0
    for j, data in enumerate(sne):

        for i in range(len(data)):
            selected_lc[c] = {}
            tmp = 4

            if data[i].meta['REDSHIFT_HELIO'] > redshift_threshold:
                continue

            for b in bb:

                x_peak = data[i].meta['PEAKMJD']

                t = np.asarray(data[i]['MJD'][data[i]['BAND'] == b])
                f0 = np.asarray(data[i]['ZEROPT'][data[i]['BAND'] == b])
                f = np.asarray(data[i]['FLUXCAL'][data[i]['BAND'] == b])
                ferr = np.asarray(data[i]['FLUXCALERR'][data[i]['BAND'] == b])
                SN = f / ferr

                t = t[f > 0]
                ferr = ferr[f > 0]
                f0 = f0[f > 0]
                SN = SN[f > 0]
                f = f[f > 0]

                if len(t[((t - x_peak) > -max_dist) & ((t - x_peak) < max_dist)]) < 2:
                    tmp -= 1
                    continue

                if np.sum(SN > high_SN_ratio_threshold) < least_num_high_SN:
                    tmp -= 1
                    continue

                low_lim = -50
                up_lim = 100

                ind = (t < up_lim + x_peak) & (t > low_lim + x_peak)

                y = f[ind]
                yerr = ferr[ind]
                x = t[ind]

                m = 27.5 - 2.5 * np.log10(y)
                merr = 2.5 / np.log(10) * yerr / y

                ind = (x - x_peak < 5) & (x - x_peak > -5)
                if len(m[ind]) == 0:
                    tmp -= 1
                    continue
                mmax = np.min(m)

                selected_lc[c][b.strip()] = {}
                selected_lc[c][b.strip()]['x'] = x
                selected_lc[c][b.strip()]['y'] = y
                selected_lc[c][b.strip()]['yerr'] = yerr
                selected_lc[c][b.strip()]['m'] = m
                selected_lc[c][b.strip()]['merr'] = merr
                selected_lc[c][b.strip()]['x_peak'] = x_peak
                selected_lc[c][b.strip()]['y_peak_m'] = mmax

                count[b.strip()] += 1

            if tmp != 0:
                c += 1

            print('total lc:', c)
            print('number of lc in u, g, r, i:', count['u'], count['g'], count['r'], count['i'])

    return selected_lc
