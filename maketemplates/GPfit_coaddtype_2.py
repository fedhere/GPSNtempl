import glob
import os
import pylab as pl
import sys
import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns

import utils.snclasses as snstuff
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")

mpl.use('agg')

avoid = ["03dh"]

pl.rcParams['figure.figsize'] = (10, 10)

# # Loading CfA SN lightcurves

# setting parameters for lcv reader
# use literature data (if False only CfA data)
LIT = True
# use NIR data too
FNIR = True

SNTYPE = 'Ic-bl'

# pl.ion()
readgood = pd.read_csv("goodGPs2_fit_selection3.csv", header=None, usecols=[0, 1, 2, 3])
# print('readgood is ' ,readgood)
# print readgood
# sys.exit()


DEBUG = False

tcorlims = {
    'R': {'tmin': 10, 'tmax': 20},
    'V': {'tmin': 10, 'tmax': 20},
    'r': {'tmin': 10, 'tmax': 20},
    'g': {'tmin': 10, 'tmax': 20},
    'U': {'tmin': 10, 'tmax': 10},
    'u': {'tmin': 10, 'tmax': 10},
    'J': {'tmin': 20, 'tmax': 10},
    'B': {'tmin': 10, 'tmax': 15},
    'H': {'tmin': 15, 'tmax': 20},
    'I': {'tmin': 10, 'tmax': 20},
    'i': {'tmin': 10, 'tmax': 20},
    'K': {'tmin': 20, 'tmax': 20},
    'm2': {'tmin': 20, 'tmax': 20},
    'w1': {'tmin': 20, 'tmax': 20},
    'w2': {'tmin': 20, 'tmax': 20}}

if __name__ == '__main__':

    # uncomment for all lcvs to be read in
    allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                         "/SESNessentials.csv", encoding="ISO-8859-1").reset_index(drop=True)['SNname'].values
    tps = pd.read_csv(os.getenv("SESNCFAlib") +
                      "/SESNessentials.csv", encoding="ISO-8859-1").reset_index(drop=True)['Type'].values

    # NUM_COLORS = len(allsne)
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)

    sns.reset_orig()  # get default matplotlib styles back

    if len(sys.argv) > 1:
        if sys.argv[1] in ['Ib', 'IIb', 'Ic', 'Ic-bl', 'Ib-c', 'Ibn']:
            SNTYPE = sys.argv[1]
        else:
            allsne = [sys.argv[1]]

    # set up SESNCfalib stuff
    su = templutils.setupvars()
    if len(sys.argv) > 2:
        bands = [sys.argv[2]]
    else:
        bands = su.bands
    nbands = len(bands)
    coffset = su.coffset

    # errorbarInflate = {"93J":30,
    #                   "05mf":1}

    ax = {}
    axv1 = {}
    axv2 = {}
    axs_com = {}
    axs_com2 = {}
    figs = []
    figs_com = []
    figs_com2 = []
    figsv1 = []
    figsv2 = []

    for j, b in enumerate(bands):
        f, ax[b] = pl.subplots(2, sharex=True, sharey=True)  # figsize=(20,20))
        f.subplots_adjust(hspace=0)
        figs.append(f)

        f1, axv1[b] = pl.subplots(3, sharex=True, sharey=True)  # figsize=(20,20))
        f1.subplots_adjust(hspace=0)
        figsv1.append(f1)

        fv2, axv2[b] = pl.subplots(2, sharex=True, sharey=True)  # figsize=(20,20))
        fv2.subplots_adjust(hspace=0)
        figsv2.append(fv2)

        fv3, axs_com[b] = pl.subplots(2, sharex=True, sharey=True)  # figsize=(20,20))
        fv3.subplots_adjust(hspace=0)
        figs_com.append(fv3)

        fv4, axs_com2[b] = pl.subplots(1, figsize=(22, 15))
        fv4.subplots_adjust(hspace=0)
        figs_com2.append(fv4)

    # pl.ion()
    dt = 0.5
    t = np.arange(-15, 50, dt)

    # set up arrays to host mean, mean shifted by peak, standard dev, and dtandard dev shifted
    mus = np.zeros((len(allsne), len(bands), len(t))) * np.nan
    musShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan
    stds = np.zeros((len(allsne), len(bands), len(t))) * np.nan
    stdsShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan

    # set a plot color for each SN by creating a colorblind set of colors
    # allsne_colormap aruments: n = number of SNe, colormap = 'colorblind'
    NUM_COLORS = len(allsne)
    su.allsne_colormaps(NUM_COLORS)
    clrs = su.allsne_colors

    for i, sn in enumerate(allsne):

        # see if SN is in the list of good SNe with ok GP fits
        goodbad = readgood[readgood[0] == sn]

        if len(goodbad) == 0:
            if DEBUG:
                # raw_input()
                print('Removed because gp fit is not ok:', sn)
            continue

        # read and set up SN object and look for photometry files
        try:
            thissn = snstuff.mysn(sn, addlit=True)
        except AttributeError:
            print('1 removed sn: ', sn)
            continue
        if len(thissn.optfiles) + len(thissn.fnir) == 0:
            print("bad sn")

        # read metadata for SN
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)

        if not thissn.type == SNTYPE:
            continue

        if thissn.snnameshort in avoid:
            continue

        # check SN is ok and load data
        if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
            print("bad sn")
            continue

        print(" starting loading ")
        print(os.environ['SESNPATH'] + "/finalphot/*" + \
              thissn.snnameshort.upper() + ".*[cf]")
        print(os.environ['SESNPATH'] + "/finalphot/*" + \
              thissn.snnameshort.lower() + ".*[cf]")

        print(glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                        thissn.snnameshort.upper() + ".*[cf]") + \
              glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                        thissn.snnameshort.lower() + ".*[cf]"))

        lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
        thissn.setphot()
        thissn.getphot()
        thissn.setphase()
        thissn.sortlc()
        # thissn.printsn()

        # check that the SN has ANY data points in all bands
        if np.array([n for n in thissn.filters.values()]).sum() == 0:
            print("bad sn")
            continue

        for j, b in enumerate(bands):

            # thissn.getmagmax(band=b, forceredo=True)

            if b == 'i':
                bb = 'ip'
            elif b == 'u':
                bb = 'up'
            elif b == 'r':
                bb = 'rp'
            else:
                bb = b

            tcore = (t > -tcorlims[b]['tmin']) * (t < tcorlims[b]['tmax'])

            if DEBUG:
                print(goodbad)

            if len(goodbad[goodbad[1] == bb]) == 0:
                if DEBUG:
                    print('2 Removed sn:', sn)
                continue

            if goodbad[goodbad[1] == bb][3].values[0] == 'n':
                if DEBUG:
                    print('3 Removed sn', sn)
                continue

            if thissn.filters[b] == 0:
                if DEBUG:
                    print('4 Removed sn', sn)
                continue

            xmin = thissn.photometry[b]['mjd'].min()
            # x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5

            #if np.isnan(thissn.maxmags[b]['epoch']):
            #     t_max = thissn.Vmax + coffset[b]
            # else:
            #     t_max = thissn.maxmags[b]['epoch']

            t_max = thissn.Vmax

            if xmin - t_max < -1000:
                x = thissn.photometry[b]['mjd'] - t_max + 2400000.5

            elif xmin - t_max > 1000:
                x = thissn.photometry[b]['mjd'] - t_max - 2400000.5
            else:
                x = thissn.photometry[b]['mjd'] - t_max

            y = thissn.photometry[b]['mag']

            if np.isnan(thissn.maxmags[b]['mag']):
                y = - y
            else:
                y = thissn.maxmags[b]['mag'] - y

            yerr = thissn.photometry[b]['dmag']

            # print(x)

            if b == 'g':
                low_lim = -20
                up_lim = 50
            elif b == 'I':
                low_lim = -20
                up_lim = 100
            elif b == 'i':
                low_lim = -20
                up_lim = 100

            elif b == 'U':
                low_lim = -15
                up_lim = 40

            elif b == 'u':
                low_lim = -25
                up_lim = 30
            else:
                low_lim = -20
                up_lim = 100

            y = y[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]
            yerr = yerr[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]
            x = x[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]

            t_sn = t[(t <= x.max()) * (t >= x.min())]

            if len(x) == 0:
                print(b, ': No data points within the limits.')
                continue

            ####################################

            # if (b == 'i') and (sn == 'sn2019myn'):
            #     low_lim = x.min()
            #     up_lim = 10
            #     t_sn = t[(t >= low_lim) * (t <= up_lim)]
            # if (b == 'g') and (sn == 'sn2019php'):
            #     low_lim = x.min()
            #     up_lim = 20
            #     t_sn = t[(t >= low_lim) * (t <= up_lim)]
            # elif (b == 'i') and (sn == 'sn2018bcc'):
            #     low_lim = x.min()
            #     up_lim = 15
            #     t_sn = t[(t >= low_lim) * (t <= up_lim)]
            # elif (b == 'i') and (sn == 'sn2019aajs'):
            #     low_lim = x.min()
            #     up_lim = 15
            #     t_sn = t[(t >= low_lim) * (t <= up_lim)]
            # elif (b == 'r') and (sn == 'sn2019myn'):
            #     low_lim = x.min()
            #     up_lim = 20
            #     t_sn = t[(t >= low_lim) * (t <= up_lim)]

            pklf = "./../../GPSNtempl_output/old_outputs/GPfit%s_%s.pkl" % (sn, b + 'p' if b in ['u', 'r', 'i']
            else b)
            if not os.path.isfile(pklf):
                print("missing file ", pklf)
                # raw_input()
                print('5 Removed sn', sn)
                continue

            ygp, gp, tmplm = pkl.load(open(pklf, "rb"))
            print(y.shape)
            print(x.shape)
            print(tmplm(x))
            # try:

            if x[0] > 0:
                delta_y = y[0] + tmplm(x)[0]
                y = y - delta_y
            else:
                # Fixing the vertical alignment issue for those SNe with a pre-shock
                if min(np.abs(x)) < 2:
                    y_min = y[np.argmin(np.abs(x))]
                    y = y - y_min
                else:
                    y_min = y[np.argmin(np.abs(x))] + tmplm(x)[np.argmin(np.abs(x))]
                    y = y - y_min

            mu, cov = gp.predict(y + tmplm(x), np.log(t_sn + 30).reshape(-1, 1))
            # except ValueError:
            #     print(b, sn)
            #
            #     if DEBUG:
            #         traceback.print_exc()
            #         # print ("error")
            #         # raw_input()
            #         print('6 Removed sn', sn)
            # continue

            std = np.sqrt(np.diag(cov))
            if (np.abs(mu) < 0.1).all():
                continue
            mu = mu - tmplm(t_sn)

            ind_start_mu = np.where(t == t_sn.min())[0][0]
            ind_end_mu = np.where(t == t_sn.max())[0][0]

            # print(ind_start_mu, ind_end_mu)

            # print(len(mus[i][j][ind_start_mu:ind_end_mu+1]), len(mu - mu[np.abs(t_sn)==np.abs(t_sn).min()]))

            mus[i][j][ind_start_mu:ind_end_mu + 1] = mu  # - mu[np.abs(t_sn) == np.abs(t_sn).min()]

            stds[i][j][ind_start_mu:ind_end_mu + 1] = std

            # color = np.array(pl.cm.nipy_spectral(snecolors[sn]))[:3]

            ax[b][0].plot(t, mus[i][j], lw=2,
                          label=thissn.snnameshort, alpha=0.5,
                          color=clrs[i])

            ax[b][0].fill_between(t,
                                  mus[i][j] - stds[i][j],
                                  mus[i][j] + stds[i][j],
                                  color='grey', alpha=0.1)

            axv1[b][0].plot(t, mus[i][j], lw=2,
                            label=thissn.snnameshort, alpha=0.5,
                            color=clrs[i])

            axv1[b][0].fill_between(t,
                                    mus[i][j] - stds[i][j],
                                    mus[i][j] + stds[i][j],
                                    color='grey', alpha=0.1)

            # if (sum(~np.isnan(np.array(mus[i][j])[tcore]))) < 4:
            #     continue
            # print((sum(~np.isnan(np.array(mus[i][j])))))
            # print((np.array(mus[i][j])[tcore]))

            try:

                if t_sn[0] > 0 or t_sn[-1] < 0:
                    truemax = np.where(t == 0)[0][0]

                else:
                    truemax = np.where(np.array(mus[i][j]) ==
                                       np.nanmax(np.array(mus[i][j])[tcore]))[0][0]

                # if truemax < 5:
                #     minloc = \
                #     np.where(np.array(mus[i][j]) == np.nanmin(np.array(mus[i][j])[tcore]))[
                #         0][0]
                #     if minloc > 0 and minloc < len(tcore):
                #         tcore = (t > -10) * (t < tcorlims[b]['tmax'])
                #         if not (sum(~np.isnan(np.array(mus[i][j])[tcore]))) == 0:
                #             truemax = np.where(np.array(mus[i][j]) ==
                #                                np.nanmax(np.array(mus[i][j])[tcore]))[0][0]
                #
                #     if np.abs(truemax - np.where(t == t[tcore][0])[0][0]) < 2:
                #         truemax = np.where(t == 0)[0][0]

                t_max_epoch = t[truemax]

            except:

                truemax = np.where(t == 0)[0][0]
                t_max_epoch = t[truemax]

            t2 = t - t_max_epoch
            t20 = np.where(t2 == 0)[0][0]

            yoffset = (mus[i][j])[t20]
            ymax = mus[i][j][truemax]
            if np.isnan(yoffset):
                yoffset = 0
                ymax = 0

            ax[b][0].scatter(t_max_epoch,
                             ymax,
                             lw=2, alpha=0.5,
                             c=clrs[i])
            axv1[b][0].scatter(t_max_epoch,
                               ymax,
                               lw=2, alpha=0.5,
                               c=clrs[i])

            if ((SNTYPE == 'Ic' and thissn.snnameshort == '94I')
                    or (SNTYPE == 'IIb' and (thissn.snnameshort == '93J' or
                                             thissn.snnameshort == '11dh'))
                    or (SNTYPE == 'Ic-bl' and (thissn.snnameshort == '98bw' or
                                               thissn.snnameshort == '06aj'))
                    or (SNTYPE == 'Ib' and (thissn.snnameshort == '08D'))):
                axv1[b][0].scatter(t_max_epoch,
                                   ymax,
                                   lw=2, alpha=1,
                                   c=clrs[i])
                axv1[b][0].errorbar(x,
                                    y, yerr,
                                    lw=1, alpha=1, fmt='.',
                                    label=thissn.snnameshort,
                                    c=clrs[i])
                axv1[b][1].errorbar(x,
                                    y, yerr,
                                    lw=1, alpha=1, fmt='.',
                                    label=thissn.snnameshort,
                                    c=clrs[i])
                axv1[b][2].errorbar(x - t_max_epoch,
                                    y, yerr,
                                    lw=1, alpha=1, fmt='.',
                                    label=thissn.snnameshort,
                                    c=clrs[i])

            tmin, tmax = t2.min(), t2.max()

            # After shifiting the mus, we remove the parts of the light curve out of the original interval
            # And we put the empty part of the light curve within that interval equal to nan.

            if (thissn.snnameshort == '16hgs' and b == 'g') or \
                    (thissn.snnameshort == '13ge' and b == 'U') or \
                    (thissn.snnameshort == '13cq' and b == 'I') or \
                    (thissn.snnameshort == '06aj' and b == 'U') or \
                    (thissn.snnameshort == '06aj' and b == 'B') or \
                    (thissn.snnameshort == '13cq' and b == 'V') or \
                    (thissn.snnameshort == '13df' and b == 'U') or \
                    (thissn.snnameshort == '13df' and b == 'B') or \
                    (thissn.snnameshort == '10jr' and b == 'B') or \
                    (thissn.snnameshort == '06bf' and b == 'r') or \
                    (thissn.snnameshort == '06bf' and b == 'i') or \
                    (thissn.snnameshort == '13cq' and b == 'B'):
                print('exceptions are: ', thissn.snnameshort)
                yoffset = mus[i][j][t == 0]
                musShifted[i][j] = (mus[i][j]) - yoffset
                stdsShifted[i][j] = stds[i][j]


            else:

                if t.min() > tmin:
                    # print(t_sn.max(), t_max_epoch)
                    ind_max = np.where(t == t.max() - t_max_epoch)[0][0]
                    musShifted[i][j][:ind_max] = (mus[i][j])[-ind_max:] - yoffset
                    stdsShifted[i][j][:ind_max] = stds[i][j][-ind_max:]

                elif t.min() < tmin:
                    ind_min = np.where(t == t.min() - t_max_epoch)[0][0]
                    musShifted[i][j][ind_min:] = (mus[i][j])[:-ind_min] - yoffset
                    stdsShifted[i][j][ind_min:] = stds[i][j][:-ind_min]

                else:

                    musShifted[i][j] = (mus[i][j]) - yoffset
                    stdsShifted[i][j] = stds[i][j]

            # print(musShifted)
            axv2[b][0].plot(t, musShifted[i][j], lw=2,
                            label=thissn.snnameshort, alpha=0.5,
                            color=clrs[i])
            axv2[b][0].fill_between(t,
                                    musShifted[i][j] - stdsShifted[i][j],
                                    musShifted[i][j] + stdsShifted[i][j],
                                    color='grey', alpha=0.1)

            axs_com[b][0].plot(t, musShifted[i][j], lw=2,
                               label=thissn.snnameshort, alpha=0.5,
                               color=clrs[i])
            axs_com[b][0].fill_between(t,
                                       musShifted[i][j] - stdsShifted[i][j],
                                       musShifted[i][j] + stdsShifted[i][j],
                                       color='grey', alpha=0.1)

            axs_com2[b].plot(t, musShifted[i][j], lw=4,
                             label=thissn.snnameshort, color=clrs[i], linestyle=LINE_STYLES[i % NUM_STYLES])

            if DEBUG:
                print("all the way down")
                # raw_input()
                print('7 Removed sn', sn)
                # # Subracting the mean and fitting GP to residuals only

                # spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)
            # pl.draw()

            xl = pl.xlabel("log time (starting 30 days before peak)")
        # raw_input()
        # print('Removed sn', sn)

    # for k,m in enumerate(mus):
    # np.save('mus.npy', mus)
    # np.save('stds.npy', stds)
    # np.save('musShifted.npy', musShifted)
    # np.save('stdsShifted.npy', stdsShifted)

    for j, b in enumerate(bands):

        mask = np.isnan(mus) + ~np.isfinite(mus) + \
               np.isnan(stds) + ~np.isfinite(stds) + \
               ~np.isfinite(1.0 / stds)
        maskShifted = np.isnan(musShifted) + ~np.isfinite(musShifted) + \
                      np.isnan(stdsShifted) + ~np.isfinite(stdsShifted) + \
                      ~np.isfinite(1.0 / stdsShifted)  # mask[50:] = True

        mus = np.ma.masked_array(mus, mask)
        stds = np.ma.masked_array(stds, mask)

        musShifted = np.ma.masked_array(musShifted, maskShifted)
        stdsShifted = np.ma.masked_array(stdsShifted, maskShifted)

        average = np.zeros(len(t)) * np.nan
        std = np.zeros(len(t)) * np.nan
        variance = np.zeros(len(t)) * np.nan
        averageShifted = np.zeros(len(t)) * np.nan
        stdShifted = np.zeros(len(t)) * np.nan
        varianceShifted = np.zeros(len(t)) * np.nan

        for l in range(len(t)):

            if (len(musShifted[:, j, :][:, l][~musShifted[:, j, :][:, l].mask]) < 2) or (
                    len(mus[:, j, :][:, l][~mus[:, j, :][:, l].mask]) < 2):
                continue

            average[l] = (np.ma.average(mus[:, j, :][:, l], weights=1.0 / stds[:, j, :][:, l] ** 2))
            std[l] = np.ma.std(stdsShifted[:, j, :][:, l], axis=0)
            averageShifted[l] = (
                np.ma.average(musShifted[:, j, :][:, l], weights=1.0 / stdsShifted[:, j, :][:, l] ** 2))
            stdShifted[l] = np.ma.std(stdsShifted[:, j, :][:, l], axis=0)

            variance[l] = np.ma.average((mus[:, j, :][:, l] - average[l]) ** 2, axis=0,
                                        weights=1.0 / stds[:, j, :][:, l] ** 2) \
                          * np.nansum(1.0 / stds[:, j, :][:, l] ** 2, axis=0) \
                          / (np.nansum(1.0 / stds[:, j, :][:, l] ** 2, axis=0) - 1)

            varianceShifted[l] = np.ma.average((musShifted[:, j, :][:, l] - averageShifted[l]) ** 2, axis=0,
                                               weights=1.0 / stdsShifted[:, j, :][:, l] ** 2) \
                                 * np.nansum(1.0 / stdsShifted[:, j, :][:, l] ** 2, axis=0) \
                                 / (np.nansum(1.0 / stdsShifted[:, j, :][:, l] ** 2, axis=0) - 1)

        window = 3
        phs = t  # np.arange(min(t) - 4, max(t) + 8, 1.0/24)
        N = len(phs)
        rollWmu = np.zeros(N) * np.nan
        rollStd = np.zeros(N) * np.nan
        rollWstd = np.zeros(N) * np.nan
        rollPc25 = np.zeros(N) * np.nan
        rollMed = np.zeros(N) * np.nan
        rollPc75 = np.zeros(N) * np.nan

        # In this list, we save the number of lightcurves within each time window along with the length of the window
        lc_num = [[], [], []]
        lc_num_total = []

        for i, hour in enumerate(phs):
            # if hour < -10:
            #     window = 5
            if hour > 15 and hour < 20:
                window = 4
            elif hour > 20 and hour < 27:
                window = 5
            elif hour > 27 and hour < 35:
                window = 6
            elif hour > 35 and hour < 45:
                window = 7
            elif hour > 45:
                window = 8

            # i need at least 1 datapoint within 3 hours of the target hour (to take median)
            indx = (t >= hour - 0.5 * window) * (t < hour + 0.5 * window)
            # print (i, hour + window/2., indx.sum())

            # remove if less than 3 datapoints within 4 hours
            # lc_num_total.append(int(np.sum(np.ma.count(musShifted[:,j,:][:,indx], axis = 1)/indx.sum())))

            if indx.sum() < 3:
                continue

            if np.sum((np.ma.count(musShifted[:, j, :][:, indx], axis=1) != 0)) < 3.0:
                continue

            if b == 'U':
                if np.sum(np.ma.count(musShifted[:, j, :][:, indx], axis=1)) / indx.sum() < 3.0:
                    continue
            elif SNTYPE == 'Ibn' or SNTYPE == 'Ib':
                if np.sum(np.ma.count(musShifted[:, j, :][:, indx], axis=1)) / indx.sum() < 3.0:
                    continue
            elif b == 'H' and SNTYPE == 'Ic-bl':
                if np.sum(np.ma.count(musShifted[:, j, :][:, indx], axis=1)) / indx.sum() < 3.0:
                    continue

            lc_num[0].append(window)
            lc_num[1].append(hour)
            lc_num[2].append(int(np.sum(np.ma.count(musShifted[:, j, :][:, indx], axis=1) / indx.sum())))

            # weighted average weighted by errorbars within hour and hour+window

            weights = 1.0 / ((stdsShifted[:, j, :][:, indx]) ** 2)

            rollWmu[i] = np.ma.average(musShifted[:, j, :][:, indx],
                                       weights=weights)

            rollStd[i] = np.ma.std(stdsShifted[:, j, :][:, indx])

            rollWstd[i] = np.ma.average((musShifted[:, j, :][:, indx] - rollWmu[i]) ** 2,
                                        weights=weights)

            mdata = np.ma.filled(musShifted[:, j, :][:, indx], np.nan)
            rollPc25[i] = np.nanpercentile(mdata, 25)
            rollPc75[i] = np.nanpercentile(mdata, 75)
            rollMed[i] = np.nanpercentile(mdata, 50)

        # average = np.ma.average(mus[:,j,:], axis=0,
        #                         weights = 1.0/stds[:,j,:]**2)

        # variance = np.ma.average((mus[:,j,:]-average)**2, axis=0,
        #                          weights=1.0/stds[:,j,:]**2) \
        #                          * np.nansum(1.0/stds[:,j,:]**2, axis=0) \
        #                          / (np.nansum(1.0/stds[:,j,:]**2, axis=0) - 1)

        # averageShifted = np.ma.average(musShifted[:,j,:], axis=0,
        #                         weights = 1.0/stdsShifted[:,j,:]**2)

        # varianceShifted = np.ma.average((musShifted[:,j,:]-averageShifted)**2, axis=0,
        #                          weights=1.0/stdsShifted[:,j,:]**2) \
        #                          * np.nansum(1.0/stdsShifted[:,j,:]**2, axis=0) \
        #                          / (np.nansum(1.0/stdsShifted[:,j,:]**2, axis=0) - 1)

        pc25, Med, pc75 = np.nanpercentile(np.ma.filled(musShifted[:, j, :], np.nan), [25, 50, 75], axis=0)
        std = np.ma.std(mus[:, j, :], axis=0)
        stdShifted = np.ma.std(musShifted[:, j, :], axis=0)

        # print(rollMed)

        if np.nansum(rollMed) != 0:
            ax[b][0].legend(ncol=4)
            ax[b][0].set_ylim(-6, 3)
            ax[b][0].set_xlabel("log time")
            axv1[b][0].legend(ncol=4)
            axv1[b][0].set_ylim(-6, 3)
            axv1[b][0].set_xlabel("log time")
            # print (mus[:,:,20:40])#,stds[:,:,20:40])
            axv2[b][0].legend(ncol=4)
            axv2[b][0].set_ylim(-6, 3)
            axv2[b][0].set_xlabel("log time")
            thisfit = {'t': t,
                       'phs': phs,
                       'average': average,
                       'averageShifted': averageShifted,
                       'variance': variance,
                       'varianceShifted': varianceShifted,
                       'stdev': std,
                       'stdShifted': stdShifted,
                       'rollingWeightedAverage': rollWmu,
                       'rollingWeightedStd': rollWstd,
                       'rollingMedian': rollMed,
                       'rollingPc25': rollPc25,
                       'rollingPc75': rollPc75,
                       'median': Med,
                       'pc25': pc25,
                       'pc75': pc75,
                       'windows': lc_num[0],
                       't_lc_per_window': lc_num[1],
                       'lc_per_window': lc_num[2]}  # ,
            # 'lc_num_total':lc_num_total}

            # print(b, thisfit['averageShifted'])

            ax[b][1].plot(t, average, 'k')

            ax[b][1].fill_between(t,
                                  average - std,
                                  average + std,
                                  color='#1A5276', alpha=0.3,
                                  label="sample standard deviation")
            ax[b][1].fill_between(t,
                                  average - variance,
                                  average + variance,
                                  color='#FFC600', alpha=0.3,
                                  label="weighted variance")
            ax[b][0].set_title(SNTYPE + ", " + b)
            ax[b][1].set_xlabel("phase (days)")
            ax[b][0].set_ylabel("mag")
            ax[b][1].set_ylabel("mag")
            ax[b][1].legend()

            ax[b][0].set_xlim(-25, 55)
            ax[b][1].set_xlim(-25, 55)

            axv1[b][1].plot(t, average, 'k')

            axv1[b][1].fill_between(t,
                                    average - std,
                                    average + std,
                                    color='#1A5276', alpha=0.3,
                                    label="sample standard deviation")
            axv1[b][1].fill_between(t,
                                    average - variance,
                                    average + variance,
                                    color='#FFC600', alpha=0.3,
                                    label="weighted variance")
            axv1[b][0].set_title(SNTYPE + ", " + b)
            axv1[b][1].set_xlabel("phase (days)")
            axv1[b][0].set_ylabel("mag")
            axv1[b][1].set_ylabel("mag")
            axv1[b][1].legend()

            axv1[b][0].set_xlim(-25, 55)
            axv1[b][1].set_xlim(-25, 55)

            axv2[b][1].plot(t, averageShifted, 'k', label='Average')
            axv2[b][1].plot(phs, rollWmu, 'r', label='Rolling Average')

            axv2[b][1].fill_between(t,
                                    averageShifted - std,
                                    averageShifted + std,
                                    color='#1A5276', alpha=0.3,
                                    label="Standard deviation")
            axv2[b][1].fill_between(phs,
                                    rollWmu - rollWstd,
                                    rollWmu + rollWstd,
                                    color='#FF5733', alpha=0.3,
                                    label="Weighted standard deviation")
            axv2[b][1].fill_between(t,
                                    averageShifted - varianceShifted,
                                    averageShifted + varianceShifted,
                                    color='#FFC600', alpha=0.3,
                                    label="Weighted variance")

            axv2[b][0].set_title(SNTYPE + ", " + b)
            axv2[b][1].set_xlabel("phase (days)")
            axv2[b][0].set_ylabel("mag")
            axv2[b][1].set_ylabel("mag")
            axv2[b][1].legend()

            axv2[b][0].set_xlim(-25, 55)
            axv2[b][1].set_xlim(-25, 55)
            axv2[b][0].grid(True)
            ax[b][0].grid(True)
            axv2[b][1].grid(True)
            ax[b][1].grid(True)

            axv1[b][2].plot(t, averageShifted, 'k')

            axv1[b][2].fill_between(t,
                                    averageShifted - std,
                                    averageShifted + std,
                                    color='#1A5276', alpha=0.3,
                                    label="sample standard deviation")
            axv1[b][2].fill_between(t,
                                    averageShifted - varianceShifted,
                                    averageShifted + varianceShifted,
                                    color='#FFC600', alpha=0.3,
                                    label="weighted variance")

            axv1[b][0].set_title(SNTYPE + ", " + b)
            axv1[b][2].set_xlabel("phase (days)")
            axv1[b][0].set_ylabel("mag")
            axv1[b][1].set_ylabel("mag")
            axv1[b][2].set_ylabel("mag")
            axv1[b][1].legend()

            axv1[b][0].set_xlim(-25, 55)
            axv1[b][1].set_xlim(-25, 55)
            axv1[b][2].set_xlim(-25, 55)
            axv1[b][0].grid(True)
            axv1[b][1].grid(True)
            axv1[b][2].grid(True)
            # pkl.dump(thisfit,
            #          open("outputs/GPs_2022/GPalltemplfit_%s_%s_rm_07rz.pkl" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            #          else b), "wb"))

            # pl.figure()

            axs_com[b][1].plot(phs, rollWmu, 'k', label='Rolling Weighted Average')
            axs_com[b][1].plot(phs, rollMed, 'r', label='Rolling Median')
            axs_com2[b].plot(phs, rollMed, 'k', linewidth=5, label='Rolling Median', zorder=10)

            axs_com[b][1].fill_between(phs,
                                       rollWmu - rollWstd,
                                       rollWmu + rollWstd,
                                       color='#1A5276', alpha=0.3,
                                       label="Rolling Weighted Standard Deviation")
            axs_com[b][1].fill_between(phs,
                                       rollPc25,
                                       rollPc75,
                                       color='#FFC600', alpha=0.3,
                                       label="75 Percentile")
            axs_com2[b].fill_between(phs,
                                     rollPc25,
                                     rollPc75,
                                     color='black', alpha=0.3,
                                     label="75 Percentile", zorder=9)
            axs_com[b][0].set_title('Comparing templates for ' + SNTYPE + ", " + b)
            axs_com[b][1].set_xlabel("phase (days)")
            axs_com[b][1].set_ylabel("mag")
            axs_com[b][1].legend()
            axs_com[b][0].set_ylabel("mag")

            axs_com[b][0].legend()
            axs_com[b][1].set_xlim(-25, 55)
            axs_com[b][1].grid(True)
            axs_com[b][0].set_xlim(-25, 55)
            axs_com[b][0].grid(True)
            axs_com2[b].set_title('Finding outliers for ' + SNTYPE + ", " + b, size=35)
            axs_com2[b].set_xlabel("phase (days)", size=30)
            axs_com2[b].set_ylabel("mag", size=30)
            axs_com2[b].legend(ncol=4, prop={'size': 20})

            axs_com2[b].set_xlim(-25, 55)
            axs_com2[b].grid(True)

            # pl.show()
            # print (variance)
            # print (average)
            # figs[j].show()

            # figs[j].savefig("outputs/GPs_2022/GPalltemplfit_%s_%s_V0_2022.png" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            # else b))
            # figsv1[j].savefig("outputs/GPs_2022/GPalltemplfit_%s_%s_V1_2022.pdf" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            # else b))
            # figsv2[j].savefig("outputs/GPs_2022/GPalltemplfit_%s_%s_V2_2022.png" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            # else b))
            # figs_com[j].savefig(
            #     "outputs/GPs_2022/GPalltemplfit_%s_%s_compare_2022.png" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            #     else b))
            # figs_com2[j].savefig(
            #     "outputs/GPs_2022/GPMedtemplfit_%s_%s_outlier_2022_2.png" % (SNTYPE, b + 'p' if b in ['u', 'r', 'i']
            #     else b))
            # os.system("pdfcrop outputs/GPalltemplfit_%s_%s_V1.pdf /Users/fbianco/science/Dropbox/papers/SESNtemplates.working/figs/GPalltemplfit_%s_%s_V1.pdf"%(SNTYPE,bb,SNTYPE,bb))
        else:
            print('No Rolling Median was found for the GP templates of subtype' , SNTYPE, ' in band ', b)
