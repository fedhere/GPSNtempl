import os
import pylab as pl
import matplotlib.pyplot as plt
import sys
import pickle as pkl
import pandas as pd
import numpy as np
import traceback
import scipy.optimize as op
import george
from george import kernels

# s = json.load( open(str(os.getenv ('PUI2015'))+"/fbb_matplotlibrc.json") )
# pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore")

mpl.use('agg')

pl.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# # Loading CfA SN lightcurves

# setting parameters for lcv reader
# use literature data (if False only CfA data)
LIT = True
# use NIR data too
FNIR = True
FITGP = False

fast_evolving = ['sn2015U', 'sn2010et', 'sn2015ap', 'sn2019aajs', 'PS1-12sk',
                 'sn2018bcc', 'sn2019rii', 'sn2019deh', 'sn2019myn',
                 'LSQ13ccw', 'sn2010jr', 'sn2011dh', 'sn2006aj', 'ASASSN-14ms',
                 'sn2010x', 'sn2007qv']  # , 'sn2016hgs', 'sn2014L']
su = templutils.setupvars()
nbands = len(su.bands)
coffset = su.coffset


def fun(x):
    print(x)
    return -math.exp(-np.dot(x, x))


def der(xy):
    xder, yder = xy[1], xy[0]
    # print ("here ", yder[1] - yder[:-1])
    np.diff(yder) / np.diff(xder)
    return np.array([np.diff(yder) / np.diff(xder), xder[:-1] + np.diff(xder) * 0.5])


def nll(p, y, x, gp, s):
    gp.kernel.parameter_vector = p

    # gp.kernel[0] = p[0]
    # gp.kernel[1] = p[1]
    # gp.set_parameter_vector(p)
    # print(gp.get_parameter_vector)
    # print('lengths: ',len(np.log(xx + 30)), len(yerr))
    # gp.compute(np.log(xx + 30), yerr)

    # Calculate smoothness of the fit
    # pred = gp.predict(y, x)[0]
    # print(pred)
    # print(x)
    try:
        smoothness = (np.nansum(np.abs(der(der([gp.predict(y, x)[0], x]))), axis=1)[0])
        smoothness = smoothness if np.isfinite(smoothness) \
                                   and ~np.isnan(smoothness) else 1e25
    except np.linalg.LinAlgError:
        smoothness = 1e25

    # print(gp.log_likelihood(y, squiet=True), smoothness)

    ll = gp.log_likelihood(y, quiet=True)  # - (smoothness) #np.sum((y - pred[inds]**2)) #
    ll -= (smoothness) ** s

    # print (p, -ll if np.isfinite(ll) else 1e25)
    return -ll if np.isfinite(ll) else 1e25


def nll_early(p, y, x, gp):
    gp.set_parameter_vector(p)
    ll = gp.log_likelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


def nll_late(p, y, x, gp):
    # gp.kernel.parameter_vector = p

    gp.set_parameter_vector(p)
    try:
        smoothness = (np.nansum(np.abs(der(der([gp.predict(y, x)[0], x]))), axis=1)[0])
        print('smoothness', smoothness)
        smoothness = smoothness if np.isfinite(smoothness) \
                                   and ~np.isnan(smoothness) else 1e25
    except np.linalg.LinAlgError:
        smoothness = 1e25

    ll = gp.log_likelihood(y, quiet=True)  # - (smoothness) #np.sum((y - pred[inds]**2)) #
    ll -= (smoothness) ** 0.25
    return -ll if np.isfinite(ll) else 1e25


def grad_nll(p, y, x, gp):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel.parameter_vector = p
    # smoothness = der(der([gp.predict(y,x)[0], x]))[0]
    # Update the kernel parameters and compute the likelihood.
    return -gp.grad_lnlikelihood(y, quiet=True)


perType = True
pars = {'Ib': [-2.44, -1.01],
        'Ib-Ca-rich': [-2.44, -1.01],
        'Ibn': [-0.06, -2.95],
        'IIb': [-2.43, -1.28],
        'Ic': [-1.66, -0.82],
        'Ic-bl': [-1.18, -1.41],
        'Ib-c': [-1.958, -1.278]}
tp = 'Ic'  # 'Ib' # 'Ibn' #'Ic' #'IIb' #
par1 = pars[tp][0]
par2 = pars[tp][1]

if __name__ == '__main__':
    # uncomment for all lcvs to be read in
    if len(sys.argv) > 1:

        if perType:
            allsne_list = pd.read_csv(os.getenv("SESNCFAlib") +
                                      "/SESNessentials.csv", encoding="ISO-8859-1")
            allsne = [sys.argv[1]]
            if allsne_list[allsne_list.SNname == allsne[0]].Type.values[0] == tp:
                pass
            else:
                if FITGP:
                    pass
                else:
                    tp = allsne_list[allsne_list.SNname == allsne[0]].Type.values[0]
                    par1 = pars[tp][0]
                    par2 = pars[tp][1]

        else:
            allsne = [sys.argv[1]]



    else:

        if perType:

            allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                                 "/SESNessentials.csv", encoding="ISO-8859-1")
            allsne = allsne.SNname[allsne.Type == tp].values

        else:
            allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                                 "/SESNessentials.csv", encoding="ISO-8859-1")['SNname'].values
    # print (allsne)

    # set up SESNCfalib stuff

    # errorbarInflate = {"93J":30,
    #                   "05mf":1}

    all_params = {}

    for sn in allsne:

        all_params[sn] = {}
        # if sn in fast_evolving:
        #     s = 0.25
        # else:
        #     s = 1

        # read and set up SN and look for photometry files
        try:
            thissn = snstuff.mysn(sn, addlit=True)
        except AttributeError:
            continue
        if len(thissn.optfiles) + len(thissn.fnir) == 0:
            print("bad sn")

        # read metadata for SN
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)
        thissn.printsn()

        # check SN is ok and load data
        if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
            print("bad sn")
        print(" starting loading ")
        # print (os.environ['SESNPATH'] + "/finalphot/*" + \
        #        thissn.snnameshort.upper() + ".*[cf]")
        # print (os.environ['SESNPATH'] + "/finalphot/*" + \
        #        thissn.snnameshort.lower() + ".*[cf]")

        # print( glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
        #                  thissn.snnameshort.upper() + ".*[cf]") + \
        #        glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
        #                  thissn.snnameshort.lower() + ".*[cf]") )

        lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
        thissn.setphot()
        thissn.getphot()
        thissn.setphase()

        thissn.sortlc()
        thissn.printsn()

        # print (lc)

        # check that it is k
        if np.array([n for n in thissn.filters.values()]).sum() == 0:
            print("bad sn")

        for b in su.bands:

            # if thissn.type == 'Ibn' or b in ['U', 'u', 'V', 'B', 'w1', 'w2']:
            #     s = 0.25
            # else:
            #     s = 1
            s = 1

            thissn.getmagmax(band=b, forceredo=True)

            bb = b
            if b == 'i':
                bb = 'ip'
            if b == 'u':
                bb = 'up'
            if b == 'r':
                bb = 'rp'

            all_params[sn][b] = []

            if thissn.filters[b] == 0:
                print(b, thissn.filters[b])
                continue

            templatePkl = "ubertemplates/UberTemplate_%s.pkl" % (b + 'p' if b in ['u', 'r', 'i']
                                                                 else b)
            # tmpl = pkl.load(open(templatePkl, "rb"))

            # Somayeh changed: The ubertemplates are being read this way:
            with open(templatePkl, 'rb') as f:
                u = pkl._Unpickler(f)
                u.encoding = 'latin1'
                tmpl = u.load()

            tmpl['mu'] = -tmpl['mu']
            tmpl['musmooth'] = -tmpl['spl_med'](tmpl['phs'])
            meansmooth = lambda x: -tmpl['spl_med'](x)  # + tmpl['spl_med'](0)

            # print ("Template for the current band", templatePkl)

            # print('Vmax is')
            # print (thissn.photometry[b]['mjd'] - thissn.Vmax)

            xmin = thissn.photometry[b]['mjd'].min()

            # print(thissn.maxmags[b])

            # if np.isnan(thissn.maxmags[b]['epoch']):
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

            y = -thissn.photometry[b]['mag']

            yerr = thissn.photometry[b]['dmag']

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
            elif b in ['K', 'H', 'J']:
                up_lim = 80
            else:
                low_lim = -20
                up_lim = 100

            y = y[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]
            yerr = yerr[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]
            x = x[np.where(np.array((x < up_lim) & (x > low_lim)))[0]]

            if len(x) == 0:
                print(b, ': No data points within the limits.')
                continue

            t = np.linspace(x.min(), x.max(), 100)

            ####################################
            # Fixing the vertical alignment for incomplete SNe LC starting after epoch=0:
            if x[0] > 0:
                delta_y = y[0] - meansmooth(x)[0]
                y = y - delta_y
                tmpl_x = meansmooth(x)
                tmpl_t = meansmooth(t)
            # Fixing the vertical alignment for incomplete SNe LC ending before epoch=0:
            elif x[-1] < 0:
                delta_y = y[-1] - meansmooth(x)[-1]
                y = y - delta_y
                tmpl_x = meansmooth(x)
                tmpl_t = meansmooth(t)


            else:

                # Fixing the vertical alignment issue for those SNe with a pre-shock:
                if (x[np.argmax(y)]) < -10:
                    y_min = y[np.argmin(np.abs(x))]
                    y = y - y_min
                # else
                else:
                    y_min = y[np.argmin(np.abs(x))] - meansmooth(x)[np.argmin(np.abs(x))]
                    y = y - y_min

                # y = y #-  y[np.argmin(np.abs(x))]
                # y = y #- max(meansmooth(t))
                tmpl_x = meansmooth(x) - meansmooth(0)  # max(meansmooth(t))
                tmpl_t = meansmooth(t) - meansmooth(0)  # max(meansmooth(t))

            if len(y) < 5:
                print(b, ': Error! Less than 5 data points were found!')
                continue

            fig = pl.figure()  # figsize=(20,20))
            if FITGP:
                ax1 = fig.add_subplot(223)
                ax2 = fig.add_subplot(224)
                ax00 = fig.add_subplot(221)
                ax01 = fig.add_subplot(222)
                fig.suptitle("%s band %s" % (sn, b), fontsize=16)
                ax01.errorbar(x, y, yerr=yerr, fmt='k.')
                ax1.errorbar(x, y, yerr=yerr, fmt='k.')
                ax2.errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.')
                ax00.set_title("residuals")
            else:
                ax1 = fig.add_subplot(212)
                ax00 = fig.add_subplot(221)
                ax01 = fig.add_subplot(222)
                # fig.suptitle("%s band %s"%(sn, b), fontsize=16)
                ax01.errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.')
                ax1.errorbar(x, y, yerr=yerr, fmt='k.')
                ax1.grid()
                # ax00.grid()
                # ax01.grid()

            # We need to make sure that the new x is either defined within the old x boundaries
            # or within [-20,100] if the min and max of the old x go beyond -20 and 100:

            # print(b ,max(meansmooth(t)))
            major_xticks = ([0, 30, 60, 90])
            ax00.plot(t, tmpl_t, color="#ca6200", label="Ibc temp")
            ax00.errorbar(x, y, yerr=yerr, color="#0571b0", label="LC")
            ax00.plot([-25, 95], [0, 0], 'k-', alpha=0.5, lw=2)
            ax00.plot(x, y - tmpl_x, color="#008837", label="Residual", zorder=10)
            # ax00.plot(x, tmpl_x, 'ko')

            ax00.axvline(0, color='grey', alpha=0.5)
            ax00.legend(fontsize=20, ncol=2, handlelength=0.5, handletextpad=0.1, columnspacing=0.8,
                        loc='upper right')
            ax00.set_ylim(-3.3, 2.3)
            ax00.set_xlim(-30, 100)
            ax00.set_ylabel("Relative Magnitude", size=25)
            ax00.set_xlabel("Time (days since peak)", size=25)
            ax00.set_xticks(major_xticks)

            # Set up the Gaussian process with the optimized hyperparameters:

            kernel = kernels.Product(kernels.ConstantKernel(-1.10), kernels.ExpSquaredKernel(-1.73))

            gp = george.GP(kernel)

            if not FITGP:
                if perType:
                    gp.kernel[0] = par1
                    gp.kernel[1] = par2
                else:

                    gp.kernel[0] = -1.958  # -1.81#-2.0#2.02#-1.10
                    gp.kernel[1] = -1.278  # -1.47#-1.68#-0.93#-1.73
            else:
                gp.kernel[0] = -1.10
                gp.kernel[1] = -1.73
            # gp.kernel[2] = -1.73

            p0 = gp.kernel.parameter_vector

            done = False

            # try:
            gp.compute(np.log(x + 30), yerr)
            done = True

            # except ValueError:
            #      traceback.print_exc()
            #      continue
            # if t[0]>-15:
            #      #adding a point at -15
            #      t=np.concatenate([np.array([-15]),t])
            #      tmpl_t = meansmooth(t)

            # print(b, x, tmpl_x)
            mu, cov = gp.predict(y - tmpl_x, np.log(t + 30))
            std = np.sqrt(np.diag(cov))

            # print("loglikelihood1: ", gp.lnlikelihood(y- tmpl_x))

            p0 = gp.kernel.parameter_vector
            # print(x)
            # print("loglikelihood1", gp.log_likelihood(y))
            if FITGP:
                ax01.set_title("pars: %.2f %.2f" % (p0[0], p0[1]))

                ax01.plot(t, mu + tmpl_t, 'red', lw=2)
                ax01.fill_between(t, mu - std + tmpl_t,
                                  mu + std + tmpl_t, color='grey', alpha=0.3)
                ax01.set_title("pars: %.2f %.2f" % (p0[0], p0[1]))
            else:
                # ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))
                major_xticks = (np.linspace(min(np.log(t + 30)), max(np.log(t + 30)), 4)).round(decimals=1)
                major_yticks = (np.linspace(min(mu + tmpl_t) - 1, max(mu + tmpl_t) + 1, 5)).astype(int)

                ax01.plot(np.log(t + 30), mu + tmpl_t, 'red', lw=2)
                ax01.text(0.05, 0.05, r"${\theta}_{1}, {\theta}_{2}$" + ": %.2f %.2f " % (p0[0], p0[1]),
                          size=20, transform=ax01.transAxes)
                ax01.fill_between(np.log(t + 30), mu - std + tmpl_t,
                                  mu + std + tmpl_t, color='grey', alpha=0.3)
                # ax01.set_title("pars: %.2f %.2f"%(p0[0], p0[1]))
                # ax01.legend(fontsize=15)
                ax01.set_xlabel("Log Time (+30 days)", size=25)
                ax01.set_xticks(major_xticks)
                ax01.set_ylim(min(mu + tmpl_t)-.3, max(mu + tmpl_t)+0.3)

                major_xticks = ([-20, 0, 20, 40, 60, 80, 100])

                ax1.plot(t, mu + tmpl_t, 'red', lw=2, label=thissn.snnameshort + ' ' + b + ', ' + thissn.type)
                ax1.fill_between(t, mu - std + tmpl_t,
                                 mu + std + tmpl_t, color='grey', alpha=0.3)
                ax1.legend(fontsize=20)
                ax1.set_ylabel("Relative Magnitude", size=25)
                ax1.set_xlabel("Time (days since peak)", size=25)
                ax1.set_xticks(major_xticks)
                ax1.set_ylim(min(mu + tmpl_t) - .3, max(mu + tmpl_t) + 0.3)
                ax1.set_xlim(-25, 105)

                for tick in ax00.xaxis.get_major_ticks():
                    tick.label.set_fontsize(25)
                for tick in ax01.xaxis.get_major_ticks():
                    tick.label.set_fontsize(25)
                for tick in ax1.xaxis.get_major_ticks():
                    tick.label.set_fontsize(25)

                for tick in ax00.yaxis.get_major_ticks():
                    tick.label.set_fontsize(25)
                for tick in ax01.yaxis.get_major_ticks():
                    tick.label.set_fontsize(25)
                for tick in ax1.yaxis.get_major_ticks():
                    tick.label.set_fontsize(25)

            # Optimizing the hyper parameters:
            if FITGP:
                try:
                    # results = op.minimize(nll_early(p0, y[x<10] - tmpl_x[x<10], np.log(t[t<10] + 30), gp)+
                    #                       nll_late(p0, y[x>10] - tmpl_x[x>10], np.log(t[t>10] + 30), gp), x0 = p0)
                    bounds = ((None, None), (None, None), (-3, 3))
                    results = op.minimize(nll, [p0[0], p0[1]],
                                          args=(y - tmpl_x,
                                                np.log(t + 30), gp, s))
                    # results = op.minimize(nll, p0,
                    #                       args=(y - tmpl_x,
                    #                             np.log(t + 30), gp))

                    #    # Update the kernel and print the final log-likelihood.
                    gp.kernel.parameter_vector = results.x
                    #    gp.kernel.parameter_vector = results.x[:2]

                    # s_res = results.x[2]

                    all_params[sn][b].append(results.x)

                    # print(b, results.x)
                except:  # Runtime.Error:
                    traceback.print_exc()
                    print('got error')
                    # pl.savefig("GPfit%s_%s.png"%(sn,b))
                    continue
                # print ("hyper parameters: ", gp.kernel)
                # print("loglikelihood2: ", gp.lnlikelihood(y- tmpl_x))

            gp.compute(np.log(x + 30), yerr)
            mu, cov = gp.predict(y - tmpl_x, np.log(t + 30))
            std = np.sqrt(np.diag(cov))
            p1 = gp.kernel.parameter_vector
            # print (p1)

            if FITGP:
                ax1.set_xlabel("time (days since peak)")
                # ax1.plot(t, mu + tmpl_t, 'r', lw=2,
                # label="%.2f %.2f %.2f %s"%(p1[0], p1[1],s_res, results.success))
                ax1.plot(t, mu + tmpl_t, 'r', lw=2,
                         label="%.2f %.2f %s" % (p1[0], p1[1], results.success))

                ax1.fill_between(t,
                                 mu + tmpl_t - std,
                                 mu + tmpl_t + std,
                                 color='grey', alpha=0.3)

                ax1.set_ylabel("normalized mag")
                ax1.legend()
                ax1.set_ylim(ax01.get_ylim())
                ax1.set_xlim(ax01.get_xlim())
                ax2.set_xlabel("log time")

                ax2.plot(np.log(t + 30), mu + tmpl_t, 'r', lw=2)
                ax2.fill_between(np.log(t + 30),
                                 mu + tmpl_t - std,
                                 mu + tmpl_t + std,
                                 color='grey', alpha=0.3)

                # # Subracting the mean and fitting GP to residuals only

                # spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)

                xl = pl.xlabel("log time (starting 30 days before peak)")
                pl.savefig("outputs/test1/GPfit%s_%s_opt_test2.png" % (sn, bb))

                # pkl.dump(all_params, open("outputs/test1/all_params_scipy_opt_test.pkl", "wb"))
                # pkl.dump((y, gp, tmpl['spl_med']), open("outputs/test1/GPfit%s_%s.pkl" % (sn, bb), "wb"))



            else:
                fig.tight_layout()
                print('Saving the plots...')
                pl.savefig("outputs/GPfit%s_%s_no_gpopt_2022.pdf" % (sn, bb))
                # pl.savefig("outputs/GPfit%s_%s_no_gpopt_2022.png" % (sn, bb))
                pkl.dump((y, gp, tmpl['spl_med']), open("outputs/GPfit%s_%s.pkl" % (sn, bb), "wb"))
