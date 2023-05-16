import os
import sys
import pickle as pkl
from IPython import embed

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import george
from george import kernels
import emcee

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils

# plt.style.use('~/GitHub/custom-matplotlib/custom.mplstyle')
# plt.rcParams['figure.figsize'] = (10, 10)

# Loading CfA SN lightcurves

# Setting parameters for lcv reader
# use literature data (if False only CfA data)
LIT = True

# Use NIR data too
FNIR = True

# Whether or not to do Gaussian process regression with Scipy BFGS optimizer or
# with EMCEE MCMC optimzation.
FITGP_opt = False
FITGP_MCMC = True


def der(xy):
    xder, yder = xy[1], xy[0]
    diffx, diffy = np.diff(yder),  np.diff(xder)
    return np.array([diffy / diffx, xder[:-1] + diffx * 0.5])


def nll(p, y,xx, x,yerr, gp, inds):
    """Returns -gp.lnlikelihood + a smoothness term."""
    gp.set_parameter_vector(p)

    gp.compute(np.log(xx + 30), yerr)
    # Calculate smoothness of the fit
    pred = gp.predict(y, x)[0]
    pred_d1 = der([pred, x])
    pred_d2 = der(pred_d1)
    smoothness = np.nansum(np.abs(pred_d2), axis=1)[0]

    # Replace with large value if smoothness is inf or nan.
    if np.isinf(smoothness) or np.isnan(smoothness):
        smoothness = 1e25

    ll = gp.lnlikelihood(y, quiet=True) #np.sum((y - pred[inds]**2)) #
    ll -= smoothness
    return -ll if np.isfinite(ll) else 1e25


def grad_nll(p, y, x, gp):
    """Returns -gp.grad_lnlikelihood"""
    gp.set_parameter_vector(p)
    return -gp.grad_lnlikelihood(y, quiet=True)


def lnprior(p):
    const = p[0]
    exp = p[1]

    if (-5 < const < 5) and (-5 < exp < 5):
        return 0.0
    else:
        return -np.inf


def lnprob(p, y,xx, x,yerr, gp, inds):
    lp = lnprior(p)

    if np.isinf(lp):
        return -np.inf

    return nll(p, y,xx, x,yerr, gp, inds)


# If any command line arguments are supplied, assume they are supernova names.
# All supernova included on the command line will be analyzed. If none are
# provided to the command line, analyze all supernovae available.
if len(sys.argv) > 1:
    sne = sys.argv[1:]
else:
    env = os.getenv("SESNCFAlib")
    sne = pd.read_csv(env + "/SESNessentials.csv")['SNname'].values

# Set up SESNCfalib stuff
su = templutils.setupvars()
nbands = len(su.bands)

# Loop through each provided supernova.
for sn in sne:
    # Set this flag for use later on
    bad_sn = False

    # Creates a folder within outputs so that plots are more organized.
    if not os.path.exists(f"outputs/{sn}"):
        os.makedirs(f"outputs/{sn}")

    # If you WANT to overwrite your plots, then comment out this else
    # block.
    # else:
    #     print(f"{sn} folder already exists. Skipping this sn.")
    #     continue

    # Read and set up SN and look for photometry files
    try:
        # WWW I added this if block because the # at the start of some sn names
        # were causing some sn names to not be recognized.
        if sn[0] == "#":
            sn = sn[1:]

        thissn = snstuff.mysn(sn, addlit=True)
    except AttributeError:
        continue

    # WWW This seems to have no bearing on the script, so I will start adding
    # whether or not a sn was a "bad sn" to the plots.
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        print("Bad Supernova.")
        bad_sn = True

    # Read metadata for sn and print out a whole bunch of stuff.
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    # thissn.printsn()

    # Check sn is ok, and load data.
    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        print("Bad Supernova.")
        bad_sn = True

    print("Starting loading...")

    lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
    # print(lc)
    thissn.setphot()
    thissn.getphot()
    thissn.setphase()
    thissn.sortlc()
    # print(thissn.photometry)
    # thissn.printsn()  # Don't really need another printsn

    # WWW What does this mean:
    # check that it is k
    if np.array([n for n in thissn.filters.values()]).sum() == 0:
        print("Bad Supernova.")
        bad_sn = True

    for b in su.bands:

        # if not b=='V':
        #     continue

        # WWW assumedly this is checking how many lightcurve datapoints there
        # are for each band.
        if thissn.filters[b] == 0:
            continue



        xmin = thissn.photometry[b]['mjd'].min()

        # print(xmin)

        # WWW Removed this line of code and replaced it with the
        # following block of if statements on suggestion from Somayeh.
        # x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5

        if xmin - thissn.Vmax < -1000:
            x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5
        elif xmin - thissn.Vmax > 1000:
            x = thissn.photometry[b]['mjd'] - thissn.Vmax - 2400000.5
        else:
            x = thissn.photometry[b]['mjd'] - thissn.Vmax

        y = thissn.photometry[b]['mag']
        y = y.min() - y
        yerr = thissn.photometry[b]['dmag']

        inds = np.where(np.array((x < 100) & (x > -20)))[0]
        y = y[inds]
        yerr = yerr[inds]
        x = x[inds]



        if len(y) < 5:
            continue



        

        if b in ['u', 'r', 'i']:
            templatePkl = f"ubertemplates/UberTemplate_{b}p.pkl"
        else:
            templatePkl = f"ubertemplates/UberTemplate_{b}.pkl"

        # Somayeh changed: The ubertemplates are being read this way:
        with open(templatePkl, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            tmpl = u.load()

        tmpl['mu'] = -tmpl['mu']
        print("Template for the current band", templatePkl)

        t = np.linspace(x.min(), x.max(), 100)

        ####################################

        tmpl['musmooth'] = -tmpl['spl_med'](tmpl['phs'])
        def meansmooth(x): return -tmpl['spl_med'](x) + tmpl['spl_med'](0)


        if x[0] >0:
            delta_y = y[0] - meansmooth(x)[0] 
            y = y - delta_y
            tmpl_x = meansmooth(x) #- max(meansmooth(t))
            tmpl_t = meansmooth(t)#- max(meansmooth(t))
           
        else:

            # Fixing the vertical alignment issue for those SNe with a pre-shock

            y_min = y[np.argmin(np.abs(x))]
            y = y - y_min
            tmpl_x = meansmooth(x) #- max(meansmooth(t))
            tmpl_t = meansmooth(t)#- max(meansmooth(t))


        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        fig.suptitle(f"{sn} band {b}", fontsize=16)

        axes[0, 1].errorbar(x, y, yerr=yerr, fmt='k.')
        axes[1, 0].errorbar(x, y, yerr=yerr, fmt='k.')
        axes[1, 1].errorbar(np.log(x + 30), y, yerr=yerr, fmt='k.')



        axes[0, 0].errorbar(x, y, yerr=yerr, label="data")
        axes[0, 0].plot([-25, 95], [0, 0], 'k-', alpha=0.5, lw=2)
        axes[0, 0].plot(x, y - meansmooth(x), label="residuals")
        axes[0, 0].plot(x, meansmooth(x), 'ko')
        axes[0, 0].plot(t, meansmooth(t), label="median")
        axes[0, 0].legend(fontsize=15)
        axes[0, 0].set_ylim(-3, 2)
        axes[0, 0].set_xlim(-30, 100)
        axes[0, 0].set_title("residuals")
        axes[0, 0].set_ylabel("normalized mag")
        axes[0, 0].set_xlabel("time (days since peak)")
        axes[0, 0].axvline(x=0, c="k", ls=":")

        # Set up the Gaussian process with the optimized hyperparameters:
        # WWW It seems like the george kernels are always working in log space.
        # The ConstantKernel accepts the parameter log_constant, so the
        # parameter we are giving it is already in log space. However, the
        # parameter for ExpSquaredKernel is not in log space (for some reason)
        # however it reports the parameter in get_parameter_vector in log space.
        # So that is why we are setting expSq_param to the parameter in log
        # space, but then passing ExpSquaredKernel the exponential of that.
        const_param = -1.10
        expSq_param = -1.73
        k1 = kernels.ConstantKernel(const_param)
        k2 = kernels.ExpSquaredKernel(np.exp(expSq_param))
        kernel = kernels.Product(k1, k2)
        gp = george.GP(kernel)

        # Testing robustness of optimization by changing p0 to this
        # gp.kernel[0] = -1
        # gp.kernel[1] = -1

        p0 = gp.kernel.parameter_vector
        done = False
        try:
            # Compute the GP on the optimized hyperparameters
            gp.compute(np.log(x + 30), yerr)
            done = True
        except ValueError:
            continue

        if t[0] > -15:
            # Adding a point at -15
            # WWW Why?
            t = np.concatenate([np.array([-15]), t])

        # Predicting with the GP trainied on the optimized hyperparameters
        mu, cov = gp.predict(y - meansmooth(x), np.log(t+30))
        std = np.sqrt(np.diag(cov))

        # Assert statement is trying to figure out if this line is needed
        # p0 = gp.kernel.parameter_vector
        assert np.all(p0 == gp.kernel.parameter_vector)

        axes[0, 1].set_title(f"pars: {p0[0]:.3f} {p0[1]:.3f}")
        axes[0, 1].plot(t, mu + meansmooth(t), 'DarkOrange', lw=2)
        axes[0, 1].fill_between(t,
                                mu - std + meansmooth(t),
                                mu + std + meansmooth(t),
                                color='grey', alpha=0.3)
        axes[0, 1].axvline(x=0, c="k", ls=":")

        # New optimization of the hyperparameters:
        if FITGP_opt:
            args = (y - meansmooth(x), np.log(t+30), gp)
            results = opt.minimize(nll, p0,
                                   jac=grad_nll, args=args,
                                   method="bfgs")
            gp.kernel.parameter_vector = results.x
            print(results)
            print("hyperparameters: ", gp.kernel.get_parameter_vector())
            print("loglikelihood", gp.lnlikelihood(y))

        elif FITGP_MCMC:
            # Set up the args that will be passed to lnprob
            args = (y - meansmooth(x),x, np.log(t+30),yerr, gp, inds)

            # Position of initial walkers will be the initial optimized
            # hyperparameters, plus some small randomness.
            pos = p0 + 1 * np.random.randn(100, 2)
            nwalkers, ndim = pos.shape

            # Intiailize the ensemble sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=args)

            # Burn-in.
            burn_state = sampler.run_mcmc(pos, 10, progress=True)
            sampler.reset()  # Reset "bookkeeping vars" according to docs. idk
            print("Burn-in Complete")

            # Full run
            niter = 200
            sampler.run_mcmc(burn_state, niter, progress=True)


            mu_af = np.mean(sampler.acceptance_fraction)
            print(f"Mean acceptance fraction: {mu_af:.3f}")

            samples = sampler.get_chain(discard=1)
            optpars = np.median(samples.reshape(samples.shape[0]*
                                                samples.shape[1],
                                                2), axis=0)

            gp.kernel.parameter_vector = optpars

            print(optpars)

            # mu_ac = np.mean(sampler.get_autocorr_time())
            # print(f"Mean autocorrelation time: {mu_ac:.3f} steps")

            # plt.clf()
            # fig, axes = plt.subplots(2, figsize=(12, 6), sharex=True)
            # samples = sampler.get_chain()
            # labels = ["Constant", "Squared Exponential"]
            # for i in range(ndim):
            #     ax = axes[i]
            #     ax.plot(samples[:, :, i], "k", alpha=0.3)
            #     ax.set_xlim(0, len(samples))
            #     ax.set_ylabel(labels[i])
            #     ax.yaxis.set_label_coords(-0.1, 0.5)
            # axes[-1].set_xlabel("step number")
            # plt.savefig("outputs/walkers.pdf")
            # plt.show()

            # samples = sampler.get_chain(flat=True)
            # fig, axes = plt.subplots(nrows=1, ncols=2,
            #                          sharex=False, sharey=False)
            # axes[0].hist(samples[:, 0], 100, color="k", histtype="step")
            # axes[0].set_xlabel("Log Constant")
            # axes[0].set_ylabel("Probability")

            # axes[1].hist(samples[:, 1], 100, color="k", histtype="step")
            # axes[1].set_xlabel("Log Exp Squared")
            # axes[1].set_ylabel("Count")
            # plt.savefig("/Users/admin/GitHub/testmcmc.png")

            # embed()
            # quit()


        # Compute the GP with the new parameters.
        gp.compute(np.log(x+30), yerr)
        mu, cov = gp.predict(y - meansmooth(x), np.log(t+30))
        std = np.sqrt(np.diag(cov))
        p1 = gp.kernel.parameter_vector

        if FITGP_opt or FITGP_MCMC:
            axes[1,0].set_xlabel("time (days since peak)")
            axes[1,0].plot(t, mu + meansmooth(t), 'r', lw=2,
                     label="%.2f %.2f"%(p1[0], p1[1]))

            axes[1,0].fill_between(t,
                             mu + meansmooth(t) - std,
                             mu + meansmooth(t) + std ,
                             color='grey', alpha=0.3)

            axes[1,0].set_ylabel("normalized mag")
            axes[1,0].legend(loc="upper right")
            axes[1,0].set_ylim(axes[0,1].get_ylim())
            axes[1,0].set_xlim(axes[0,1].get_xlim())

            axes[1,0].axvline(x=0, c="k", ls=":")

            axes[1,1].set_xlabel("log time")

            axes[1,1].plot(np.log(t+30), mu + meansmooth(t), 'r', lw=2)
            axes[1,1].fill_between(np.log(t+30),
                             mu + meansmooth(t) - std,
                             mu + meansmooth(t) + std ,
                             color='grey', alpha=0.3)

            # # Subracting the mean and fitting GP to residuals only

            #spl = InterpolatedUnivariateSpline(templ.phs, ysmooth)


            xl = plt.xlabel("log time (starting 30 days before peak)")
            plt.savefig(f"outputs/{sn}/GPfit{sn}_{b}.png")
        else:
            pkl.dump((y, gp, tmpl['spl_med']), open(f"outputs/{sn}/GPfit{sn}_{b}.pkl", "wb"))
