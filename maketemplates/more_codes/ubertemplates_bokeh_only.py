import os
import pickle as pkl
import sys

import bokeh
import numpy as np
import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import BoxZoomTool, HoverTool, ResetTool, TapTool
from bokeh.models import ColumnDataSource
from bokeh.models.callbacks import CustomJS
from bokeh.plotting import Figure as bokehfigure
from bokeh.plotting import figure as bokehfigure
from bokeh.plotting import save as bokehsave
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib.colors import LogNorm

# print ("bokeh version", bokeh.__version__)
# , HBox, VBoxForm, BoxSelectTool, TapTool
# from bokeh.models.widgets import Select
# Slider, Select, TextInput
from bokeh.plotting import output_file
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

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

from savgol import savitzky_golay
from snclasses import *
from templutils import *
from makePhottable import *
from colors import rgb_to_hex

MINEP, MAXEP = -30, 100
archetypicalSNe = ['94I', '93J', '08D', '05bf', '04aw', '10bm', '10vgv']

colorTypes = {'IIb': 'FireBrick',
              'Ib': 'SteelBlue',
              'Ic': 'DarkGreen',
              'Ic-bl': 'DarkOrange',
              'other': 'purple'}

# prepping SN data from scratch
# (dont do that if you are testing or tuning plots)
PREP = True
# PREP = False
BOKEHIT = True
# BOKEHIT = False
font = {'family': 'normal',
        'size': 20}

# setting up snclass and other SESNCFAlib stuff
su = setupvars()

pl.rc('font', **font)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def setcolors(inputSNe):
    cm = pl.get_cmap('nipy_spectral')  # viridis')
    Nsne = len(inputSNe)
    print(Nsne)
    sncolors = [''] * Nsne
    for i in range(Nsne):
        sncolors[i] = (cm(1. * i / Nsne))
    sncolors = np.asarray(sncolors)

    np.random.seed(666)
    np.random.shuffle(sncolors)
    pkl.dump(sncolors, open("input/sncolors.pkl", 'wb'))
    return (sncolors)


class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))


def type_divider(sourcedata):
    # This function divides the sourcedata into individual sourcedata dicts containing different types

    ss_IIb = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}
    ss_Ib = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}
    ss_Ic = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}
    ss_Ic_bl = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}
    ss_Ibn = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}
    ss_others = {'id': [], 'x': [], 'y': [], 'yerr': [], 'colors': [], 'mask': []}

    for i, n in enumerate(np.unique(sourcedata['id'])):

        indx = sourcedata['id'] == n

        if np.unique(sourcedata['type'][indx])[0] == 'IIb':
            ss_IIb['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_IIb['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_IIb['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_IIb['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_IIb['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_IIb['mask'].extend(np.asarray(sourcedata['mask'][indx]))

        elif np.unique(sourcedata['type'][indx])[0] == 'Ib':
            ss_Ib['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_Ib['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_Ib['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_Ib['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_Ib['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_Ib['mask'].extend(np.asarray(sourcedata['mask'][indx]))

        elif np.unique(sourcedata['type'][indx])[0] == 'Ic':
            ss_Ic['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_Ic['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_Ic['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_Ic['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_Ic['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_Ic['mask'].extend(np.asarray(sourcedata['mask'][indx]))

        elif np.unique(sourcedata['type'][indx])[0] == 'Ic-bl':
            ss_Ic_bl['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_Ic_bl['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_Ic_bl['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_Ic_bl['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_Ic_bl['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_Ic_bl['mask'].extend(np.asarray(sourcedata['mask'][indx]))

        elif np.unique(sourcedata['type'][indx])[0] == 'Ibn':
            ss_Ibn['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_Ibn['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_Ibn['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_Ibn['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_Ibn['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_Ibn['mask'].extend(np.asarray(sourcedata['mask'][indx]))

        else:
            ss_others['id'].extend(np.asarray(sourcedata['id'][indx]))
            ss_others['x'].extend(np.asarray(sourcedata['x'][indx]))
            ss_others['y'].extend(np.asarray(sourcedata['y'][indx]))
            ss_others['yerr'].extend(np.asarray(sourcedata['yerr'][indx]))
            ss_others['colors'].extend(np.asarray(sourcedata['colors'][indx]))
            ss_others['mask'].extend(np.asarray(sourcedata['mask'][indx]))

    return ss_IIb, ss_Ib, ss_Ic, ss_Ic_bl, ss_Ibn, ss_others


def select2ObsPerDay(data):
    dataTimeByDay = (data['x']).astype(int)
    # print data, dataTimeByDay
    minx = int(data['x'].min())
    # print np.unique(dataTimeByDay, return_counts=True)
    for sid in np.unique(data['id']):
        thissn = np.where((data['id'] == sid))[0]
        # print sid, np.where((data['id'] ==  sid))[0]
        bc = np.bincount(dataTimeByDay[thissn] - minx) > 2
        for i in np.arange(len(bc))[bc]:
            # print i + minx, bc[i]
            theseindx = np.where((data['id'] == sid) *
                                 (dataTimeByDay == i + minx))[0]
            choices = np.random.choice(theseindx, len(theseindx) - 2)
            # print theseindx[np.argsort(data['yerr'][theseindx])],
            # print data['yerr'][theseindx][np.argsort(data['yerr'][theseindx])]

            # print data['mask']#[thissn]
            # data['mask'][choices] = True
            # print theseindx[np.argsort(data['yerr'][theseindx])][2:] #= True
            data['mask'][theseindx[np.argsort(data['yerr'][theseindx])][2:]] = True
    return data

    sys.exit()


def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series) + 1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):  # we are forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * trend
        result.append(level + trend)
    return result


def update_xyaxis(f, rg, xy='y'):
    if xy == 'y':
        fax = f.y_range
    else:
        fax = f.x_range

        fax.start = rg[0]
        fax.end = rg[1]


def bokehplot(source, b):
    # making Bokeh plots

    # htmlout2 = "outputs2/UberTemplate_%s.html"%\
    #               (b + 'p' if b in ['u', 'r', 'i'] else b)
    # output_file(htmlout2)

    ss_IIb, ss_Ib, ss_Ic, ss_Ic_bl, ss_Ibn, ss_others = type_divider(source)

    for i, tp in enumerate(['Ib', 'IIb', 'Ic', 'Ic_bl', 'Ibn', 'others']):

        # htmlout1 = "outputs2/type_%s_%s.html"%\
        #           (tp, b + 'p' if b in ['u', 'r', 'i'] else b)
        # output_file(htmlout1)

        globals()['TOOLS%s%s' % (i, b)] = [BoxZoomTool(), TapTool(), ResetTool(), HoverTool(
            tooltips=[
                ("ID", "@id"),
                ("phase", "@x"),
                ("Delta mag", "@y"),
                ("Error", "@yerr"),
            ])]

        s = ColumnDataSource(data={})
        s.data = eval('ss_' + tp)

        globals()['p%s%s' % (i, b)] = bokehfigure(plot_width=600, plot_height=300,
                                                  tools=globals()['TOOLS%s%s' % (i, b)],
                                                  title="SNe lightcurves of type %s in band %s" % (tp, b))

        s.data['yerrx'] = []
        s.data['yerry'] = []

        miny, maxy = 0, 0

        for px, py, err, sn, c in zip(s.data['x'], s.data['y'], s.data['yerr'],
                                      s.data['id'], s.data['colors']):
            s.data['yerrx'].append([px, px])
            s.data['yerry'].append([py - err, py + err])

        globals()['p%s%s' % (i, b)].multi_line('yerrx', 'yerry', source=s, color='colors')

        globals()['p%s%s' % (i, b)].circle('x', 'y', size=5, source=s,
                                           color='grey', fill_color='colors', alpha=0.5)

        #         update_xyaxis(globals()['p%s%s' % (i,b)], (miny - 0.5, maxy + 0.5))
        update_xyaxis(globals()['p%s%s' % (i, b)], (-30., 150.), xy='x')

        globals()['p%s%s' % (i, b)].yaxis.axis_label = "relative magnitude"
        globals()['p%s%s' % (i, b)].xaxis.axis_label = "phase (days)"

    return (globals()['p0%s' % (b)], globals()['p1%s' % (b)], globals()['p2%s' % (b)], globals()['p3%s' % (b)],
            globals()['p4%s' % (b)])


def preplcvs(inputSNe, workBands):
    keys = ['mag', 'dmag', 'phase', 'name', 'type']
    allSNe = {}
    for b in workBands:
        allSNe[b] = {}
        for k in keys:
            allSNe[b][k] = []
        # 'mag': [], 'dmag': [], 'phase': [], 'name': [], 'type': []}

    if workBands == su.bands:
        # prepare stuff for latex tables to be passed to makePhottable
        bands1 = ['U', 'u', 'B', 'V', 'R', 'g', 'r', 'I', 'i']
        bands2 = ['w2', 'm2', 'w1', 'H', 'J', 'K']

        tmp1 = {}
        for b in bands1:
            tmp1[b] = {}
            tmp1[b + "[min,max]"] = {}

        tmp2 = {}
        for b in bands2:
            tmp2[b] = {}
            tmp2[b + "[min,max]"] = {}

        tmp2['Any'] = {}
        tmp2["Any[min,max]"] = {}

    # iterate over all SNe in metadata file
    # print (inputSNe)
    for f in inputSNe:
        print(f)
        if f.startswith("#"): continue
        print("\n\n####################################################\n\n\n", f)

        # read and set up SN and look for photometry files
        print(" looking for files ")
        thissn = snstuff.mysn(f, addlit=True)
        if len(thissn.optfiles) + len(thissn.fnir) == 0:
            continue

        # read metadata for SN
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)
        thissn.setVmax()

        # setting date of maximum if not in metadata
        if np.isnan(thissn.Vmax) or thissn.Vmax == 0:
            # only trust selected GP results (only one really)
            if '06gi' in thissn.snnameshort:
                try:
                    # print ("getting max from GP maybe?")
                    thissn.gp = pkl.load(open('gplcvs/' + f + \
                                              "_gp_ebmv0.00.pkl", "rb"))
                    # set to gregorian
                    if thissn.gp['maxmjd']['V'] < 2400000 and \
                            thissn.gp['maxmjd']['V'] > 50000:
                        thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                    else:
                        thissn.Vmax = thissn.gp['maxmjd']['V']

                    # print ("GP vmax", thissn.Vmax)
                except IOError:
                    continue

        if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
            print('Vmax not found: ', thissn.snnameshort)
            continue
        # print ("Vmax", thissn.snnameshort, thissn.Vmax)
        # load data
        print(" starting loading ")
        lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
        # set up photometry
        # thissn.printsn()
        thissn.setphot()
        thissn.getphot()

        if np.array([n for n in iter(thissn.filters.values())]).sum() == 0:
            continue

        # thissn.plotsn(photometry=True)
        thissn.setphase()
        print(" finished ")
        # thissn.printsn()

        if workBands == su.bands:
            # add SN photometry to dataframe for latex table
            add2DF(thissn, tmp1, tmp2, bands1, bands2)

        if workBands == su.bands:
            # add SN photometry to dataframe for latex table
            add2DF(thissn, tmp1, tmp2, bands1, bands2)

        # work by band
        for b in workBands:
            # print (b)
            if b in ['w1', 'w2', 'm2']:
                # dont trust 06aj
                if thissn.snnameshort == '06aj': continue

            # look for the right phase offset
            # we want the peaks to align

            if not thissn.gp['max'][b] is None:
                offset = thissn.gp['max'][b][0]
                moffset = thissn.gp['max'][b][1]
                pl.plot(thissn.photometry['phase'], thissn.photometry['mag'])
                pl.plot([offset[0], offset[1]], [pl.ylim()[0], pl.ylim()[1]])
                pl.ylim(pl.yim()[1], pl.ylim()[0])
                # pl.show()
                # raw_input()
            else:
                # print ("no max")
                # raw_input()
                offset = None
                # continue
                # if offset is None:
                moffset = snstuff.coffset[b]
                # print b, thissn.snnameshort
            # add photometry to data container
            # print ("Vmax, offset", thissn.Vmax, offset)
            indx = (thissn.photometry[b]['phase'] > MINEP) * \
                   (thissn.photometry[b]['phase'] < MAXEP)

            allSNe[b]['mag'].append(thissn.photometry[b]['mag'][indx])

            # tstphot = thissn.photometry[b]['dmag'][indx]
            # tstphot[tstphot>0.3] = 0.3
            allSNe[b]['dmag'].append(thissn.photometry[b]['dmag'][indx])
            # np.zeros(len(thissn.photometry[b]['dmag'])) + 0.2)
            #
            allSNe[b]['phase'].append(thissn.photometry[b]['phase'][indx])
            allSNe[b]['name'].append(thissn.snnameshort)
            allSNe[b]['type'].append(thissn.type)

        ## remove duplicate entries from  array
    for b in workBands:
        for i in range(len(allSNe[b]['mag'])):
            if len(allSNe[b]['phase'][i]) == 0:
                continue
            records_array = np.unique(np.array(list(zip(allSNe[b]['phase'][i],
                                                        allSNe[b]['mag'][i],
                                                        allSNe[b]['dmag'][i]
                                                        ))), axis=0)

            allSNe[b]['phase'][i] = records_array[:, 0]
            allSNe[b]['mag'][i] = records_array[:, 1]
            allSNe[b]['dmag'][i] = records_array[:, 2]

        # print('FindMe2', allSNe[b][allSNe[b]['type'] == np.unique(allSNe[b]['type'])[0]])

        # for b_ in np.unique(allSNe[b]['type']):

        #     pl.plot()

    # making latex tables?
    if workBands == su.bands:
        # create latex tables
        bands = []
        for b in bands1:
            bands.append(b)
            bands.append(b + "[min,max]")

        # TODO: commented by Somayeh
        '''tabletex = "../../papers/SESNexplpars/tables/AllPhotOptTable.tex"
        add2table(tmp1, bands, tabletex)

        bands = []
        for b in bands2:
            bands.append(b)
            bands.append(b + "[min,max]")
        tabletex = os.getenv("DB") + "/papers/SESNexplpars/tables/AllPhotUVNIRTable.tex"
        add2table(tmp2, bands, tabletex) '''

    pkl.dump(allSNe, open('input/allSNe.pkl', 'wb'))

    return allSNe


def wSmoothAverage(data, sigma, err=True):
    '''weighted average weighted by both gaussian kernel and errors
    '''
    ut = data.copy()
    ut['epochs'] = ut['phs'][~np.isnan(ut['mu'])]
    ut['med'] = ut['med'][~np.isnan(ut['mu'])]
    ut['std'] = ut['std'][~np.isnan(ut['mu'])]
    ut['mu'] = ut['mu'][~np.isnan(ut['mu'])]

    if err:
        yerr = 1.0 / ut['std'] ** 2
    else:
        yerr = np.ones(len(ut['std']))

    ysmooth = ut['mu'][~np.isnan(ut['mu'])]
    '''
    ysmooth = np.array([
        np.average(ut['mu'],
                   weights=[np.exp(-(ph)**2 * 0.5 / sigma**2) * yerr
                            for ph in (- ut['epochs'][ti] + ut['epochs'])])
        for ti in range(len(ut['epochs']))])
    '''
    # phases where to calculate it
    window = 1.0 / 24
    ut['phs'] = np.arange(-20 - window, 100 + window,
                          window) + window * 0.5  # every hour in days
    # print(ysmooth.shape, )
    return ut, InterpolatedUnivariateSpline(ut['epochs'], ysmooth)


def wAverageByPhase(data, sigma, err=True, phsmax=100, window=5):
    dm = {}

    phs = np.arange(-20 - window, phsmax + window,
                    1.0 / 24)  # every hour in days

    N = len(phs)
    wmu = np.zeros(N) * np.nan
    med = np.zeros(N) * np.nan
    std = np.zeros(N) * np.nan
    wstd = np.zeros(N) * np.nan
    wgstd = np.zeros(N) * np.nan
    wgmu = np.zeros(N) * np.nan
    pc25 = np.zeros(N) * np.nan
    pc75 = np.zeros(N) * np.nan
    err_ratio = np.zeros(N) * np.nan
    weights_mean = np.zeros(N) * np.nan
    weights_med = np.zeros(N) * np.nan

    # sad c-style loop
    for i, hour in enumerate(phs):
        # i need at least 1 datapoint within 3 hours of the target hour (to take median)
        indx = (data['x'] >= hour) * (data['x'] < hour + window)
        # print (i, hour + window/2., indx.sum())

        # remove if less than 3 datapoints within 4 hours

        if indx.sum() < 3:
            continue

        # weighted average weighted by errorbars within hour and hour+window

        weights = 1.0 / ((data['yerr'][indx]) ** 2)

        wmu[i] = np.average(data['y'][indx], axis=0,
                            weights=weights)
        std[i] = np.std(data['y'][indx])

        wstd[i] = np.average((data['y'][indx] - wmu[i]) ** 2, axis=0,
                             weights=weights)
        # median
        # med[i] = np.median(data['y'][indx])

        pc25[i], med[i], pc75[i] = np.percentile(data['y'][indx], [25, 50, 75])
        indx_high = (data['y'][indx] < med[i])
        indx_low = (data['y'][indx] > med[i])

        err_ratio[i] = np.mean((data['yerr'][indx][indx_low]) ** 2) / np.mean((data['yerr'][indx][indx_high]) ** 2)
        weights_mean[i] = np.mean(1. / (data['yerr'][indx]) ** 2)
        weights_med[i] = np.median(1. / (data['yerr'][indx]) ** 2)

    interpmed = np.poly1d(np.polyfit(phs[~np.isnan(med)],
                                     wmu[~np.isnan(med)],
                                     3))
    # below checks the polynnomial spline that is removed before smoothing to avoid regression to mean

    # pl.figure()
    # pl.plot(phs[~np.isnan(med)],wmu[~np.isnan(med)],'.')
    # pl.plot(data['x'], interpmed(data['x']))
    # plt.gca().invert_yaxis()
    # pl.show()

    for i, hour in enumerate(phs):
        # exponential decay importance with time
        gtemp = np.exp(-(data['x'] - hour - window * 0.5) ** 2 * 0.5 / sigma ** 2)
        # np.exp(((data['x'] - hour - window * 0.5) /
        #                sigma)**2 / 2)
        weights = 1.0 / ((data['yerr']) ** 2) * gtemp
        wgmu[i] = np.average((data['y'] - interpmed(data['x'])), weights=weights)
        wgstd[i] = np.average((data['y'] - interpmed(data['x'])) ** 2, axis=0, weights=weights)

    # interpmed(data['x']) -
    # add back polynomial to fit general trend
    wgmu = wgmu + interpmed(phs)

    phs = phs + window * 0.5  # shift all phases by 1.5 hours

    phs0 = np.abs(phs) == min(np.abs(phs))
    phs15 = np.abs(phs - 15) == min(np.abs(phs - 15))
    phsm10 = np.abs(phs + 10) == min(np.abs(phs + 10))

    dm['15'], dm['15min'], dm['15max'] = (wmu[phs0] - wmu[phs15])[0], \
                                         (wmu[phs0] - wgstd[phs0] - (wmu[phs15] + wgstd[phs15]))[0], \
                                         (wmu[phs0] + wgstd[phs0] - (wmu[phs15] - wgstd[phs15]))[0]

    dm['-10'], dm['-10min'], dm['-10max'] = (wmu[phsm10] - wmu[phs0])[0], \
                                            (wmu[phsm10] - wgstd[phsm10] - (wmu[phs0] + wgstd[phs0]))[0], \
                                            (wmu[phsm10] + wgstd[phsm10] - (wmu[phs0] - wgstd[phs0]))[0]

    # pl.plot(phs, mu, 'b')
    # pl.plot(phs, med, 'g')
    # pl.plot(phs, wmu, 'r')
    # pl.errorbar(data['x'], data['y'], yerr=data['yerr'],
    #            color='k', fmt='.')
    return phs, wgmu, wgstd, wmu, med, std, wstd, dm, pc25, pc75, err_ratio, weights_med, weights_mean


def doall(b=su.bands, weights_plot=False, weights_heatmap=False):
    # read in csv file with metadata
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") +
                           "/SESNessentials.csv", encoding="ISO-8859-1")['SNname'].values  # [:5]

    if os.path.isfile('input/sncolors.pkl'):
        print('reading sncolors')
        with open('input/sncolors.pkl', 'rb') as f:
            sncolors = pkl.load(f, encoding="latin")
        # sncolors =  pkl.load(open('input/sncolors.pkl'))

        if not len(sncolors) == len(inputSNe):
            print("redoing SNcolors")
            # raw_input()

            sncolors = setcolors(inputSNe)
    else:
        sncolors = setcolors(inputSNe)
    dms = {}
    workBands = b
    templates = {}

    if PREP:
        allSNe = preplcvs(inputSNe, workBands)
    else:
        allSNe = pkl.load(open('input/allSNe.pkl'))

    for b in workBands:

        data = allSNe[b]

        sourcedata = dict(
            id=[],
            type=[],
            x=[],
            y=[],
            yerr=[],
            colors=[],
            mask=[])

        sncount = 0
        snN = len(data['phase'])
        badcount = 0

        for i, tmp in enumerate(data['phase']):

            flag = False
            if len(tmp) == 0:
                continue
            # print "tmp", tmp
            # sorted phases order
            indx = np.argsort(tmp)

            corephases = (tmp > -5) * (tmp < 5)

            if (b == 'U' and '03dh' in data['name'][i]) \
                    or ('03lw' in data['name'][i]):
                continue

            if corephases.sum() < 1:
                print(data['name'][i], b,
                      "has no data points between phs = -5 and phs = 5")
                continue

            # if data['name'][i] in ['13cq']:
            # flag = True  # continue
            # if flag:
            #     badcount += 1

            # set offset to minimum mag first
            # magoffset is the index of the minimum (brightest) dp
            magoffset = np.where(data['mag'][i] ==
                                 min(data['mag'][i]))[0]

            # if more than one peak have min value (nearly impossible w floats) choose first
            if not isinstance(magoffset, int):
                if len(magoffset) > 1:
                    tmpmo = magoffset[(data['mag'][i][magoffset] == min(data['mag'][i][magoffset]))][0]
                    magoffset = np.asarray([tmpmo])
            ymin = data['mag'][i][magoffset]

            if data['phase'][i][magoffset] < -10:
                # If there is a shock breakout, we want to make sure that is not chosen as the min mag
                magoffset = np.where(data['mag'][i][corephases] ==
                                     min(data['mag'][i][corephases]))[0]
                ymin = data['mag'][i][corephases][magoffset]
            elif b == 'g' and '16hgs' in data['name'][i]:
                # Shock breakout for 16hgs in g starts at phase -9.3 so we make an exception for it.
                magoffset = np.where(data['mag'][i][corephases] ==
                                     min(data['mag'][i][corephases]))[0]
                ymin = data['mag'][i][corephases][magoffset]

            # if the time of the maximum is more than 3 days off from t_max expected for this band
            # be suspicious and reset it if you can !

            # if np.abs(tmp[magoffset] - snstuff.coffset[b]) > 3:
            #     # print(data['name'][i], b)
            #     magoffset = np.where(np.abs(tmp) == np.min(np.abs(tmp)))[0]
            #     # flag = True
            #

            sncount += 1
            # set up key for hover tool: same name and type for all points
            sourcedata['id'] = sourcedata['id'] + [data['name'][i]] * len(indx)
            sourcedata['type'] = sourcedata['type'] + [data['type'][i]] * len(indx)

            sntp = data['type'][i]
            if not sntp in ['Ib', 'IIb', 'Ic', 'Ic-bl', 'Ibn']:
                sntp = 'other'

            sourcedata['yerr'] = sourcedata['yerr'] + list(data['dmag'][i][indx])
            sourcedata['colors'] = sourcedata['colors'] + \
                                   [rgb_to_hex(255. * sncolors[i])] * len(indx)

            maskhere = [False] * len(indx)

            # removing epochs <0 for dh03 due to GRB contamination
            if '03dh' in data['name'][i]:
                maskhere = np.array(maskhere)
                maskhere[tmp[indx] < 0] = True
                maskhere = maskhere.tolist()
            if '06jc' in data['name'][i] and b in ['H', 'J', 'K']:
                maskhere = np.array(maskhere)
                maskhere[tmp[indx] > 30] = True
                maskhere = maskhere.tolist()

            sourcedata['mask'] = sourcedata['mask'] + maskhere
            sourcedata['x'] = list(sourcedata['x']) + list(tmp[indx])
            sourcedata['y'] = sourcedata['y'] + list(-(data['mag'][i][indx]
                                                       - ymin))

        sourcedata['x'] = np.asarray(sourcedata['x'])
        sourcedata['y'] = np.asarray(sourcedata['y'])
        sourcedata['yerr'] = np.asarray(sourcedata['yerr'])

        indx = ~(np.isnan(sourcedata['x']) * np.isnan(sourcedata['yerr']) *
                 np.isnan(sourcedata['y']))

        sourcedata['x'] = np.asarray(sourcedata['x'])[indx]
        sourcedata['y'] = np.asarray(sourcedata['y'])[indx]
        sourcedata['yerr'] = np.asarray(sourcedata['yerr'])[indx]

        sourcedata['id'] = np.asarray(sourcedata['id'])[indx]
        sourcedata['type'] = np.asarray(sourcedata['type'])[indx]

        sourcedata['colors'] = np.asarray(sourcedata['colors'])[indx]
        sourcedata['mask'] = np.asarray(sourcedata['mask'])[indx]

        globals()['p0%s' % (b)], globals()['p1%s' % (b)], globals()['p2%s' % (b)], globals()['p3%s' % (b)], globals()[
            'p4%s' % (b)] = bokehplot(sourcedata, b)

    htmlout = "ubertemplates/bokeh/UberTemplate_PerType_PerBand2.html"
    output_file(htmlout)

    bkpl = []

    for i in range(5):
        tmp = []
        for b in ['U', 'u', 'B', 'V', 'R', 'g', 'r', 'I', 'i', 'w2', 'm2', 'w1', 'H', 'J', 'K']:
            tmp.append(globals()['p%s%s' % (i, b)])
        #     tmp = np.asarray(tmp).reshape(1,15)[0]
        bkpl.append(tmp)

    layout = gridplot(bkpl, width=500, height=500, toolbar_location='right')

    bokehsave(layout)

    # pd.DataFrame.from_dict(dms, orient='index').\
    #     to_csv("outputs/dmsUberTemplates.csv")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        doall(b=[sys.argv[1]])
    else:
        doall()
