import numpy as np
import glob
import os
import inspect
import sys
import pickle as pkl

from makePhottable import *
from bokeh.plotting import Figure as bokehfigure
from bokeh.plotting import save as bokehsave
from bokeh.plotting import vplot, figure, output_file
from bokeh.models import  BoxZoomTool, HoverTool, ResetTool, TapTool
from bokeh.models import ColumnDataSource, CustomJS,  Range1d

#, HBox, VBoxForm, BoxSelectTool, TapTool
#from bokeh.models.widgets import Select
#Slider, Select, TextInput
from bokeh.io import gridplot
from bokeh.plotting import output_file
from numpy import convolve
import matplotlib.gridspec as gridspec
 

try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

RIri = False

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from colors import hexcolors, allcolors
from snclasses import *
from templutils import *
from sklearn import gaussian_process
import optparse
import readinfofile as ri

import pandas as pd
import pickle as pkl
import snclasses as snstuff
import pylabsetup
su = setupvars()

def movingaverage (values, window, ws):
    weights = (np.repeat(1.0, window))
    sma = np.convolve(values*ws, weights, 'valid')
    return sma / np.convolve(weights, ws, 'valid')

def holt_winters_second_order_ewma(x, span, beta):
    '''lifted frm connor johnson 
    http://connor-johnson.com/2014/02/01/
    smoothing-with-exponentially-weighted-moving-averages/
    '''
    N = x.size
    alpha = 2.0 / ( 1 + span )
    s = np.zeros(( N, ))
    b = np.zeros(( N, ))
    s[0] = x[0]
    for i in range( 1, N ):
        s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
        b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
    return s

def wAverageByPhase (data, sigma):
    mu = np.zeros(2600) * np.nan
    wmu = np.zeros(2600) * np.nan    
    med = np.zeros(2600) * np.nan
    std = np.zeros(2600) * np.nan
    phs =  np.linspace(-20,100,2600)
    
    for i, hour in enumerate(phs):
        indx = (data['x']>=hour) * (data['x']<hour+3.0)
        if indx.sum() < 1:
            continue
        mu[i] = np.average(data['y'][indx],
                           axis=0,
                           weights = 1.0/data['yerr'][indx]**2)
        med[i] = np.median(data['y'][indx])
        std[i] = np.std(data['y'][indx])
        
        gtemp = np.exp(((data['x'][indx] - hour - 1.5)/sigma)**2 / 2)
        wmu[i] = np.average(data['y'][indx] ,
                            weights =   1.0/((data['yerr'][indx])**2) / gtemp)
        #np.average(data['y'][indx]*gtemp,
        #                   axis=0,
        #        weights = 1.0/((data['yerr'][indx])**2)/gtemp)

    phs = phs+1.5
    
    #pl.plot(phs, mu, 'b')
    #pl.plot(phs, med, 'g')
    #pl.plot(phs, wmu, 'r')
    #pl.errorbar(data['x'], data['y'], yerr=data['yerr'],
    #            color='k', fmt='.')
    return phs, wmu, mu, med, std


def gpme2(data, kernel, nugget):
    '''gpme2:  
       runs gaussian processed w skitlearn
    '''
    nonan = ~np.isnan(data['y']) * (np.isfinite(data['y'])) * (~np.isnan(data['y']) * (np.isfinite(data['y'])))
    print (data['x'])
    # add a little random offset so that 2 epochs are NEVER the same
    x = data['x'][nonan] + 0.0001 * np.random.randn(len(data['x'][nonan]))
    XX = (x - x.min() + 0.1)
    X = np.atleast_2d(np.log(XX)).T
    gp = gaussian_process.GaussianProcess(theta0=kernel[0],
                                          thetaL=kernel[1],
                                          thetaU=kernel[2],
                                          nugget=nugget)

    #pl.figure()
    #pl.errorbar(data['x'][nonan], data['y'][nonan],
    #        yerr=data['yerr'][nonan], fmt='.')
    try:
        gp.fit(X, data['y'][nonan])
    except:
        print ("failed gp")
        raw_input()
        return 1
    XXX = np.atleast_2d(np.log(data['phases'] - data['phases'][0] + 0.1)).T
    print (XXX, data['phases'], np.log(data['phases']))
    mu, std = gp.predict(XXX, eval_MSE=True)
    print (data['phases'], mu.min())
    pl.plot(data['phases'], mu, 'k-', linewidth=2)
    pl.fill_between(data['phases'], mu - std, mu + std, alpha=0.3, color='k')
    #pl.show()

def gpme(data, kernel):
    '''gpme:  
       runs gaussian processed w George
    '''
        
    XX = np.log(data['x'] - data['x'].min() + 0.1)
    gp = george.GP(kernel)
    gp.compute(XX, data['yerr'])
    mu, cov = gp.predict(data['y'], xx)
    std = np.sqrt(np.diag(cov))
    pl.plot(data['phases'], mu, 'k-', linewidth=2)
    pl.fill_between(data['phases'], mu - std, mu + std, alpha=0.3, color='k')



def errorbar(fig, fig2, s, s2, mu, std, phases):
    '''errorbar:
       makes errorbars plot in Bokeh figure
    '''
    #x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):

    fig.circle('x', 'y', size=5, source=s,
             color='grey', fill_color='colors', alpha=0.5)

    s.data['yerrx'] = []
    s.data['yerry'] = []
    for px, py, err in zip(s.data['x'], s.data['y'], s.data['yerr']):
        s.data['yerrx'].append((px, px))
        s.data['yerry'].append((py - err, py + err))
    fig.multi_line('yerrx', 'yerry', source = s, color='gray', alpha=0.5)
    fig.patches(xs='px', ys='py', source=s2,
                 color="grey", alpha=0.2)

    fig.set(y_range=Range1d(-10., 6.), x_range=Range1d(-50., 550.))
    fig2.circle('x', 'y', size=5, source=s2,
                color='grey', fill_color='colors',
                fill_alpha='alpha', alpha='alpha')
    s2.data['yerrx']=[]
    s2.data['yerry']=[]

    fig2.multi_line('yerrx', 'yerry', source = s2, line_alpha='alpha')
    fig.set(y_range=Range1d(-10., 6.), x_range=Range1d(-50., 550.))
    std[np.isnan(std)]=0
    mu[np.isnan(mu)]=0
    s2.data['px'] = [np.concatenate([phases, phases[::-1]])]
    s2.data['py'] = [np.concatenate([-mu+std, -mu[::-1]-std[::-1]])]
    s2.data['scolors'] = s.data['colors']
    s.data['px'] = [np.concatenate([phases, phases[::-1]])]
    s.data['py'] = [np.concatenate([-mu+std, -mu[::-1]-std[::-1]])]
    fig2.patches(xs='px', ys='py', source=s2,
                 color="grey", alpha=0.2)
    s.callback = CustomJS(args=dict(s2=s2), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        console.log(inds);
        var d = cb_obj.get('data');
        var d2 = s2.get('data');
        console.log(d2, inds)
        console.log("here", d2['px']);
        var color = d2['scolors'][inds];
        console.log(color);
        function getAllIndexes(arr, val) {
            var indexes = [], i;
            for(i = 0; i < arr.length; i++)
              if (arr[i] === val)
                indexes.push(i);
            return indexes;
        }
        var indices = getAllIndexes(d.id, d.id[inds]);
        console.log("here", d2['px']);
        for (i = 0; i < d['colors'].length; i++) {
            d2['x'].push(d['x'][i])
            d2['y'].push(d['y'][i])
            d2['yerr'].push(d['yerr'][i])
            d2['id'].push(d['id'][i])
            if (indices.indexOf(i) >= 0) {
               d2['colors'].push(color) 
               d2['alpha'].push(1) }
            else {
               d2['colors'].push("grey")
               d2['alpha'].push(0.000000)}
             
        }
        console.log(d2)
        
s2.trigger('change');
    """)
    #fig2.patches(xs='px', ys='py', source=s2,
    #             color="grey", alpha=0.2)

  
def getgps():
    '''getgps:
       reads in  gaussian processes stored in pkl file
    '''
    for b in su.bands:
        data = pkl.load(open("alldata_%s.pkl" % b, "rb"))
        plotme(data, b)
        gpme2(data, [0.3, None, None], data['yerr'] ** 2)
   

def doall():
    
    allGPs = {}
    workBands = su.bands
    
    bands1=[]
    bands2=[]
    for b in workBands:
        allGPs[b] = {'mag': [], 'dmag': [], 'phase': [], 'name': [], 'type': []}

    # read in csv file with metadata
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values


    print (inputSNe)
        
    # iterate over all SNe in metadata file
    for f in inputSNe:
        if f.startswith("#"): continue
        print (f)
    
        # read and set up SN and look for photometry files
        print (" looking for files ")
        thissn = snstuff.mysn(f, addlit=True)
        if len(thissn.optfiles) + len(thissn.fnir) == 0:
            continue

        # read metadata for SN
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)

        # setting date of maximum if not in metadata
        if np.isnan(thissn.Vmax) or thissn.Vmax == 0:
            # only trust selected GP results (only one really)
            if '06gi' in thissn.snnameshort:
                try:
                    print ("getting max from GP maybe?")
                    thissn.gp = pkl.load(open('gplcvs/' + f + \
                                             "_gp_ebmv0.00.pkl", "rb"))
                    if thissn.gp['maxmjd']['V'] < 2400000 and \
                       thissn.gp['maxmjd']['V'] > 50000:
                        thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                    else:
                        thissn.Vmax = thissn.gp['maxmjd']['V']

                    print ("GP vmax", thissn.Vmax)
                    #if not raw_input("should we use this?").lower().startswith('y'):
                    #    continue
                except IOError:
                    continue
        print (thissn.Vmax)

        if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
            continue

        # load data
        print (" starting loading ")    
        lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
        thissn.printsn()
        # set up photometry
        thissn.setphot()
        thissn.getphot()
        if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
            continue

        #thissn.plotsn(photometry=True)
        thissn.setphase()
        print (" finished ")
        thissn.printsn()
        

        if workBands == su.bands:
            # add SN photometry to dataframe for latex table
            add2DF(thissn, tmp1, tmp2, bands1, bands2)

        for b in workBands:
            print (b)
            if b in ['w1','w2','m2']:
                if thissn.snnameshort  == '06aj': continue
                
            # look for the right phase offset
            # we want the peaks to align
            
            if not thissn.gp['max'][b] is None:
                offset = thissn.gp['max'][b][0]
                moffset = thissn.gp['max'][b][1]
                pl.plot(thissn.photometry['phase'], thissn.photometry['mag'])
                pl.plot([offset[0], offset[1]], [pl.ylim()[0], pl.ylim()[1]])
                pl.ylim(pl.yim()[1], pl/ylim()[0])
                #pl.show()
                #raw_input()
            else:
                #print ("no max")
                #raw_input()
                offset = None
                #continue
                #if offset is None:
                moffset = snstuff.coffset[b]
                #print b, thissn.snnameshort

            # add photometry to data container
            print ("Vmax, offset", thissn.Vmax, offset)
            allGPs[b]['mag'].append(thissn.photometry[b]['mag'])
            allGPs[b]['dmag'].append(thissn.photometry[b]['dmag'])
            allGPs[b]['phase'].append(thissn.photometry[b]['phase'])  
            allGPs[b]['name'].append(thissn.snnameshort)
            allGPs[b]['type'].append(thissn.sntype)


    # making latex tables?
    if workBands == su.bands:
        # create latex tables
        bands = []
        for b in bands1:
            bands.append(b)
            bands.append(b + "[min,max]")

        tabletex = "../../papers/SESNexplpars/tables/AllPhotOptTable.tex"
        add2table(tmp1, bands, tabletex)

        bands = []
        for b in bands2:
            bands.append(b)
            bands.append(b + "[min,max]")
        tabletex = "../../papers/SESNexplpars/tables/AllPhotUVNIRTable.tex"
        add2table(tmp2, bands, tabletex)

    #gaussian processes container
    gp = {}
    
    # plot the photometric datapoints
    for b in workBands:
        print ("plotting band: ", b)
        source = ColumnDataSource(data={})

        source.data, ax = plotme(allGPs[b], b)

        # preparing GPs
        x = np.array(source.data['x'])
        if len(x)<2:
            continue

        indx1 = (x<100) & (~np.array(source.data['mask']))
        x = x[indx1] 
        #raw_input()
        if  len(x) == 0:
            continue
        indx = np.argsort(x)
        x = x[indx]
        
        #x = np.log(x[indx] - x[indx[0]] + 0.1)

        phases = np.arange(x[0], 100, 1.0/24.0)
        y = -np.array(source.data['y'])[indx1][indx]
        # np.concatenate([tmp for tmp in ])
        
        yerr = np.array(source.data['yerr'])[indx1][indx]
        # np.concatenate([tmp for tmp in source.data['yerr']])[indx]

        # save all anonymized photometric datapoints
        dataphases = {'x': x, 'y': y, 'yerr': yerr, 'phases': phases,
                  'allGPs': allGPs[b]}

        #continue        
        pkl.dump(dataphases, open('alldata_%s.pkl' % b, 'wb'))
        #continue

        print ("calculating stats and average ", b)
        yerr = dataphases['yerr']
        yerr[np.isnan(yerr) * ~(np.isfinite(yerr))] = 0.1
        yerr[yerr==0] = 0.1
        gsig = 2
        if b in ['K','H','J','i','u','U']:
            gsig = 4
        phs, wmu,  mu, med, std = wAverageByPhase(dataphases, gsig)

        wmu[std==0]=np.nan
        #mymean = np.empty_like(mu)*np.nan
        #tmp = movingaverage (dataphases['y'], 24 * 3 + 1,
        #                        1.0 / np.sqrt(dataphases['yerr']))
        #mymean[len(dataphases['y']) - len(tmp)/2 : - (len(dataphases['y']) - len(tmp)/2)] = tmp
        smoothedmean = np.empty_like(mu) * np.nan
        #smoothedmean[25:-25] = smooth(mu, window_len=51)[25:-25]
        #print ((dataphases['y']), (dataphases['x']), (dataphases['yerr']))
        smoothedmean = holt_winters_second_order_ewma
        #smoothIrregSampling(-y, x, yerr, window_len=2)
        #print len(smoothedmean)
        #print len(smoothedmean), len(mu), len(mymean)

        #ax[0].plot(phs, mu, 'k-', lw=1, alpha=0.7)
        #ax[0].plot(phs, smoothedmean, 'k-', lw=2)
        #ax[0].fill_between(phs, smoothedmean-std, smoothedmean+std,
        #                color = 'k', alpha=0.5)
        ax[0].fill_between(phs, wmu-std, wmu+std,
                        color = 'k', alpha=0.5)        
        ax[0].plot(phs, med, 'r-', alpha=0.7, lw=2)
        ax[0].plot(phs, wmu, 'k-', lw=2)
               

        ax1 = pl.subplot(ax[2][-1,0])
        #ax1.plot(phs, mu, 'g-', lw=1, alpha=0.7, label="rolling mean")
        ax1.fill_between(phs, wmu-std, wmu+std,
                         color = 'k', alpha=0.5, label=r"$\sigma$")
        ax1.plot(phs, med, '-', color = 'IndianRed', alpha=0.7, lw=2, label="median")
        ax1.plot(phs, wmu, 'k-', lw=2, label="weigthed/smoothed")
        #ax1.plot(x, -smoothedmean, '-', color = 'SteelBlue', lw=2, label="smoothed")        
        #ax1.plot(phs, mymean, '-', color='yellow', alpha=0.7, lw=2,
        #         label="my mean")
        ax1.legend(framealpha=0.5, ncol=3, loc=3, prop={'size': 15})
        ax1.set_ylabel("relative magnitude", fontsize = 20)
        ax1.set_xlabel("phase (days since Vmax)", fontsize = 20)
        ax1.set_ylim(ax1.get_ylim()[1], ax1.get_ylim()[0])
        ax1.set_xlim(-25,105)
        
        #gpme2(dataphases, [0.3, 0.1, 5], yerr ** 2)
        #sys.exit()
        '''
        # making Bokeh plots
        htmlout = "UberTemplate_%s.html" % (b + 'p' if b in ['u', 'r', 'i'] else b)
        output_file(htmlout)
        print (htmlout)
 
        TOOLS = [BoxZoomTool(), TapTool(), ResetTool(), HoverTool(
         tooltips=[
              ("ID", "@id"),
              ("SN type", "@type"),
              ("phase", "@x"),
              ("Delta mag", "@y"),
              ("Error", "@yerr"),
         ])]

        s2 = ColumnDataSource(data=dict(x=[], y=[], yerr=[],
                                        colors=[], id=[], alpha=[]))
        
        p = bokehfigure(plot_width=800, plot_height=400,
                            tools=TOOLS, title=b)
        p2 = bokehfigure(plot_width=800, plot_height=400)

        errorbar(p, p2, source, s2, wmu, std, phs)

        layout = vplot(p,p2)

        bokehsave(layout)
        #p.set(x_range=Range1d(xlim[0], xlim[1]), y_range=Range1d(ylim[1],ylim[0]))

        
        # run gaussian processes on data with George optimizing parameters
        result = op.minimize(snstuff.getskgpreds, (4.0, 1.0), args=(x,
                                                                y,
                                                                yerr,
                                                                phases),
                         bounds=((3.0, None), (10, None)),
                         tol=1e-5)
        kernel = result.x[1] * 10 * kernelfct(result.x[0])

        gp[b] = george.GP(kernel)
        if 'gpy' not in gp.keys():
            thissn.gp['gpy'] = {}
        thissn.gp['gpy'][b] = y

        XX = np.log(x - x.min() + 0.1)

        try:
            gp[b].compute(XX, yerr)
        except ValueError:
            print("Error: cannot compute GP")
            continue

        phases = np.arange(x.min(), x.max(), 0.1)

        try:
            epochs = np.log(phases - phases.min() + 0.1)
        except ValueError:
            print("Error: cannot set phases")
            continue

        tmptime = np.abs(phases - x[1])

        mu, cov = gp[b].predict(y, epochs)
        indx = np.where(tmptime == tmptime.min())[0][0]
        if indx == 0:
            indx = indx + 1

        mu[:indx + 1] = np.poly1d(np.polyfit(x[:2],
                                             y[:2], 
                                             1))(phases[:indx + 1])
        std = np.sqrt(np.diag(cov))

        pl.plot(phases, mu, 'k-', lw=2)
        pl.fill_between(phases, mu-std, mu+std, color = 'k', alpha = 0.3)
        '''
        pl.savefig("UberTemplate_%s.pdf" % (b + 'p' if b in ['u', 'r', 'i']
                                            else b))
        #pl.show()
        


if __name__ == '__main__':
    doall()
