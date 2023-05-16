import numpy as np
import glob
import os
import inspect
import sys
import pickle as pkl
import pandas as pd
        
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

specialSNe = ['94I', '93J', '08D', '05bf', '04aw', '10bm', '10vgv']

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

from snclasses import *
from templutils import *
from makePhottable import *
from colors import hexcolors, allcolors

PREP = True
#PREP = False
su = setupvars()
font = {'family' : 'normal',
        'size'   : 20}

pl.rc('font', **font)

def plotme(data, b, verbose=False):
    '''plotme: 
       makes matplotlib plot for band b with all datapoints from all SNe
       prepares dataframes for Bokeh plot
    '''
    
    pl.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.05)
    ax1 = pl.subplot(gs[:-1,:])
    ax2 = pl.subplot(gs[-1,1])
    
    sourcedata = dict(
         id=[],
         type=[],
         x=[],
         y=[],
         yerr=[],
         colors=[],
         mask = [])
    sncount = 0
    for i, tmp in enumerate(data['phase']):
        
        if len(tmp) == 0:
            continue
        #print "tmp", tmp
        indx = np.argsort(tmp)

        # set offset to minimum mag first
        # magoffser is the index of the peak dp
        magoffset = np.where(data['mag'][i] == min(data['mag'][i]))[0]

        if len(magoffset) > 1:
            if verbose: print (np.abs(tmp[magoffset]).min())
            tmpmo =  magoffset[np.abs(tmp[magoffset]) == \
                                   np.abs(tmp[magoffset]).min()][0]
            magoffset = tmpmo
            
        # if the maximum is more than 3 days off be suspicious and reset it
        # if you can !
        
        if np.abs(tmp[magoffset]-snstuff.coffset[b]) > 3 and (np.abs(tmp-snstuff.coffset[b]) < 1).any():
            # we can add exceptions here
            if not (b == 'u' and '13dx' in data['name'][i])\
               and not (b == 'i' and '13dx' in data['name'][i])\
               and not (b == 'H' and '09iz' in data['name'][i]):
                magoffset = np.where(np.abs(tmp) == np.min(np.abs(tmp)))[0]
            #print magoffset, data['mag'][i][magoffset],  tmp[magoffset]
        if data['name'][i] == '03dh' :
            magoffset = np.where(np.abs(tmp) == np.min(np.abs(tmp)))[0]
            
        if not isinstance(magoffset, int) and len(magoffset) > 1:
            tmpmo =  magoffset[(data['mag'][i][magoffset] == \
                                   min(data['mag'][i][magoffset]))][0]
            magoffset = tmpmo
        
        sncount += 1
        sourcedata['id'] = sourcedata['id'] + [data['name'][i]] * len(indx)
        sourcedata['type'] = sourcedata['type'] + [data['type'][i]] * len(indx)
        if verbose: print (sourcedata['x'])
        if verbose: print (tmp)
        if verbose: print (list(tmp[indx]))
        if verbose: print (data['phase'][i][magoffset])
        sourcedata['x'] = sourcedata['x'] + list(tmp[indx] - data['phase'][i][magoffset]) 
        sourcedata['y'] = sourcedata['y'] + list(-(data['mag'][i][indx] - data['mag'][i][magoffset]))
        sourcedata['yerr'] = sourcedata['yerr'] + list(data['dmag'][i][indx])
        sourcedata['colors'] = sourcedata['colors'] + [hexcolors[::3][i]] * len(indx)
        maskhere = [False] * len(indx)
        
        # removing epochs <0 for dh03 due to GRB contamination
        if '03dh' in data['name'][i] :
            maskhere = np.array(maskhere)
            maskhere[tmp[indx]<0] = True
            maskhere = maskhere.tolist()
        if '06jc' in data['name'][i]: 
            maskhere = np.array(maskhere)
            maskhere[tmp[indx]>30] = True
            maskhere = maskhere.tolist()
            

        #print sourcedata['colors']
        sourcedata['mask'] = sourcedata['mask'] + maskhere
        ax1.errorbar(tmp[indx],
                    data['mag'][i][indx] - data['mag'][i][magoffset],
                    yerr=data['dmag'][i][indx],
                    fmt='-', color=allcolors[i],
                    label=data['name'][i], alpha=0.8)
        if verbose: print ("colors", data['name'][i], allcolors[i])
        ax2.errorbar(tmp[indx][~np.array(maskhere)],
                     data['mag'][i][indx][~np.array(maskhere)] - \
                     data['mag'][i][magoffset],
                     yerr=data['dmag'][i][indx][~np.array(maskhere)],
                     fmt='.', color=allcolors[i],
                     alpha=0.8)

        
    ax1.set_title(b + "(%d)" % (sncount), fontsize=20)
    ax1.set_ylabel("relative magnitude", fontsize = 20)
    #ax1.set_xlabel("phase (days since Vmax)", fontsize = 20)
    ax2.set_ylabel("relative magnitude", fontsize = 20)
    ax2.set_xlabel("phase (days since Vmax)", fontsize = 20)
    ax2.yaxis.tick_right()
    ax2.grid(True)

    ax2.yaxis.set_label_position("right")
    ax1.legend(framealpha=0.5, ncol=4, numpoints=1, prop={'size': 15})
    ax1.set_ylim(ax1.get_ylim()[1], ax1.get_ylim()[0])
    ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])
    ax2.set_xlim(-25,105)    

    return sourcedata, (ax1, ax2, gs)

def errorbar(fig, fig2, s, s2, mu, std, phases):
    '''errorbar:
       makes errorbars plot in Bokeh figure
    '''
    #x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):

    fig.circle('x', 'y', size=5, source=s,
             color='grey', fill_color='colors', alpha=0.5)

    s.data['yerrx'] = []
    s.data['yerry'] = []

    s.data['specialyerrx'] = []
    s.data['specialyerry'] = []
    s.data['specialid'] = []
    s.data['specialtype'] = []    
    s.data['specialx'] = []
    s.data['specialy'] = []
    s.data['specialcolors'] = []        

    miny, maxy = 0, 0
    for px, py, err, sn, tp, c in zip(s.data['x'], s.data['y'], s.data['yerr'],
                                   s.data['id'], s.data['type'], s.data['colors']):
        s.data['yerrx'].append((px, px))
        s.data['yerry'].append((py - err, py + err))
        
        miny =  (py - err) if (py - err) < miny else miny
        maxy =  (px + err) if (py - err) > maxy else maxy        
        if sn in specialSNe:
            s.data['specialx'].append(px)
            s.data['specialy'].append(py)
            s.data['specialyerrx'].append((px, px))
            s.data['specialyerry'].append((py - err, py + err))                        
            s.data['specialid'].append(sn)
            s.data['specialtype'].append(tp)
            s.data['specialcolors'].append(c)                        
                   
    fig.multi_line('yerrx', 'yerry', source = s, color='gray', alpha=0.5)
    fig.patches(xs='px', ys='py', source=s2,
                 color="grey", alpha=0.2)

    fig2.circle('x', 'y', size=5, source=s2,
                color='grey', fill_color='colors',
                fill_alpha='alpha', alpha='alpha')
    s2.data['yerrx']=[]
    s2.data['yerry']=[]
    fig2.multi_line('yerrx', 'yerry', source = s2, line_alpha='alpha')
    fig2.line(x=phases, y=-mu, color='black')
    std[np.isnan(std)] = 0
    mu[np.isnan(mu)] = 0
    s2.data['mu'] = (- mu -1).tolist()
    s2.data['phase'] = phases.tolist()
    #fig2.multi_line('yerrx', 'yerry', source = s2, line_alpha='
    
    s2.data['px'] = [np.concatenate([phases, phases[::-1]])]
    s2.data['py'] = [np.concatenate([-mu + std, -mu[::-1] - std[::-1]])]
    s2.data['scolors'] = s.data['colors']
    s.data['px'] = [np.concatenate([phases, phases[::-1]])]
    s.data['py'] = [np.concatenate([-mu + std, -mu[::-1] - std[::-1]])]
    #print (s.data)
    fig2.circle('specialx', 'specialy', size=5, source=s,
                color='grey', fill_color='specialcolors', alpha=1)
    #fig2.multi_line('specialyerrx', 'specialyerry', source = s, color='gray', alpha=0.5)
    #                color='gray', alpha=0.5)
    #fig.patches(xs='px', ys='py', source=s2,
    #             color="grey", alpha=0.2)

    #s2.add((-mu).tolist(), 'mu')
    #s2.add(phases.tolist(), 'phase')    
    #fig2.line(x='phase', y='mu', source=s2)
    fig2.patches(xs='px', ys='py', source = s2,
                 color="grey", alpha=0.2)

        
    miny = min(s.data['py'][0]) if miny > min(s.data['py'][0]) else miny
    maxy = max(s.data['py'][0]) if maxy < max(s.data['py'][0]) else maxy
    s.callback = CustomJS(args=dict(s2=s2), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        console.log(inds);
        var d = cb_obj.get('data');
        var d2 = s2.get('data');
        console.log(d2, inds)
        d2['x'] = []
        d2['y'] = []
        d2['id'] = []
        d2['yerr'] = []
        d2['colors'] = []
        d2['alpha'] = []
        d2['px'] = d['px']
        d2['py'] = d['py']
        d2['yerrx'] = d['yerrx']
        d2['yerry'] = d['yerry']
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
    fig.set(y_range=Range1d(miny - 0.5, maxy + 0.5), x_range=Range1d(-50., 150.))
    fig2.set(y_range=Range1d(miny - 0.5, maxy + 0.5), x_range=Range1d(-50., 150.))
    #fig2.patches(xs='px', ys='py', source=s2,
    #             color="grey", alpha=0.2)


def bokehplot(source, wmu, std, phs, b):
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
        TOOLS2 = [HoverTool(            
         tooltips=[
              ("ID", "@specialid"),
              ("SN type", "@specialtype"),
         ])]

        s2 = ColumnDataSource(data=dict(x=[], y=[], yerr=[],
                                        colors=[], id=[], alpha=[], mu=[], phase=[]))        
        
        p = bokehfigure(plot_width=800, plot_height=400,
                            tools=TOOLS, title=b)
        p2 = bokehfigure(plot_width=800, plot_height=400)#, tools=TOOLS2)

        errorbar(p, p2, source, s2, wmu, std, phs)

        layout = vplot(p,p2)

        bokehsave(layout)
 
def preplcvs(inputSNe, workBands):
    
    allGPs = {}
    for b in workBands:
        allGPs[b] = {'mag': [], 'dmag': [], 'phase': [], 'name': [], 'type': []}
    
    if workBands == su.bands:
        # prepare stuff for latex tables to be passed to makePhottable
        bands1 = ['U', 'u', 'B', 'V', 'R', 'r', 'I', 'i']
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
    for f in inputSNe:
        if f.startswith("#"): continue
        print ("\n\n####################################################\n\n\n", f)
    
        # read and set up SN and look for photometry files
        #print (" looking for files ")
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
                    #print ("getting max from GP maybe?")
                    thissn.gp = pkl.load(open('gplcvs/' + f + \
                                             "_gp_ebmv0.00.pkl", "rb"))
                    #set to gregorian
                    if thissn.gp['maxmjd']['V'] < 2400000 and \
                       thissn.gp['maxmjd']['V'] > 50000:
                        thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                    else:
                        thissn.Vmax = thissn.gp['maxmjd']['V']

                    #print ("GP vmax", thissn.Vmax)
                except IOError:
                    continue

        if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
            continue
        #print ("Vmax", thissn.snnameshort, thissn.Vmax)
        # load data
        print (" starting loading ")    
        lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
        # set up photometry
        #thissn.printsn()
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

        if workBands == su.bands:
            # add SN photometry to dataframe for latex table
            add2DF(thissn, tmp1, tmp2, bands1, bands2)

        # work by band
        for b in workBands:
            #print (b)
            if b in ['w1','w2','m2']:
                #dont trust 06aj
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
            #print ("Vmax, offset", thissn.Vmax, offset)
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


    pkl.dump(allGPs, open('allSNe.pkl', 'wb'))

    return allGPs

   

def wAverageByPhase (data, sigma):
    window = 100.0 / 24.0 # n hours in days
    phs =  np.arange(-20 - window, 100 + window,
                     1.0/24) # every hour in days

    N = len(phs)
    mu = np.zeros(N) * np.nan
    wmu = np.zeros(N) * np.nan    
    med = np.zeros(N) * np.nan
    std = np.zeros(N) * np.nan
    
    for i, hour in enumerate(phs):
        # i need at least 1 datapoint within 3 hours of the target hour (why?)
        indx = (data['x'] >= hour) * (data['x'] < hour + window)
        #print (i, hour + window/2., indx.sum())
        if indx.sum() < 3:
            continue
        #weighted average weighted by errorbars within hour and hour+window
        mu[i] = np.average(data['y'][indx],
                           axis = 0,
                           weights = 1.0/(data['yerr'][indx]**2))
        med[i] = np.median(data['y'][indx])
        std[i] = np.std(data['y'][indx])

        #exponential decay importance with time
        gtemp = np.exp(((data['x'][indx] - hour - window * 0.5) /
                        sigma)**2 / 2)
        wmu[i] = np.average(data['y'][indx] ,
                            weights =   1.0/((data['yerr'][indx])**2) *
                            gtemp)
        #np.average(data['y'][indx]*gtemp,
        #                   axis=0,
        #        weights = 1.0/((data['yerr'][indx])**2)/gtemp)

    phs = phs + window * 0.5 #shift all phases by 1.5 hours
    
    #pl.plot(phs, mu, 'b')
    #pl.plot(phs, med, 'g')
    #pl.plot(phs, wmu, 'r')
    #pl.errorbar(data['x'], data['y'], yerr=data['yerr'],
    #            color='k', fmt='.')
    return phs, wmu, mu, med, std

    
def doall():
    # read in csv file with metadata
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values
    workBands = su.bands
    if PREP:
        allGPs = preplcvs(inputSNe, workBands)
    else:
        allGPs = {}
        allGPs = pkl.load(open('allSNe.pkl'))


    #gaussian processes container
    gp = {}

    # plot the photometric datapoints
    for b in workBands:
        print ("")
        

        source = ColumnDataSource(data={})
        source.data, ax = plotme(allGPs[b], b)
        #pl.show()
        # preparing GPs
        
        x = np.array(source.data['x']) #timeline
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

        phases = np.arange(x[0], 100,  1.0/24.0)
        y = -np.array(source.data['y'])[indx1][indx]
        # np.concatenate([tmp for tmp in ])
        
        yerr = np.array(source.data['yerr'])[indx1][indx]
        # np.concatenate([tmp for tmp in source.data['yerr']])[indx]

        # save all anonymized photometric datapoints
        dataphases = {'x': x, 'y': y, 'yerr': yerr, 'phases': phases,
                  'allGPs': allGPs[b]}

        pkl.dump(dataphases, open('alldata_%s.pkl' % b, 'wb'))
        
        print ("calculating stats and average ", b)

        #fixing yerr
        yerr = dataphases['yerr']
        yerr[np.isnan(yerr) * ~(np.isfinite(yerr))] = 0.1
        yerr[yerr==0] = 0.1
        gsig = 100
        if b in ['K','H','J','i','u','U']:
            gsig = gsig * 2
        phs, wmu,  mu, med, std = wAverageByPhase(dataphases, gsig)

        wmu[std==0]=np.nan
        #mymean = np.empty_like(mu)*np.nan
        #tmp = movingaverage (dataphases['y'], 24 * 3 + 1,
        #                        1.0 / np.sqrt(dataphases['yerr']))
        #mymean[len(dataphases['y']) - len(tmp)/2 : - (len(dataphases['y']) - len(tmp)/2)] = tmp
        smoothedmean = np.empty_like(mu) * np.nan
        #smoothedmean[25:-25] = smooth(mu, window_len=51)[25:-25]
        #print ((dataphases['y']), (dataphases['x']), (dataphases['yerr']))
        smoothedmean = smoothIrregSampling(-y, x, yerr, window_len=2)
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

        pl.savefig("UberTemplate_%s.pdf" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b))
        #pl.show() 
        #bokehplot(source, wmu, std, phs, b)
        #'''
        #break
        ut = {}#pd.DataFrame()
        ut['phs'] = phs
        ut['mu'] = wmu
        ut['mu'][ut['mu'] ==0] = np.nan
        ut['std'] = std

        from scipy.interpolate import interp1d
        import templutils as tpl
        ut['phs'] = ut['phs'][~np.isnan(ut['mu'])]
        ut['mu'] = ut['mu'][~np.isnan(ut['mu'])]
        ut['std'] = ut['std'][~np.isnan(ut['mu'])]
        f2 = interp1d(ut['phs'], ut['mu'], kind='cubic')
        
        ysmooth = np.array([
            np.average(ut['mu'],
                       weights=[np.exp(-(ph)**2 * (0.5 / 3)**2)
                                for ph in (ut['phs'][ti] - ut['phs'])])
            for ti in range(len(ut['phs']))])
        from scipy.interpolate import InterpolatedUnivariateSpline
        ut['spl'] = InterpolatedUnivariateSpline(ut['phs'], ysmooth)

        #ut.to_csv("UberTemplate_%s.csv" % \
        #           (b + 'p' if b in ['u', 'r', 'i']
        #                                    else b))
        print (ut)
        pkl.dump(ut, open("UberTemplate_%s.pkl" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b), 'wb'))
if __name__ == '__main__':
    doall()
