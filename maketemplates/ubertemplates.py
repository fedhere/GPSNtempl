import numpy as np
import glob
import os
import inspect
import sys
import pickle as pkl
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

import bokeh
from bokeh.io import curdoc

from bokeh.plotting import Figure as bokehfigure
from bokeh.plotting import figure as bokehfigure
from bokeh.plotting import save as bokehsave
from bokeh.plotting import show as bokehshow
from bokeh.plotting import figure, output_file
from bokeh.layouts import column
from bokeh.models import  BoxZoomTool, HoverTool, ResetTool, TapTool
from bokeh.models import ColumnDataSource, CustomJS,  Range1d

print ("bokeh version", bokeh.__version__)
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

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
from makePhottable import *
from colors import hexcolors, allcolors, colormaps, rgb_to_hex

MINEP, MAXEP = -100, 365.25 * 2
archetypicalSNe = ['94I', '93J', '08D', '05bf', '04aw', '10bm', '10vgv']

colorTypes = {'IIb':'FireBrick',
             'Ib':'SteelBlue',
             'Ic':'DarkGreen',
             'Ic-bl':'DarkOrange',
             'other':'purple'}

#prepping SN data from scratch
#(dont do that if you are testing or tuning plots)
PREP = True 
#PREP = False
BOKEHIT = True
#BOKEHIT = False
font = {'family' : 'normal',
        'size'   : 20}

#setting up snclass and other SESNCFAlib stuff
su = setupvars()

pl.rc('font', **font)

def setcolors(inputSNe):
    cm = pl.get_cmap('nipy_spectral')#viridis')
    Nsne = len(inputSNe)
    print Nsne
    sncolors = [''] * Nsne
    for i in range(Nsne):
        sncolors[i] = (cm(1.*i/Nsne))
    sncolors = np.asarray(sncolors)
    
    np.random.seed(666)
    np.random.shuffle(sncolors)
    pkl.dump(sncolors, open("input/sncolors.pkl", 'wb'))          
    return (sncolors)

def select2ObsPerDay(data):

    dataTimeByDay =  data['x'].astype(int)
    #print data, dataTimeByDay
    minx = int(data['x'].min())
    #print np.unique(dataTimeByDay, return_counts=True)
    for sid in np.unique(data['id']):
        thissn = np.where((data['id'] == sid))[0]        
        #print sid, np.where((data['id'] == sid))[0]
        bc = np.bincount(dataTimeByDay[thissn] - minx) > 2
        for i in np.arange(len(bc))[bc]:
            #print i + minx, bc[i]
            theseindx = np.where((data['id'] == sid) *
                            (dataTimeByDay == i + minx))[0]
            choices = np.random.choice(theseindx, len(theseindx) - 2)
            #print theseindx[np.argsort(data['yerr'][theseindx])],
            #print data['yerr'][theseindx][np.argsort(data['yerr'][theseindx])]            
                
            #print data['mask']#[thissn]
            #data['mask'][choices] = True
            #print theseindx[np.argsort(data['yerr'][theseindx])][2:] #= True
            data['mask'][theseindx[np.argsort(data['yerr'][theseindx])][2:]] = True
    return data
            
    sys.exit()

    
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): # we are forecasting
          value = result[-1]
        else:
          value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


def plotme(data, b, sncolors, axtype=None, verbose=False):
    #pl.ion()
    '''plotme: 
       makes matplotlib plot for band b with all datapoints from all SNe
       prepares dataframes for Bokeh plot
    '''
    pl.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 2)
    gs.update(wspace=0.05)
    ax0 = pl.subplot(gs[:-1, :])
    ax2 = pl.subplot(gs[-1, 1])
    
    sourcedata = dict(
         id=[],
         type=[],
         x=[],
         y=[],
         yerr=[],
         colors=[],
         mask = [])

    sncount = 0
    snN = len(data['phase'])
    badcount  = 0
    for i, tmp in enumerate(data['phase']):

        
        flag = False
        if len(tmp) == 0:
            continue
        #print "tmp", tmp
        #sorted phases order
        indx = np.argsort(tmp)

        corephases = (tmp > -5) * (tmp < 5)

        if corephases.sum()<1:
            print  (data['name'][i], b,
                    "has no datapoints between near 0. Moving on")
            flag = True
        #print(data['name'][i])
        if data['name'][i] in ['03lw', '04dk', '04gt', '06fo',
                               '07D', '13cq']:
            flag = True #continue
        if flag:
            badcount += 1
        # set offset to minimum mag first
        # magoffset is the index of the minimum (brightest) dp
        if ('13dx' == data['name'][i] and b == 'i') :
            magoffset = np.where(data['mag'][i][tmp>-2] ==
                                 min(data['mag'][i][tmp>-2]))[0]            
        else:
            magoffset = np.where(data['mag'][i] ==
                             min(data['mag'][i]))[0]
        
        if '16gkg' in data['name'][i]:
            magoffset = [0]
        #if more than one peak have min value (nearly impossible w floats) choose first
        if len(magoffset) > 1:
            if verbose: print (np.abs(tmp[magoffset]).min())
            tmpmo =  magoffset[np.abs(tmp[magoffset]) == \
                                   np.abs(tmp[magoffset]).min()][0]
            magoffset = tmpmo
            
        #if the maximum is more than 3 days off from expected for this band
        #be suspicious and reset it if you can !
        
        if np.abs(tmp[magoffset] - snstuff.coffset[b]) > 3\
           and (np.abs(tmp - snstuff.coffset[b]) < 1).any():
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
        #set up key for hover tool: same name and type for all points
        sourcedata['id'] = sourcedata['id'] + [data['name'][i]] * len(indx)
        sourcedata['type'] = sourcedata['type'] + [data['type'][i]] * len(indx)
        if verbose: print ('old epochs:', sourcedata['x'])
        if verbose: print ('phases:', tmp)
        if verbose: print ('peak:', list(tmp[indx]))
        if verbose: print (data['phase'][i][magoffset])
        sntp = data['type'][i]
        if not sntp in ['Ib', 'IIb', 'Ic','Ic-bl']:
            sntp = 'other'


        sourcedata['yerr'] = sourcedata['yerr'] + list(data['dmag'][i][indx])
        sourcedata['colors'] = sourcedata['colors'] +\
                               [rgb_to_hex(255. * sncolors[i])] * len(indx)
        sourcedata['typegroup'] = sourcedata['colors'] +\
                               [rgb_to_hex(255. * sncolors[i])] * len(indx)        
        #[hexcolors[::3][i]] * len(indx)                               
                               #[colormaps[sntp](i*1.0/snN)]  * len(indx)
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
        if not flag:
            sourcedata['x'] = sourcedata['x'] + list(tmp[indx]
                                                 - data['phase'][i][magoffset]) 
            
            sourcedata['y'] = sourcedata['y'] + list(-(data['mag'][i][indx]
                                                       - data['mag'][i][magoffset]))
            ax0.errorbar(tmp[indx],
                         data['mag'][i][indx] - data['mag'][i][magoffset],
                         yerr=data['dmag'][i][indx],
                         fmt='-', color=sncolors[i],
                         label=data['name'][i], alpha=0.5)
            ax2.errorbar(tmp[indx][~np.array(maskhere)],
                         data['mag'][i][indx][~np.array(maskhere)] - \
                         data['mag'][i][magoffset],
                         yerr=data['dmag'][i][indx][~np.array(maskhere)],
                         fmt='.', color=sncolors[i],
                         alpha=0.5)
            if axtype:
                axtype.errorbar(tmp[indx][~np.array(maskhere)],
                                data['mag'][i][indx][~np.array(maskhere)] - \
                                data['mag'][i][magoffset],
                                yerr=data['dmag'][i][indx][~np.array(maskhere)],
                                fmt='.', color=colorTypes[sntp],
                                alpha=0.5)            
        else:
            sourcedata['x'] = sourcedata['x'] + list(tmp[indx])
            
            sourcedata['y'] = sourcedata['y'] + list(-(data['mag'][i][indx]
                                                       - data['mag'][i].min()))            
            ax0.errorbar(tmp[indx],
                         data['mag'][i][indx] - data['mag'][i].min(),
                         yerr=data['dmag'][i][indx],
                         fmt='--', color=sncolors[i],
                         label=data['name'][i], alpha=0.5)
            ax2.errorbar(tmp[indx][~np.array(maskhere)],
                         data['mag'][i][indx][~np.array(maskhere)],
                         yerr=data['dmag'][i][indx][~np.array(maskhere)],
                         fmt='.', color=sncolors[i],
                         alpha=0.5)

        
    ax0.set_title(b + "(%d)" % (sncount - badcount), fontsize=20)
    ax0.set_ylabel("relative magnitude", fontsize = 20)
    #ax0.set_xlabel("phase (days since Vmax)", fontsize = 20)
    ax2.set_ylabel("relative magnitude", fontsize = 20)
    ax2.set_xlabel("phase (days since Vmax)", fontsize = 20)
    ax2.yaxis.tick_right()
    ax2.grid(True)
    ax0.grid(True)

    ax2.yaxis.set_label_position("right")
    ax0.legend(framealpha=0.5, ncol=4, numpoints=1, prop={'size': 13})
    ax0.set_ylim(ax0.get_ylim()[1], ax0.get_ylim()[0])
    #ax0.set_xlim(20,80)
    #ax0.set_ylim(4,-2)    
    
    ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])
    ax2.set_xlim(-27,105)

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

    sncolordic = {}
    for i,k in enumerate(sourcedata['id']):
        sncolordic[k] = sourcedata['colors'][i]
    pkl.dump(sncolordic, open('outputs/colorSNe.pkl', 'wb'))

  
  
    
    return sourcedata, (ax0, ax2, gs)

def update_xyaxis(f, rg,  xy='y'):
    if xy == 'y':
        fax = f.y_range
    else:
        fax = f.x_range        
        
        fax.start = rg[0] 
        fax.end   = rg[1]

def bkerrorbar(fig, fig2, s, s2, mu, std, phases):
    '''errorbar:
       makes errorbars plot in Bokeh figure
    '''
    #x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):

    s.data['yerrx'] = []
    s.data['yerry'] = []

    #s.data['specialyerrx'] = []
    #s.data['specialyerry'] = []
    #s.data['specialid'] = []
    #s.data['specialtype'] = []    
    #s.data['specialx'] = []
    #s.data['specialy'] = []
    #s.data['specialcolors'] = []
    #s.data['specialname'] = []        

    miny, maxy = 0, 0
    
    for px, py, err, sn, tp, c in zip(s.data['x'], s.data['y'], s.data['yerr'],
                                   s.data['id'], s.data['type'], s.data['colors']):
        s.data['yerrx'].append((px, px))
        s.data['yerry'].append((py - err, py + err))
        
        miny =  (py - err) if (py - err) < miny else miny
        maxy =  (px + err) if (py - err) > maxy else maxy
    s.data['yerrx'] = np.asarray(s.data['yerrx'])
    s.data['yerry'] = np.asarray(s.data['yerry'])

    s2.data['yerrx'] = np.asarray(s.data['yerrx'])
    s2.data['yerry'] = np.asarray(s.data['yerry'])

    s2.data['id'] = np.asarray(s.data['id'])
    s2.data['type'] = np.asarray(s.data['type'])    
    
    fig.multi_line('yerrx', 'yerry', source = s, color='gray', alpha=0.5)
    #fig.patches(xs='specialx', ys='specialy', source=s,
    #             color="grey", alpha=0.2)
    fig.circle('x', 'y', size=5, source=s,
               color='grey', fill_color='colors', alpha=0.5)    
    
    fig2.circle('x', 'y', size=5, source=s2,
                color='grey', fill_color='colors',
                fill_alpha='alphas', alpha='alphas')
    fig2.multi_line('yerrx', 'yerry', source = s2, line_alpha='alphas')
    #fig2.line(x=phases, y=-mu, color='black')
    std[np.isnan(std)] = 0
    mu[np.isnan(mu)] = 0
    
    s2.add((- mu).tolist(), 'mu')
    s2.add(phases.tolist(), 'phase')
    #s.add(np.concatenate([phases, phases[::-1]]).tolist(), 'px') 
    #s.add(np.concatenate([-mu + std, -mu[::-1] - std[::-1]]).tolist(), 'py')    
    #fig2.multi_line('yerrx', 'yerry', source = s2, line_alpha='
    #s2.data['px'] = [np.concatenate([phases, phases[::-1]])]
    #s2.data['py'] = [np.concatenate([-mu + std, -mu[::-1] - std[::-1]])]
    #s2.data['scolors'] = s.data['colors']

    px = [np.concatenate([phases, phases[::-1]])]
    py = [np.concatenate([-mu + std, -mu[::-1] - std[::-1]])]    

    '''
    for sn in archetypicalSNe:
        indx = np.array(s.data['id']) == sn
        if not indx.sum() :
            continue
        fig2.circle(np.array(s.data['x'])[indx], np.array(s.data['y'])[indx], size=5, 
                    color='grey', fill_color=np.array(s.data['colors'])[indx], alpha=1,
                    legend=sn)

        fig2.multi_line(xs=list(np.array(s.data['yerrx'])[indx]),
                        ys=list(np.array(s.data['yerry'])[indx]),
                        color='gray', alpha=0.5)
        #fig.patches(xs='px', ys='py', source=s,
        #            color="grey", alpha=0.2)
    '''

    fig2.line(x='phase', y='mu', source=s2)    
    fig2.patches(xs=px, ys=py, 
                 color="grey", alpha=0.3)
    
    update_xyaxis(fig, (miny - 0.5, maxy + 0.5))
    update_xyaxis(fig, (-30., 150.), xy='x')
    update_xyaxis(fig2, (miny - 0.5, maxy + 0.5))
    update_xyaxis(fig2, (-50., 150.), xy='x')


    miny = min(py[0]) if miny > min(py[0]) else miny
    maxy = max(py[0]) if maxy < max(py[0]) else maxy
    s.callback = CustomJS(args=dict(s2=s2), code="""
        var inds = cb_obj.get('selected')['1d'].indices;
        console.log("inds", inds);
        console.log(cb_obj.get('selected')['1d'])
        var d1 = cb_obj.get('data');
        console.log("here",d1['id'][inds])
        var d2 = s2.get('data');
        function getAllIndexes(arr, val) {
            var indexes = [], i;
            for(i = 0; i < arr.length; i++)
              if (arr[i] === val)
                indexes.push(i);
            return indexes;
        }
        d2['alphas']=Array(d2.id.length).fill(0.1);
        var indices = getAllIndexes(d1.id, d1.id[inds]);
        console.log("indices", indices[0], indices)
        console.log("here", d1['id'][indices[0]])

        console.log("now")
        console.log("now", d2.id.length, indices.indexOf(inds[0]))
        console.log("done")

        for (i = 0; i < d2.id.length; i++) {
            if (d2.id[i] == d1.id[inds]) {
               console.log("i", d2.id[i], d1.id[inds])  
               d2['alphas'][i]=1.0 }
            else {d2['alphas'][i]=0.01 }
        }
        console.log(d2['alphas'])
        s2.trigger('change');
    """)


    
    #fig.set(y_range=Range1d(miny - 0.5, maxy + 0.5), x_range=Range1d(-50., 150.))
    #fig2.set(y_range=Range1d(miny - 0.5, maxy + 0.5), x_range=Range1d(-50., 150.))
    #fig2.patches(xs='px', ys='py', source=s2,
    #             color="grey", alpha=0.2)


def bokehplot(source, wmu, std, phs, b):
           # making Bokeh plots
        htmlout = "outputs/UberTemplate_%s.html"%\
                  (b + 'p' if b in ['u', 'r', 'i'] else b)
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
        TOOLS2 = [BoxZoomTool(),   ResetTool(), HoverTool( 
         tooltips=[
              ("ID", "@id"),
              ("SN type", "@type"),
         ])]
        s2 =  ColumnDataSource(data={})

        s2.data['x'] = np.asarray(source.data['x'])

        s2.data['y'] = np.asarray(source.data['y'])
        s2.data['colors'] = np.asarray(source.data['colors'])
        s2.data['alpha'] = np.asarray([0.1]*len(source.data['colors']))
        
        s2.add([0.1 for c in source.data['colors']], 'alphas')
                
        
        #ColumnDataSource(data=dict(x=[], y=[], yerr=[],
             #                           colors=[], id=[], alpha=[], mu=[], phase=[]))        
        
        p = bokehfigure(plot_width=600, plot_height=300,
                        tools=TOOLS, title=b)
        p2 = bokehfigure(plot_width=600, plot_height=300,
                         tools=TOOLS2)

        
        bkerrorbar(p, p2, source, s2, wmu, std, phs)

        p.yaxis.axis_label = "relative magnitude"
        p2.xaxis.axis_label = "phase (days)"
        p2.yaxis.axis_label = "relative magnitude"        
        layout = column(p,p2)

        bokehsave(layout)

        print ("\n\n bokeh plot saved to " + "outputs/UberTemplate_%s.html"%\
                  (b + 'p' if b in ['u', 'r', 'i'] else b))
        #curdoc().add_root(layout)
        #bokehshow(layout)        
 
def preplcvs(inputSNe, workBands):

    keys = ['mag','dmag', 'phase',  'name', 'type']
    allSNe = {}
    for b in workBands:
        allSNe[b] = {}
        for k in keys:
            allSNe[b][k] = []
        #'mag': [], 'dmag': [], 'phase': [], 'name': [], 'type': []}
    
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
                pl.ylim(pl.yim()[1], pl.ylim()[0])
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
            indx = (thissn.photometry[b]['phase'] > MINEP) * \
                   ( thissn.photometry[b]['phase'] < MAXEP)

            allSNe[b]['mag'].append(thissn.photometry[b]['mag'][indx])
            allSNe[b]['dmag'].append(thissn.photometry[b]['dmag'][indx])
            #np.zeros(len(thissn.photometry[b]['dmag'])) + 0.2)
            #
            allSNe[b]['phase'].append(thissn.photometry[b]['phase'][indx])  
            allSNe[b]['name'].append(thissn.snnameshort)
            allSNe[b]['type'].append(thissn.sntype)

        ## remove duplicate entries from  array
    for b in workBands:
        for i in range(len(allSNe[b]['mag'])):
            if len(allSNe[b]['phase'][i]) == 0:
                continue
            records_array = np.unique(np.array(zip(allSNe[b]['phase'][i],
                                                   allSNe[b]['mag'][i],
                                                   allSNe[b]['dmag'][i]
                                                   )), axis=0)
            
            allSNe[b]['phase'][i] = records_array[:,0]            
            allSNe[b]['mag'][i] = records_array[:,1]
            allSNe[b]['dmag'][i] = records_array[:,2]
                


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
        tabletex = os.getenv("DB") + "/papers/SESNexplpars/tables/AllPhotUVNIRTable.tex"
        add2table(tmp2, bands, tabletex)            


    pkl.dump(allSNe, open('input/allSNe.pkl', 'wb'))

    return allSNe

   

def wSmoothAverage (data, sigma, err=True):
    '''weighted average weighted by both gaussian kernel and errors
    '''
    ut = data.copy()
    ut['epochs'] = ut['phs'][~np.isnan(ut['mu'])]
    ut['med'] = ut['med'][~np.isnan(ut['mu'])]    
    ut['std'] = ut['std'][~np.isnan(ut['mu'])]
    ut['mu'] = ut['mu'][~np.isnan(ut['mu'])]

    if err:
        yerr = 1.0 / ut['std']**2
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
                       #phases where to calculate it
    window = 1.0/24
    ut['phs'] =  np.arange(-20 - window, 100 + window,
                           window)  + window * 0.5 # every hour in days
    #print(ysmooth.shape, )
    return ut, InterpolatedUnivariateSpline(ut['epochs'], ysmooth)

    
def wAverageByPhase (data, sigma, err=True, phsmax=100, window=5):
    dm = {}
    
    phs =  np.arange(-20 - window, phsmax + window,
                     1.0/24) # every hour in days

    N = len(phs)
    wmu = np.zeros(N) * np.nan
    med = np.zeros(N) * np.nan
    std = np.zeros(N) * np.nan
    wstd = np.zeros(N) * np.nan
    wgstd = np.zeros(N) * np.nan        
    wgmu = np.zeros(N) * np.nan    
    pc25 = np.zeros(N) * np.nan
    pc75 = np.zeros(N) * np.nan    
    
    #sad c-style loop
    for i, hour in enumerate(phs):
        # i need at least 1 datapoint within 3 hours of the target hour (to take median)
        indx = (data['x'] >= hour) * (data['x'] < hour + window)
        #print (i, hour + window/2., indx.sum())

        #remove if less than 3 datapoints within 4 hours

        if indx.sum() < 3:
            continue


        #weighted average weighted by errorbars within hour and hour+window

        weights = 1.0 / ((data['yerr'][indx])**2)
        
        wmu[i] = np.average(data['y'][indx], axis = 0,
                            weights=weights)
        std[i] = np.std(data['y'][indx]) 

        wstd[i] = np.average((data['y'][indx]-wmu[i])**2,  axis = 0,
                             weights=weights)
        #median
        #med[i] = np.median(data['y'][indx])
        pc25[i], med[i], pc75[i] = np.percentile(data['y'][indx], [25, 50, 75])
        

    interpmed = np.poly1d(np.polyfit(phs[~np.isnan(med)],
                                     wmu[~np.isnan(med)],
                                     3))
    # below checks the polynnomial spline that is removed before smoothing to avoid regression to mean
    
    #pl.figure()
    #pl.plot(phs[~np.isnan(med)],wmu[~np.isnan(med)],'.')
    #pl.plot(data['x'], interpmed(data['x']))
    #pl.show()
    
    for i, hour in enumerate(phs):

        #exponential decay importance with time
        gtemp = np.exp(-(data['x'] - hour - window * 0.5)**2 * 0.5 / sigma**2)
                       #np.exp(((data['x'] - hour - window * 0.5) /
        #                sigma)**2 / 2)
        weights = 1.0/((data['yerr'])**2) * gtemp
        wgmu[i] = np.average(data['y'] - interpmed(data['x']),
                            weights=weights) 
        wgstd[i] = np.average((data['y'] - interpmed(data['x']))**2,  axis = 0,
                             weights=weights)


    #add back polynomial to fit general trend
    wgmu = wgmu + interpmed(phs)
    
    phs = phs + window * 0.5 #shift all phases by 1.5 hours

    phs0 = np.abs(phs) == min(np.abs(phs))
    phs15 = np.abs(phs - 15) == min(np.abs(phs - 15))
    phsm10 = np.abs(phs + 10) == min(np.abs(phs + 10))

        
    dm['15'],dm['15min'],dm['15max'] = (wmu[phs0] - wmu[phs15])[0],\
                (wmu[phs0] - wgstd[phs0] - (wmu[phs15] + wgstd[phs15]))[0],\
                (wmu[phs0] + wgstd[phs0] - (wmu[phs15] - wgstd[phs15]))[0]

    dm['-10'],dm['-10min'],dm['-10max'] = (wmu[phsm10] - wmu[phs0])[0],\
                 (wmu[phsm10] -  wgstd[phsm10] - (wmu[phs0] + wgstd[phs0]))[0],\
                 (wmu[phsm10] +  wgstd[phsm10] - (wmu[phs0] - wgstd[phs0]))[0]
    
    #pl.plot(phs, mu, 'b')
    #pl.plot(phs, med, 'g')
    #pl.plot(phs, wmu, 'r')
    #pl.errorbar(data['x'], data['y'], yerr=data['yerr'],
    #            color='k', fmt='.')
    return phs, wgmu, wgstd, wmu, med, std, wstd, dm, pc25, pc75

    
def doall(b = su.bands):
    # read in csv file with metadata
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") +
                           "/SESNessentials.csv")['SNname'].values#[:5]

    if os.path.isfile('input/sncolors.pkl'):
        print ('reading sncolors')
        sncolors =  pkl.load(open('input/sncolors.pkl'))
        
        if not len(sncolors) == len(inputSNe):
            print ("redoing SNcolors")
            #raw_input()

            sncolors = setcolors(inputSNe)
    else:
        sncolors = setcolors(inputSNe)            
    dms = {}
    workBands = b

    if PREP:
        allSNe = preplcvs(inputSNe, workBands)
    else:
        allSNe = {}
        allSNe = pkl.load(open('input/allSNe.pkl'))

    
    # plot the photometric datapoints
    for b in workBands:

        print ("")
        fig = pl.figure(figsize=(15, 10))
        axtype = fig.add_subplot(111)

    

        source = ColumnDataSource(data={})
        source.data, ax = plotme(allSNe[b], b, sncolors, axtype=axtype)

        #print (source.data)
        for k in source.data.keys():
            if isinstance(source.data[k][0], np.float32):
                print k, np.isnan(source.data[k]).sum()
        
        x = np.array(source.data['x']) #timeline
        if len(x)<2:
            continue
        #select max 2 obs per day
        source.data = select2ObsPerDay(source.data)
        #selectOneEpochPerDay(source.data)
        #remove phases >100 days
        indx1 = (x<100) & (~np.array(source.data['mask']))
        x = x[indx1] 

        if  len(x) == 0:
            continue
        
        #sort all epochs
        indx = np.argsort(x)
        x = x[indx]

        #create phases every hour from first dp to 100 days
        phases = np.arange(x[0], 100,  1.0/24.0)
        #observed magnitudes and errors
        y = - np.array(source.data['y'])[indx1][indx]
        yerr = np.array(source.data['yerr'])[indx1][indx]

        # save all anonymized photometric datapoints
        dataphases = {'x': x, 'y': y, 'yerr': yerr, 'phases': phases,
                  'allSNe': allSNe[b]}

        pkl.dump(dataphases, open('data/alldata_%s.pkl' % b, 'wb'))
        
        print ("calculating stats and average ", b)

        #fixing yerr that are nan tp 0.3 mag
        yerr = dataphases['yerr']
        yerr[np.isnan(yerr) * ~(np.isfinite(yerr))] = 0.3
        yerr[yerr==0] = 0.1

        #gaussian sigma 5 hours - empirically found to be good for most bands, esp V
        gsig = 5

        #set larger gaussian kernel for poorly observed bands (IR, u)
        if b in ['K','H','J','i','u','U']:
            gsig = gsig * 2

        #set weithed averge, median etc
        phs, wmu, wgstd, mu, med, std, wstd, dm, pc25, pc75 = wAverageByPhase(dataphases, gsig)
        dms[b] = dm

        wmu[std==0] = np.nan
        wmu = wmu - wmu[np.abs(phs) == min(np.abs(phs))]
        #mymean = np.empty_like(mu)*np.nan
        #tmp = movingaverage (dataphases['y'], 24 * 3 + 1,
        #                        1.0 / np.sqrt(dataphases['yerr']))
        #mymean[len(dataphases['y']) - len(tmp)/2 : - (len(dataphases['y']) - len(tmp)/2)] = tmp
        smoothedmean = np.empty_like(mu) * np.nan
        #smoothedmean[25:-25] = smooth(mu, window_len=51)[25:-25]
        #print ((dataphases['y']), (dataphases['x']), (dataphases['yerr']))
        #smoothedmean = wAverageByPhase
        #smoothIrregSampling(-y, x, yerr, window_len=2)
        #print len(smoothedmean)
        #print len(smoothedmean), len(mu), len(mymean)

        #ax[0].plot(phs, mu, 'k-', lw=1, alpha=0.7)
        #ax[0].plot(phs, smoothedmean, 'k-', lw=2)
        #ax[0].fill_between(phs, smoothedmean-std, smoothedmean+std,
        #                color = 'k', alpha=0.5)

        ax[0].plot(phs, med, 'r-', alpha=0.7, lw=2)
        ax[0].plot(phs[std>0], wmu[std>0], 'k-', lw=2)
        #ax[0].fill_between(phs[std>0], wmu[std>0]-std[std>0],
        #                   wmu[std>0]+std[std>0],
        #                   color = 'k', alpha=0.3)
        ax[0].fill_between(phs[std>0],
                           pc25[std>0],
                           pc75[std>0],
                           color = 'k', alpha=0.3)        
        ax[0].fill_between(phs[std>0], wmu[std>0]-wgstd[std>0],
                           wmu[std>0]+wgstd[std>0],
                           color = 'k', alpha=0.5)                
               
        ax1 = pl.subplot(ax[2][-1,0])
        #ax1.plot(phs, mu, 'g-', lw=1, alpha=0.7, label="rolling mean")
        for thisax in [ax1, axtype]:
            thisax.fill_between(phs[std>0],
                             pc25[std>0],
                             pc75[std>0],
                             color = 'k', alpha=0.3, label="IQR")
            #ax1.fill_between(phs[std>0], wmu[std>0]-std[std>0], wmu[std>0]+std[std>0],
            #                 color = 'k', alpha=0.3, label=r"$\sigma$")
            
            thisax.fill_between(phs[std>0], wmu[std>0]-wgstd[std>0],
                             wmu[std>0]+wgstd[std>0],
                             color = 'k', alpha=0.6, label=r"$\sigma$")        
            thisax.plot(phs, med, '-', color = 'IndianRed', alpha=0.7,
                     lw=2, label="median")
            thisax.plot(phs[std>0], wmu[std>0], 'k-', lw=2, label="mean")
            #ax1.plot(x, -smoothedmean, '-', color = 'SteelBlue', lw=2, label="smoothed")        
            #ax1.plot(phs, mymean, '-', color='yellow', alpha=0.7, lw=2,
            #         label="my mean")
            thisax.set_ylabel("relative magnitude", fontsize = 20)
            thisax.set_xlabel("phase (days since Vmax)", fontsize = 20)
            thisax.set_ylim(4.5, -1.5)#ax1.get_ylim()[1], ax1.get_ylim()[0])
            thisax.set_xlim(-27, 105)

        handles, labels = axtype.get_legend_handles_labels()
        artistIIb = pl.Line2D((0,1),(0,0), color=colorTypes['IIb'],
                              marker = 'o', linestyle='')
        artistIb = pl.Line2D((0,1),(0,0), color=colorTypes['Ib'],
                             marker = 'o', linestyle='')
        artistIc = pl.Line2D((0,1),(0,0), color=colorTypes['Ic'],
                             marker = 'o', linestyle='')
        artistIcbl = pl.Line2D((0,1),(0,0), color=colorTypes['Ic-bl'],
                               marker = 'o', linestyle='')
        artistother = pl.Line2D((0,1),(0,0), color=colorTypes['other'],
                                marker = 'o', linestyle='')
        axtype.legend([artistIIb, artistIb, artistIc, artistIcbl, artistother] + 
                      [handle for handle in handles],
                      ['IIb','Ib','Ic','Ic-bl','other'] +
                      [label for label in labels],
                      framealpha=0.5, ncol=2, prop={'size': 15})

        
        ax[1].set_ylim(4.5, -1.5)#ax[1].get_ylim()[0], ax[1].get_ylim()[1])
        #pl.show()
        ax1.grid(True)    

        pl.savefig("outputs/UberTemplate_%s.pdf" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b))
        fig.savefig("outputs/UberTemplate_%s_types.pdf" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b))

        #pl.show() 

        
        if BOKEHIT:
            bokehplot(source, np.asarray(wmu), np.asarray(wgstd), np.asarray(phs), b)
        
        ut = {}#pd.DataFrame()
        ut['phs'] = phs
        ut['med'] = med
        
        ut['mu'] = wmu
        ut['mu'][ut['mu'] == 0] = np.nan
        
        ut['std'] = std
        ut['wstd'] = wgstd        
        ut['pc25'] = pc25
        ut['pc75'] = pc75
        
        ut, ut['spl'] = wSmoothAverage(ut,  3)
        
        '''
        ut.to_csv("UberTemplate_%i.csv" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b))
        '''
        #ut['spl'] = ut['mu']
        
        pkl.dump(ut, open("outputs/UberTemplate_%s.pkl" % \
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b), 'wb'))
        # redo for R and V up to 40 days only
        
        if b in ['V', 'R']:
            #set weithed averge, median etc

            phs2, wmu2, wgstd2, mu2, med2, std2, wstd2, dm2, pc25_2, pc75_2 = wAverageByPhase(dataphases, gsig, phsmax=40)
            wmu2[std2==0] = np.nan
            wmu2 = wmu2 - wmu2[np.abs(phs2) == min(np.abs(phs2))]
            
            smoothedmean2 = np.empty_like(mu2) * np.nan
            ut2 = {}#pd.DataFrame()
            ut2['phs'] = phs2
            ut2['med'] = med2
            
            ut2['mu'] = wmu2
            ut2['mu'][ut2['mu'] == 0] = np.nan
            ut2['std'] = std2
            ut2['wstd'] = wgstd2            
            ut2['pc25'] = pc25_2
            ut2['pc75'] = pc75_2
            ut2, ut2['spl'] = wSmoothAverage(ut2,  2)
            

            pkl.dump(ut2, open("outputs/UberTemplate_%s40.pkl"%b, 'wb'))      

 
    pd.DataFrame.from_dict(dms, orient='index').\
        to_csv("outputs/dmsUberTemplates.csv")
               
if __name__ == '__main__':
    if len(sys.argv)>1:
        doall(b = [sys.argv[1]])
    else:
        doall()
