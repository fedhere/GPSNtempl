import numpy as np
import glob
import os,inspect,sys
from makePhottable import *

try:
     os.environ['SESNPATH']
     os.environ['SESNCFAlib']

except KeyError:
     print "must set environmental variable SESNPATH and SESNCfAlib"
     sys.exit()

RIri = False

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib")+"/templates"
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

hexcolors=['#999999','#333333','#000000','#FFCC00','#FF9900','#FF6600','#FF3300','#99CC00','#CC9900','#FFCC33','#FFCC66','#FF9966','#FF6633','#CC3300','#CC0033','#CCFF00','#CCFF33','#333300','#666600','#999900','#CCCC00','#FFFF00','#CC9933','#CC6633','#330000','#660000','#990000','#CC0000','#FF0000','#FF3366','#FF0033','#99FF00','#CCFF66','#99CC33','#666633','#999933','#CCCC33','#FFFF33','#996600','#993300','#663333','#993333','#CC3333','#FF3333','#CC3366','#FF6699','#FF0066','#66FF00','#99FF66','#66CC33','#669900','#999966','#CCCC66','#FFFF66','#996633','#663300','#996666','#CC6666','#FF6666','#990033','#CC3399','#FF66CC','#FF0099','#33FF00','#66FF33','#339900','#66CC00','#99FF33','#CCCC99','#FFFF99','#CC9966','#CC6600','#CC9999','#FF9999','#FF3399','#CC0066','#990066','#FF33CC','#FF00CC','#00CC00','#33CC00','#336600','#669933','#99CC66','#CCFF99','#FFFFCC','#FFCC99','#FF9933','#FFCCCC','#FF99CC','#CC6699','#993366','#660033','#CC0099','#330033','#33CC33','#66CC66','#00FF00','#33FF33','#66FF66','#99FF99','#CCFFCC','#CC99CC','#996699','#993399','#990099','#663366','#660066','#006600','#336633','#009900','#339933','#669966','#99CC99','#FFCCFF','#FF99FF','#FF66FF','#FF33FF','#FF00FF','#CC66CC','#CC33CC','#003300','#00CC33','#006633','#339966','#66CC99','#99FFCC','#CCFFFF','#3399FF','#99CCFF','#CCCCFF','#CC99FF','#9966CC','#663399','#330066','#9900CC','#CC00CC','#00FF33','#33FF66','#009933','#00CC66','#33FF99','#99FFFF','#99CCCC','#0066CC','#6699CC','#9999FF','#9999CC','#9933FF','#6600CC','#660099','#CC33FF','#CC00FF','#00FF66','#66FF99','#33CC66','#009966','#66FFFF','#66CCCC','#669999','#003366','#336699','#6666FF','#6666CC','#666699','#330099','#9933CC','#CC66FF','#9900FF','#00FF99','#66FFCC','#33CC99','#33FFFF','#33CCCC','#339999','#336666','#006699','#003399','#3333FF','#3333CC','#333399','#333366','#6633CC','#9966FF','#6600FF','#00FFCC','#33FFCC','#00FFFF','#00CCCC','#009999','#006666','#003333','#3399CC','#3366CC','#0000FF','#0000CC','#000099','#000066','#000033','#6633FF','#3300FF','#CC999CC','#33CCFF','#66CCFF','#6699FF','#3366FF','#0033CC','#3300CC','#00CCFF','#0099FF','#0066FF','#0033FF', '#999999','#333333','#000000','#FFCC00','#FF9900','#FF6600','#FF3300','#99CC00','#CC9900','#FFCC33','#FFCC66','#FF9966','#FF6633','#CC3300','#CC0033','#CCFF00','#CCFF33','#333300','#666600','#999900','#CCCC00','#FFFF00','#CC9933','#CC6633','#330000','#660000','#990000','#CC0000','#FF0000','#FF3366','#FF0033','#99FF00','#CCFF66','#99CC33','#666633','#999933','#CCCC33','#FFFF33','#996600','#993300','#663333','#993333','#CC3333','#FF3333','#CC3366','#FF6699','#FF0066','#66FF00','#99FF66','#66CC33','#669900','#999966','#CCCC66','#FFFF66','#996633','#663300','#996666','#CC6666','#FF6666','#990033','#CC3399','#FF66CC','#FF0099','#33FF00','#66FF33','#339900','#66CC00','#99FF33','#CCCC99','#FFFF99','#CC9966','#CC6600','#CC9999','#FF9999','#FF3399','#CC0066','#990066','#FF33CC','#FF00CC','#00CC00','#33CC00','#336600','#669933','#99CC66','#CCFF99','#FFFFCC','#FFCC99','#FF9933','#FFCCCC','#FF99CC','#CC6699','#993366','#660033','#CC0099','#330033','#33CC33','#66CC66','#00FF00','#33FF33','#66FF66','#99FF99','#CCFFCC','#CC99CC','#996699','#993399','#990099','#663366','#660066','#006600','#336633','#009900','#339933','#669966','#99CC99','#FFCCFF','#FF99FF','#FF66FF','#FF33FF','#FF00FF','#CC66CC','#CC33CC','#003300','#00CC33','#006633','#339966','#66CC99','#99FFCC','#CCFFFF','#3399FF','#99CCFF','#CCCCFF','#CC99FF','#9966CC','#663399','#330066','#9900CC','#CC00CC','#00FF33','#33FF66','#009933','#00CC66','#33FF99','#99FFFF','#99CCCC','#0066CC','#6699CC','#9999FF','#9999CC','#9933FF','#6600CC','#660099','#CC33FF','#CC00FF','#00FF66','#66FF99','#33CC66','#009966','#66FFFF','#66CCCC','#669999','#003366','#336699','#6666FF','#6666CC','#666699','#330099','#9933CC','#CC66FF','#9900FF','#00FF99','#66FFCC','#33CC99','#33FFFF','#33CCCC','#339999','#336666','#006699','#003399','#3333FF','#3333CC','#333399','#333366','#6633CC','#9966FF','#6600FF','#00FFCC','#33FFCC','#00FFFF','#00CCCC','#009999','#006666','#003333','#3399CC','#3366CC','#0000FF','#0000CC','#000099','#000066','#000033','#6633FF','#3300FF','#CC999CC','#33CCFF','#66CCFF','#6699FF','#3366FF','#0033CC','#3300CC','#00CCFF','#0099FF','#0066FF','#0033FF']

allcolors=["YellowGreen", "Aquamarine", "RoyalBlue", "Violet", "Tomato", "RosyBrown","Blue", "BlueViolet", "Brown", "BurlyWood", "CadetBlue", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Crimson", "Cyan", "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray", "DarkGreen", "DarkKhaki", "DarkMagenta", "DarkOliveGreen", "DarkOrange", "DarkOrchid", "DarkRed", "DarkSalmon", "DarkSeaGreen", "DarkSlateBlue", "DarkSlateGray", "DarkTurquoise", "DarkViolet", "DeepPink", "DeepSkyBlue", "DimGray", "DodgerBlue", "FireBrick", "Turquoise", "ForestGreen", "Fuchsia", "Gainsboro", "OliveDrab", "Gold", "GoldenRod", "Gray", "Green", "GreenYellow", "Wheat", "HotPink", "IndianRed", "Indigo",  "SteelBlue", "Khaki", "Lavender", "LavenderBlush", "LawnGreen","SpringGreen",  "Lime", "LimeGreen", "Linen", "Magenta", "Maroon", "MediumAquaMarine", "MediumBlue", "MediumOrchid", "MediumPurple", "MediumSeaGreen", "MediumSlateBlue", "MediumSpringGreen", "MediumTurquoise", "MediumVioletRed", "MidnightBlue", "MintCream", "MistyRose", "Moccasin", "Navy", "OldLace", "Olive", "OliveDrab", "Orange", "OrangeRed", "Orchid", "PaleGoldenRod", "PaleGreen", "PaleTurquoise", "PaleVioletRed", "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum", "PowderBlue", "Purple", "Red", "RosyBrown", "RoyalBlue", "SaddleBrown", "Salmon", "SandyBrown", "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue", "SlateGray", "SpringGreen", "SteelBlue", "Tan", "Teal", "Thistle", "Tomato", "Turquoise", "Violet", "Wheat"]

from snclasses import *
from templutils import *
from sklearn import gaussian_process
import optparse
import readinfofile as ri

import pandas as pd
import pickle as pkl
import snclasses as snstuff 

su=setupvars()

def gpme2(data, kernel, nugget):
     nonan = ~np.isnan(data['y'])*(np.isfinite(data['y']))
     print np.isnan(data['x']), 
     print np.isnan(data['x'][nonan]) 
     print np.isnan(data['y']),
     print np.isnan(data['y'][nonan]) 
     print np.isnan(data['yerr']),
     print np.isnan(data['yerr'][nonan])

     x = data['x'][nonan]+0.001*np.random.randn(len(data['x'][nonan]))
     XX =(x - x.min() + 0.1)
     X = np.atleast_2d(np.log(XX)).T
     gp = gaussian_process.GaussianProcess(theta0=kernel[0], thetaL=kernel[1], thetaU=kernel[2], nugget=nugget)
     print X
     print data['y'][nonan]
     gp.fit(X, data['y'][nonan])  
     XXX = np.atleast_2d(np.log(data['phases'] - data['phases'][0] + 0.1 )).T
     print XXX, data['phases'], np.log(data['phases'])
     mu, std = gp.predict(XXX, eval_MSE=True)
     pl.plot(data['phases'],mu,'k-',linewidth=2)
     pl.fill_between(data['phases'],mu-std,mu+std,alpha=0.3,color='k')
     pl.show()

def gpme(data, kernel):
     XX = np.log(data['x'] - data['x'].min() + 0.1)
     gp = george.GP(kernel)
     gp.compute(XX, data['yerr'])
     mu, cov = gp.predict(data['y'], xx)
     std = np.sqrt(np.diag(cov))
     pl.plot(data['phases'],mu,'k-',linewidth=2)
     pl.fill_between(data['phases'],mu-std,mu+std,alpha=0.3,color='k')

def plotme(data, b):
    pl.figure(figsize = (15,15))
    pl.title(b + "(%d)"%(np.array([1 for tmp in data]).sum()))

    sourcedata = dict(
         id =  [],
         x = [],
         y = [],
         yerr = [],
         colors = []   )
    
    for i,tmp in enumerate(data['phase']):
        if len(tmp)==0: continue
        #print "tmp", tmp
        indx = np.argsort(tmp)
        magoffset = np.where(data['mag'][i] == min(data['mag'][i]))[0]
        #print magoffset, data['mag'][i][magoffset], tmp[magoffset]
        if len(magoffset)>1:
             magoffset = magoffset[(np.abs(tmp[magoffset]) == \
                                   np.abs(tmp[magoffset]).min())[0]]
             #print magoffset, data['mag'][i][magoffset], tmp[magoffset]
        if np.abs(tmp[magoffset]) > 1 and (np.abs(tmp)<1).any():
             magoffset = np.where(np.abs(tmp) == np.min(np.abs(tmp)))[0]
             #print magoffset, data['mag'][i][magoffset],  tmp[magoffset]
        if not isinstance(magoffset, int) and len(magoffset)>1: 
             magoffset = magoffset[(data['mag'][i][magoffset] == \
                                    min(data['mag'][i][magoffset]))[0]]
        sourcedata['id'] =  sourcedata['id']+[data['name'][i]]*len(indx)
        sourcedata['x'] = sourcedata['x']+list(tmp[indx])
        sourcedata['y'] = sourcedata['y']+list(-(data['mag'][i][indx]-data['mag'][i][magoffset]))
        sourcedata['yerr'] = sourcedata['yerr']+list(data['dmag'][i][indx])
        sourcedata['colors'] = sourcedata['colors']+[hexcolors[::3][i]]*len(indx)
        #print sourcedata['colors']

        pl.errorbar(tmp[indx],
                    data['mag'][i][indx] - data['mag'][i][magoffset],
                    yerr=data['dmag'][i][indx],
                    fmt='-', color = allcolors[i],
                    label=data['name'][i], alpha=0.8)
        
    pl.legend(framealpha=0.5, ncol=3, numpoints=1, prop={'size':12})
    pl.ylim(pl.ylim()[1], pl.ylim()[0])
    print  pl.xlim(), pl.ylim()
    return sourcedata, pl.xlim(), pl.ylim()
       
def errorbar(fig, source):
             #x, y, xerr=None, yerr=None, color='red', point_kwargs={}, error_kwargs={}):

  fig.circle('x', 'y', size=5, source=source,
             color='grey', fill_color='colors', alpha=0.5)


  y_err_x = []
  y_err_y = []
  for px, py, err in zip(source.data['x'], source.data['y'],  source.data['yerr']):
       y_err_x.append((px, px))
       y_err_y.append((py - err, py + err))
  fig.multi_line(y_err_x, y_err_y, color='colors', alpha=0.5)

def doall():
  allGPs = {}
  for b in su.bands:
     allGPs[b] = {'mag':[], 'dmag':[], 'phase':[], 'name':[]}
     
  inputSNe = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values
  
  bands1 = ['U','u','B','V','R','r','I','i']
  bands2 = ['w2','m2','w1','H','J','K']

  tmp1={}
  for b in bands1:
       tmp1[b]={}
       tmp1[b+"[min,max]"] = {}
  tmp2={}
  for b in bands2:
       tmp2[b]={}
       tmp2[b+"[min,max]"] = {}
   
  for f in inputSNe[:]:
    print f
    thissn = snstuff.mysn(f, addlit=True)
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        continue
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    print (" looking for files ")
    if np.isnan(thissn.Vmax) or thissn.Vmax == 0 :
        if '06gi' in thissn.snnameshort:
             try:
                  print ("getting max from GP maybe?")
                  thissn.gp =  pkl.load(open('gplcvs/' + f + \
                                             "_gp_ebmv0.00.pkl", "rb"))
                  if thissn.gp['maxmjd']['V'] < 2400000 and thissn.gp['maxmjd']['V'] > 50000:
                       thissn.Vmax = thissn.gp['maxmjd']['V'] + 2400000.5
                  else: 
                       thissn.Vmax = thissn.gp['maxmjd']['V']

                  print "GP vmax", thissn.Vmax
                  #if not raw_input("should we use this?").lower().startswith('y'):
                  #    continue
             except IOError:
                  continue


    

    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        continue
    print (" starting loading ")
    lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
    thissn.setphot()
    thissn.getphot()
    if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
        continue
    
    #thissn.plotsn(photometry=True)
    thissn.setphase()
    print (" finished ")    
    thissn.printsn()
    print su.bands
    add2DF(thissn, tmp1, tmp2, bands1, bands2)

    for b in su.bands:
         print b
         if not thissn.gp['max'][b] is None:
              offset = thissn.gp['max'][b][0]
              moffset = thissn.gp['max'][b][1]
         else:
              offset = None
              #continue
              #if offset is None:
              moffset = snstuff.coffset[b]
              print b, thissn.snnameshort
         print "Vmax, offset", thissn.Vmax, offset
         print offset
         allGPs[b]['mag'].append(thissn.photometry[b]['mag']-moffset)
         allGPs[b]['dmag'].append(thissn.photometry[b]['dmag'])
         allGPs[b]['phase'].append(thissn.photometry[b]['phase'])#-\
                                  #offset*0.1)        
         allGPs[b]['name'].append(thissn.snnameshort)
  print allGPs

  print allGPs['V']['phase']

  bands = []
  for b in bands1:
       bands.append(b)
       bands.append(b+"[min,max]")

  tabletex = "../../papers/SESNexplpars/tables/AllPhotOptTable.tex"
  add2table(tmp1[bands1], bands, tabletex)
    

  bands = []
  for b in bands2:
       bands.append(b)
       bands.append(b+"[min,max]")
  tabletex = "../../papers/SESNexplpars/tables/AllPhotUVNIRTable.tex"
  add2table(tmp2[bands2], bands, tabletex)
   
 
  
  for b in su.bands:

    print ""

    from bokeh.plotting import Figure as figure
    from bokeh.plotting import save as save
    from bokeh.models import ColumnDataSource, BoxZoomTool, HoverTool
    #, HBox, VBoxForm, BoxSelectTool, TapTool
    #from bokeh.models.widgets import Select
    #Slider, Select, TextInput
    from bokeh.io import gridplot
    from bokeh.plotting import output_file
    source = ColumnDataSource(
         data = {})
          
    source.data, xlim, ylim = plotme(allGPs[b], b)
    
    pl.savefig("GPdata_%s.png"%(b+'p' if b in ['u','r','i'] else b))
    #pl.show()
    htmlout = "GPdata_%s.html"%(b+'p' if b in ['u','r','i'] else b)
    output_file(htmlout)
    print (htmlout)
    TOOLS = "tap,wheel_zoom,box_zoom,reset"
    
    hover = HoverTool(
         tooltips=[
              ("ID", "@id"),
              ("phase", "@x"),
              ("Delta mag", "@y"),
              ("Error", "@yerr"),
         ])
    
    p = figure(plot_width=800, plot_height=600,
                            tools=TOOLS, title=b)
 
    #p.set(x_range=Range1d(xlim[0], xlim[1]), y_range=Range1d(ylim[1],ylim[0]))
    errorbar(p, source)
    #p.circle('x', 'y', size=7, source=source,
    #         color='grey', fill_color='colors', alpha=0.5)
    p.xaxis.axis_label = "phase (days since Vmax)"
    p.yaxis.axis_label = "Mag"
    print xlim,ylim

    p.add_tools(hover)
    save(p)
    x = np.concatenate([tmp for tmp in allGPs[b]['phase']])
    if  len(x)==0: continue
    indx = np.argsort(x)
    x = x[indx]
    #x = np.log(x[indx] - x[indx[0]] + 0.1) 

    phases = np.arange(x[0], x[-1], 0.1)
    y = np.concatenate([tmp for tmp in allGPs[b]['mag']])[indx]
    yerr = np.concatenate([tmp for tmp in allGPs[b]['dmag']])[indx]
    
    dataphases = {'x':x, 'y':y, 'yerr':yerr, 'phases':phases, 
                  'allGPs':allGPs[b]}
    pkl.dump(dataphases, open('alldata_%s.pkl'%b, 'wb'))
    continue

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

    pl.show()
     
def getgps():
     for b in su.bands:
          data = pkl.load(open("alldata_%s.pkl"%b, "rb"))
          plotme(data, b)
          gpme2(data, [0.3,None,None], data['yerr']**2)

if __name__ == '__main__':
     doall()
