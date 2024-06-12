#!/usr/bin/env python

import sys,os,glob, inspect
#,re,numpy,math,pyfits,glob,shutil,glob
import optparse
import scipy as sp
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
from scipy import optimize
#from mpmath import polyroots
import time
import pprint, pickle
import seaborn as sns
from sklearn.utils import shuffle

#from snclasses import myass#setupvars


bandscols = {'U':1,'B':3,'V':5,'R':7,'I':9,'g':13, 'r':15, 'i':17,'z':11, 'w1':-1,'w2':-2,'w1':-3}
TEMPLDIR=os.path.realpath(os.path.abspath(os.path.split(inspect.getfile\
( inspect.currentframe() ))[0]))
MINX,MAXX=-10,40
NEW_X=np.linspace(MINX,MAXX,(MAXX-MINX)*10)

def smoothListGaussian(data,strippedXs=False,degree=15):  
     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=np.zeros(len(data),float)
     for i in range(len(smoothed)-window):
          smoothed[i+window/2+1]=np.sum(np.array(data[i:i+window])*weight)/np.sum(weight)  
     smoothed[0:window/2]=data[0:window/2]
     smoothed[-window/2:]=data[-window/2:]
     return np.array(smoothed)  






class setupvars:
    def __init__(self):
        self.types=['Ib','Ic','IIb','Ic-bl','IIb-n']
        self.bands =['U','u','B','V','R','I','g','r','i', 'J','H','K','w1','w2','m2']
        self.bandoffsets = {'U': 2, 'u': 2, 'B': 1, 'V': 0, 'R': -1, 'I': -2,
                            'r': -1, 'i': -2, 'g': 0,
                            'J': -3, 'H': -4, 'K': -5, 'w1': 3, 'm2': 4, 'w2': 5}
        self.coffset = {'U': -3.3, 'u': -3.3,
                       'B': -2.3,
                       'V': 0, 'g': 0,
                       'R': 1.8, 'r': 1.8,
                       'I': 3.5, 'i': 3.5,
                       'J': 8.5, 'H': 10.1, 'K': 10.5,
                       'w1': 2600. * 1.42e-3 - 8.32,
                       'w2': 1928. * 1.42e-3 - 8.32,
                       'm2': 2246. * 1.42e-3 - 8.32}
        self.bandsnonir =['U','u','B','V','R','I','g','r','i']
        self.bandsnir =['J','H','K']
        self.bandsindex ={'U':0,'u':0,'B':1,'V':2,'R':3,'I':4,'g':5,'r':6,'i':7,
                          'J':8,'H':9,'K':10, 'w1':11,'w2':12,'m2':13}
        self.cbands = ['U-B','B-i',
                       'B-I','B-V',
                       'V-R','R-I','r-i','V-r', 'V-i','V-I','r-g', 'g-i','V-g', 'R-g',
                       'V-H', 'r-K','V-K','H-K',
                       'B-r', 'B-J', 'B-H', 'B-K', 'H-i', 'J-H', 'K-J']
        self.cs = {'U-B':0,'B-V':1,'V-R':2,'R-I':3,'r-i':4,'V-r':5, 'V-i':6,
                   'V-I':7, 'V-H':8,'I-H':9,'H-J':9,'J-K':10,'r-g':19, 'g-i':20,'V-g':21, 'R-g':22,
                   'B-I':11,'B-i':11,'r-K':12,'V-K':13,'H-K':14,
                   'B-r':15,'B-R':19, 'B-J':16, 'B-H':17, 'B-K':18}
        self.photcodes = {'U':('01','06','Ul'), 'B':('02','07','Bl'),
                          'V':('03','08','Vl'), 'R':('04','09','Rl'),
                          'I':('05','0a','Il'), 'r':('13','0b','rl'),
                          'g':('16','16','gl'),'i':('14','0c','il'), 'u':('15','15','ul'),
                          'H':('H','H','Hl'), 'J':('J','J','Jl'), 'K':('K','K','Kl'),
                          'm2':('m2','m2','m2'), 'w2':('w2','w2','w2'), 'w1':('w1','w1','w1')}
        self.mycolors = {'U':'k','u':'k','B':'#0066cc','V':'#47b56c','R':'#b20000','I':'m',
                         'r':'#b20000','i':'m', 'g': '#317e4b',
                         'J':'#4F088A','H':'#FFB700','K':'#A4A4A4',
                         'm2':'#708090', 'w2':'#a9b2bc', 'w1':'#434d56'}
        self.mycolorcolors = {'U-B':'k','B-V':'#0066cc','V-R':'#47b56c','R-I':'#b20000',
                              'V-I':'m','V-i':'m','V-r':'#47b56c','r-i':'#b20000',
                              'V-H':'#9999EE','I-H':'#9999EE','J-K':'#70B8FF','H-J':'#FFCC80',
                              'r-K':'purple', 'V-K':'SlateBlue', 'B-I':'#0B0B3B',
                              'H-K':'#FFCC80', 'r-g': '#317e4b', 'g-i':'#b54759','V-g': '#a347b5', 'R-g':'#b56c47',
                              'B-i':'#0B0B3B','B-r':'#0B0B3B' ,'B-R':'#0B0B3B',
                              'w1':'k','w2':'k','m2':'k'}
        self.myshapes = {'U':'^','u':'^','B':'s','V':'o','R':'v','I':'>',
                         'r':'d','i':'h','J':'^','H':'s','K':'o', 'g':'o',
                         'w1':'v','w2':'v','m2':'v'}
        self.mytypecolors = {'Ib':'k','Ic':'b','IIb':'g','Ic-bl':'r','IIb-n':'y', 'other':'k'}
        self.mysymbols = {'Ib':'o','Ic':'s','IIb':'^','Ic-bl':'v','IIb-n':'>', 'other':'x'}
        self.mylines = {'Ib':'dashed','Ic':'solid','IIb':'solid','Ic-bl':'dotted',
                        'IIb-n':'solid', 'other':'solid'}
        self.instcodes = {'kepler':0,'shooter':0,'mini':1}
        self.insts =['shooter', 'kepler', 'mini']

        self.ebmvs={'83V':0.0178, '93J':0.0690,'94I':0.0302,'95F':0.0311,
                    '95bb':0.0948,'96cb':0.0262,'97X':0.0237,'97dq':0.0382,
                    '97ef':0.0366,'98dt':0.0219,'98fa':0.0382,'00H':0.1964,
                    '99dn':0.0451,'01ai':0.0081,'01ej':0.0460,'01gd':0.0098,
                    '02ap':0.0620,'02ji':0.0217,'03dh':0.0214,'03jd':0.0784,
                    '04aw':0.0180,'04ao':0.0893,'04dk':0.1357,'04dn':0.0415,
                    '04eu':0.0466,'04fe':0.0210,'04ff':0.0281,'04ge':0.0754,
                    '04gk':0.0247,'04gq':0.0627,'04gt':0.0398,'04gv':0.0286,
                    '05U':0.0143,'05ar':0.0394,'05az':0.0097,'05bf':0.0385,
                    '05da':0.2483,'05ek':0.1811,'05eo':0.0585,'05hg':0.0901,
                    '05kf':0.0378,'05kl':0.0219,'05kz':0.046,'05la':0.0100,
                    '05mf':0.0153,'05nb':0.0320,'06F':0.1635,'06T':0.0647,
                    '06aj':0.1267,'06ba':0.0452,'06bf':0.0216,'06cb':0.0094,
                    '06ck':0.0245,'06el':0.0973,'06ep':0.0310,'06fo':0.0250,
                    '06gi':0.0205,'06ir':0.0393,'06jc':0.0173,'06lc':0.0556,
                    '06ld':0.0144,'06lv':0.0245,'06ss':0.0178,'07C':0.0363,
                    '07D':0.2881,'07I':0.0250,'07ag':0.0250,'07aw':0.0338,
                    '07bg':0.0179,'07ce':0.0200,'07cl':0.0370,'07gr':0.0535,
                    '07hb':0.0518,'07iq':0.1182,'07ke':0.0954,'07kj':0.0691,
                    '07ru':0.2254,'07rz':0.1723,'07uy':0.0194,'08D':0.0194,
                    '08an':0.0450,'08aq':0.0383,'08ax':0.0186,'08bo':0.0513,
                    '08cw':0.0060,'08hh':0.0427,'09bb':0.0844,'09er':0.0389,
                    '09iz':0.0729,'09jf':0.0971,'09mg':0.0388,'09K':0.0491,
                    '03bg':0.0197,'10as':0.1472,'11dh':0.0308,'11ei':0.0506,
                    '07Y':0.0184, '99ex':0.0172, '07c':0.0363,'07d':0.2881,
                    '07i':0.0250,'06f':0.1635,'06t':0.0647,'13df':0.017,
                    '98bw':0.0509, '03lw':0.9040,'10bh':0.1000,'13dx':0.0368,
                    '11bm':0.0289,'11fu':0.0664,'11hs':0.0107,'13cq':0.0174,
                    '12bz':0.0303,'PTF10vgv':0.1382,'PTF10qts':0.0252,'iPTF13bvn':0.0436}

#        self.ebmvhost={'02ap':0.03,'03jd':0.10,'04aw':0.352,'07Y':0.09,'07gr':0.038,'07ru':0.01,'08D':0.59,'08ax':0.28}
#        self.ebmvhost={'02ap':0.03,'03jd':0.10,'04aw':0.352,'07gr':0.038,'07ru':0.01,'08D':0.5406}#,'08ax':0.3814}
#        self.ebmvhost={'02ap':0.03,'04aw':0.352,'07gr':0.03,'08D':0.5406}
        self.ebmvhost={'02ap':0.03,'03jd':0.10,'04aw':0.352,'07gr':0.038,'99ex':1.0,#Hamuy02
                       '07ru':0.01,'08D':0.5406,'07Y':0.112,#Stritzinger09
                       '08ax':0.4, '03bg':0,'11ei':0.18,'11dh':0,#Taubenberger11
                       #'09jf':0.05, #Valenti11
#                       '08bo':0.24,
#                       '08ax':0.3814, 
#                       '04dk':0.201,'04dn':0.5265,
#                       '04fe':0.294,
#                       #'04ff':0.274,
#                       '04gq':0.19,'05az':0.43,'05kz':0.47,
#                       '05hg':0.63,
#                       '05mf':0.383,
#                       '06el':0.21,
#                       #'06C':0.65,
#                       '05bf':0.007,
#                       '07uy':0.601, '09jf':0.0146
                       '11bm': 0.032# valenti13
                        }

        self.ebmvcfa={'02ap':0.03,'03jd':0.10,'04aw':0.352,
                      '07ru':0.01,'08D':0.5406,
                      '06el' : 0.147  , '06ep':   0.448  ,'06fo'  : 0.201,
                      '07gr':   0.0462,
                      '07kj' :  0.295  ,'07uy' :  0.378  ,'08bo' :  0.294,
                      '09er' :  0.110  ,'09iz' :  0.064  ,'09jf' :  0.045,
                      '05bf':   0.05  ,'05hg' :  0.244  ,'05kl' :  1.344,
                      '05kz' :  0.437  ,'05mf' :  0.231  ,'06aj' :  0.141,
                      '06bf' :  0.368  ,'06F' :   0.533  ,'06lv' :  0.574  ,
                      '06T' :   0.397  ,'07ag' :  0.627  ,'07C' :   0.650  ,
                      '07ce' :  0.082  ,'07cl' :  0.258  ,'08D' :   0.572  }

#'09iz':0.1,
#07grxt Drout reddening via Chen 2014 was 0.06, drout is 0.038
#05hg Cano reddening 0.63 WRONG! from drout photometry?
#09iz guessing the reddening
##'06aj':0.04, gives trouble although reddening should be well constrained!!
        self.AonEBmV={'U': 5.434, 'u':5.155, 'B': 4.315, 'V': 3.315, 'R': 2.673,
                      'I': 1.940, 'r': 2.751,'i': 2.086, 'g': 3.315,
                      'J': 0.902,'H': 0.576, 'K': 0.367,
                      'uu':5.155,'vv':3.315,'w1':4.3,'w2':4.3,'m2':4.3}

    def allsne_colormaps(self, n, colormap = 'colorblind'):
        clrs = sns.color_palette(colormap, n_colors=n)
        clrs = shuffle(clrs, random_state=42)
        self.allsne_colors = clrs


class atemplate:
     def __init__(self):
          self.median=None
          self.mean  =None
          self.std   =None
          self.x     =None
          self.tempfuncy=None

class templatesn:
     def __init__(self, sn, dist=0.0, e_dist=0.0, incl=0.0, ebv=0.0, sntype='',
                  mjdmax=0.0, e_mjdmax=0.0, ebmvhost=0.0, vmr=0.0,peak=0.0,
                  e_peak=0.0, peak_ap=0.0, e_peak_ap=0.0,
                  phot=None, absphot=None, normphot=None,
                  new_x=None, new_y=None, yfunc=None):
        self.sn=sn
        self.dist=dist
        self.e_dist=e_dist
        self.incl=incl
        self.ebv=ebv
        self.type=sntype
        self.mjdmax=mjdmax
        self.e_mjdmax=e_mjdmax
        self.ebmvhost=ebmvhost
        self.vmr=vmr
        self.peak=peak
        self.e_peak=e_peak
        self.peak_ap=peak_ap
        self.e_peak_ap=e_peak_ap

        self.phot = phot
        self.absphot = absphot
        self.normphot = normphot
        if new_x:
             self.new_x = new_x
        else:
             self.new_x = NEW_X
        self.new_y = new_y        

        self.yfunc=yfunc


class gptemplclass:
    def __init__(self):
        self.su=setupvars()
        self.gptemplate = {}
        for b in self.su.bands:
             self.gptemplate[b]=atemplate()

        self.typetemplate= {'Ib':None,'Ic':None,'IIb':None,'Ic-bl':None,'IIb-n':None}

        
class Mytempclass:
    def __init__(self):
        self.su=setupvars()
        self.template = {}
        for b in self.su.bands:
             self.template[b]=atemplate()

        self.typetemplate= {'Ib':None,'Ic':None,'IIb':None,'Ic-bl':None,'IIb-n':None}

    def calctemplate(self,tptype=None):
         thissn = mysn(f, addlit=True)
         thissn.readinfofileall(verbose=False, earliest=False, loose=True)
         if tptype is None:
              tytype = self.typetemplate

        
    def loadtemplate(self, b, x=None,mean=None,median=None,std=None):       
        if b in self.su.bands:
            print ("ready to load temaplte")
            self.template[b].x=x
            self.template[b].mean=mean
            self.template[b].median=median
            self.template[b].std=std
        else: 
            print ("wrong band ", b)

    def gettempfuncy(self,b):
        if 1:#not self.template[b].tempfuncy():
            from scipy.interpolate import interp1d
            self.template[b].tempfuncy = interp1d(self.template[b].x,
                                                  self.template[b].mean, kind='cubic',
                                                  bounds_error=False)
            if np.sum(np.isnan(self.template[b].tempfuncy(self.template[b].x))) == \
               len(self.template[b].x) or \
               np.std(np.array(self.template[b].tempfuncy(self.template[b].x)\
                               [~np.isnan(self.template[b].tempfuncy(
                                    self.template[b].x))]))>10:
                self.template[b].tempfuncy = interp1d(self.template[b].x,
                                                      self.template[b].mean,
                                                      kind='linear', bounds_error=False)

    def tempfuncstd(self,b):
        from scipy.interpolate import interp1d
        tempfuncstd = interp1d(self.template[b].x,self.template[b].std,
                               kind='cubic', bounds_error=False)
        if np.sum(np.isnan(tempfuncstd(self.template[b].x)))==len(self.template[b].x) \
        or np.std(np.array(tempfuncstd(self.template[b].x)\
                           [~np.isnan(tempfuncstd(self.template[b].x))]))>10:
            tempfuncstd = interp1d(self.template[b].x,self.template[b].std,
                                   kind='linear', bounds_error=False)


    def loadtemplatefile(self, new=False, sntype=None):
        for b in self.su.bands:
             print (b, self.template[b].mean)
             if self.template[b].mean:
                 print ("template already read in")
                 continue
###preparing template functions"
             if new:
                  if not sntype:
                       print ("must input sn type")
                       sys.exit()
                  myfile ='templates/new/mytemplate'+b+'_'+sntype+'.pkl'
             else:
                  myfile='templates/mytemplate'+b+'.pkl'
             if not os.path.isfile(myfile):
                   print ("file not there: ", myfile)
                   continue
             pkl_file = open(myfile,'rb')
             self.template[b] = pickle.load(pkl_file)
             pprint.pprint(self.template[b])

    def templateupdate(self,s):
         if self.typetemplate[sn.type]==None:
              typetemplate[sn.type]=copy.deepcopy(template)
         else:
            
              for b in ['V','R']:#su.bands:
                new_x = NEW_X
                new_y = interp1d(sn.photometry[b]['mjd']-sn.Vmax, s['normphot'],
                                 kind='cubic', bounds_error=False)(s['new_x'])
                
                
                print (np.sum(np.isnan(np.array(new_y))), np.isnan(np.array(new_y).all()),
                       len(new_y))
                if np.sum(np.isnan(np.array(new_y)))== len(new_y):
                    print ("REDOING SPLINE WITH LINEAR")
                    new_y = interp1d(sn.photometry[b]['mjd']-sn.Vmax, 
                                     sn.photometry[b]['mag']-sn.maxdata,
                                     kind='linear', bounds_error=False)(s['new_x'])
                    
                if np.std(np.array(new_y[~np.isnan(new_y)]))>10:
                    new_y = interp1d(sn.photometry[b]['mjd']-sn.Vmax,
                                     sn.photometry[b]['mag']-sn.maxdata,
                                     kind='linear', bounds_error=False)(s['new_x'])
                print (s['new_y'], s['normphot'])
                print (s['sn'],np.min(s['new_y'][~np.isnan(s['new_y'])]))

                typetemplate[sn.type][b]=stats.stats.nanmean([new_y,
                                                              typetemplate[sn.type][b]],
                                                             axis=0)
                typetemplate[sn.type][b]=stats.stats.nanmean([new_y,
                                                              typetemplate[sn.type][b]],
                                                             axis=0)

def mycavvaccaleib(x,p, secondg=False,earlyg=False, verbose=False):
     if verbose:
          print ("\np,x",p,x)
     try:
          #needed in case i am passing an array of 1 element for x
          if verbose:
               print (len(x))
          if len(x)==1:
               x=x[0]
     except:
          pass
     latebump=False
     if p==None:
          return (x)*99e9
     if p[8]>p[1]:
          if verbose:
               print ("late bump")
          latebump=True

          
     '''
     fit the magnitudes with a vacca leibundgut (1997) analytical model 
     
     p is the parameter list
     if secondg=1: secondgaussian added
     if secondg=0: secondgaussian not    
     parameters are: 
     p[0]=first gaussian normalization (negative if fitting mag)
     p[1]=first gaussian mean
     p[2]=first gaussian sigma
     p[3]=linear decay offset
     p[4]=linear decay slope
     p[5]=exponxential rise slope
     p[6]=exponential zero point
     p[7]=second gaussian normalization (negative if fitting mag)
     p[8]=second gaussian mean
     p[9]=second gaussian sigma
     '''
     g=p[4]*(x)+p[3]
     g+=p[0]*np.exp(-(x-p[1])**2/p[2]**2)
     g*=(np.exp(-p[5]*(x-p[6]))+1)
     if latebump and  earlyg:
          g*=1e5
     if secondg:
          g+=p[7]*np.exp(-(x-p[8])**2/p[9]**2)    
     if latebump and p[8]-p[1]<15 :
          g+=p[7]*np.exp(-(x-p[8])**2/p[9]**2)
     try:
          len(g)
     except TypeError:
          g=[g]

          #(np.zeros(len(g),float)+1)
     if p[8]-p[1]>70:
          g+=(np.zeros(len(g),float)+1)
     return g


def loadlitlist(band):        
    print (TEMPLDIR)
    try:
        f=TEMPLDIR+"/"+"templatelist"+band+".txt"
        templates=np.loadtxt(f,usecols=(0,1,2,3,4,5,7,8,9,10,13,14,15,16),
                             dtype={'names': ('sn','dist','e_dist','incl','ebv',
                                              'type','mjdmax','e_mjdmax',
                                              'peak_ap','e_peak_ap','ebmvhost',
                                              'vmr','peak','e_peak'),
                                    'formats': ('S6','f','f','f','f','S5','f',
                                                'f','f','f','f','f','f','f')},
                             skiprows=1, comments='#')

        print ("reading files list ",f," worked")
    except:
        print ("reading files list ",f," failed")


    sne=[]
    print ("the templates are: ",templates['sn'])

    for i,sn in enumerate(templates['sn']):         
        sne.append(templatesn(sn,dist=templates[i]['dist'],
                              e_dist=templates[i]['e_dist'],
                              incl=templates[i]['incl'],
                              ebv=templates[i]['ebv'],
                              sntype=templates[i]['type'],
                              mjdmax=templates[i]['mjdmax'],
                              e_mjdmax=templates[i]['e_mjdmax'],
                              ebmvhost=templates[i]['ebmvhost'],
                              vmr=templates[i]['vmr'],
                              peak=templates[i]['peak'],
                              e_peak=templates[i]['e_peak'],
                              peak_ap=templates[i]['peak_ap'],
                              e_peak_ap=templates[i]['e_peak_ap']))

#    for i,sn in  enumerate(templates['sn']):
        sne[i].phot=(np.loadtxt(TEMPLDIR+"/"+"sn"+sn+".dat",
                                usecols=(0,bandscols[band], bandscols[band]+1),
                                skiprows=1, unpack=1))
        #dereddening
        sne[i].phot[1]+=3.2*sne[i].ebv

        sne[i].absphot=sne[i].phot[1]-(sne[i].peak_ap-sne[i].peak)
        sne[i].normphot=sne[i].absphot-sne[i].peak
        #sne[i]['phot'][1]-2.5*np.log10((sne[i]['dist']/1e6/10.0)**2)
#    mysn=np.where(templates['sn']=='1994I')[0]
#    print sne[mysn]
##        Mv = m - 2.5 log[ (d/10)2 ].
##        flux = 10**(-lc['mag']/2.5)*5e10
##        dflux = flux*lc['dmag']/LN10x2p5
#        print sn
#    
    print ("loaded sne")
    return sne, templates

def splinetemplates(sne):
    for s in sne: 
#        pl.figure()
#        minx=max(-20.1,min(s['phot'][0]-s['mjdmax']))+0.1
#        maxx=min(40.1,max(s['phot'][0]-s['mjdmax']))-0.1
        s.new_x = NEW_X
        s.new_y = interp1d(s.phot[0]-s.mjdmax, s.normphot, kind='cubic',
                           bounds_error=False)(s.new_x)
        NaN = float('nan')

        print (np.sum(np.isnan(np.array(s.new_y))), np.isnan(np.array(s.new_y).all()),
               len(s.new_y))
        if np.sum(np.isnan(np.array(s.new_y)))== len(s.new_y):
            print ("REDOING SPLINE WITH LINEAR")
            s.new_y = interp1d(s.phot[0]-s.mjdmax, s.normphot, kind='linear',
                               bounds_error=False)(s.new_x)
            
        if np.std(np.array(s.new_y[~np.isnan(s.new_y)]))>10:
            s.new_y = interp1d(s.phot[0]-s.mjdmax, s.normphot, kind='linear',
                               bounds_error=False)(s.new_x)
#        print (s['new_y'], s['normphot'])
        

            
        print (s.sn,np.min(s.new_y[~np.isnan(s.new_y)]))

    return sne
