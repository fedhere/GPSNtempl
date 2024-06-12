import sys,os,glob, inspect
#,re,numpy,math,pyfits,glob,shutil,glob
import optparse
import scipy as sp
import numpy as np
import pprint, pickle
import pylab as pl
from numpy import nanmean,nanmedian,nanstd

LN10x2p5=5.75646273249
def is_empty(any_structure, verbose=False):
    if any_structure:
        if verbose: print('Structure is not empty.')
        return False
    else:
        print('Structure is empty.')
        return True

def smoothIrregSampling(x, t, yerr, window_len=11, window='hamming'):
    import scipy.stats as spst
    
    if x.ndim != 1:
        raise ValueError ("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError ("Input vector needs to be bigger than window size.")
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError ("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    tuse = t#[]
    print (tuse)
    return np.array([np.nansum(x  * (1.0/yerr*spst.norm.pdf(t - dp, scale=window_len))) /
                     (1.0/yerr*spst.norm.pdf(t - dp)).sum()
                     for dp in tuse])
    return np.array([np.nansum(x / yerr * spst.norm.pdf(t - dp, scale=window_len)) /
                     np.nansum(1.0 / yerr) /
                     spst.norm.pdf(t - dp).sum()
                     for dp in tuse])
  
def loadsn(f):
    if f.split('/')[-1].startswith('slc'):
        print ("lightcurve type 1 ",f)
        try:
            lc=np.loadtxt(f,usecols=(0,1,5,6,7),\
                          dtype={'names': ('photcode','mjd',\
                                           'mag','dmag','ccmag'),\
                                 'formats': ('S2', 'f', 'f','f','f')})
            
            flux = 10**(-lc['mag']/2.5)*5e10
            dflux = flux*lc['dmag']/LN10x2p5
        except:
            try:
                lc=np.loadtxt(f,usecols=(0,1,5,6),\
                              dtype={'names': ('photcode','mjd',\
                                               'mag','dmag'),\
                                     'formats': ('S2', 'f', 'f','f')})
                
                flux = 10**(-lc['mag']/2.5)*5e10
                dflux = flux*lc['dmag']/LN10x2p5
            except:
                print ("file ",f," failed. wrong file format? moving on")
                return 0,0,0,0
            #            dflux = 10**(-lc['dmag']/2.5)
    elif f.split('/')[-1].startswith('sn'):
        print ("lightcurve type II")
        try:
            lc=np.loadtxt(f,usecols=(0,1,6,7,8,4,5),
                          dtype={'names': ('photcode','mjd','mag','dmag',
                                           'ccmag', 'flux', 'dflux'),\
                                 'formats': ('S2', 'f', 'f','f','f','f','f')})
            lc['photcode']=['%02d'%int(p) for p in lc['photcode']]
            #                print lc['photcode']
            flux = lc['flux']
            dflux=lc['dflux']
        except:
            print ("file ",f," failed. wrong file format? moving on")
            return 0,0,0,0
    else:
        print ("what kind of file is this??? "+f)
        return 0,0,0,0

    snname=f.split('/')[-1].split('.')
    for s in snname:
        if 'sn9' in s: name = s.replace('sn','sn19')
        elif 'sn0' in s: name = s.replace('sn','sn20')
        elif 'sn1' in s: name = s.replace('sn','sn20')
	    
    return lc,flux,dflux, name



def binMean(X,Y,numBins=8,xmin=None,xmax=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    bins = np.linspace(xmin,xmax,numBins+1)
#    print bins,Y

    YY = np.array([nanmean(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYmedian = np.array([nanmedian(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return bins[:-1]+(bins[1]-bins[0])*0.5,YY,YYmedian,YYstd

def loadpickledsn(snname, printsn=False):
	myfile=snname+'.pkl'
	if not os.path.isfile(myfile):
		print ("file not there: ", myfile)
		sys.exit()
	pkl_file = open(myfile,'rb')
	print (myfile, pkl_file)
	sn= pickle.load(pkl_file)
        
	if printsn:
		sn.printsn(template=True,printlc=False,photometry=True,color=False, extended=True)

	return sn
