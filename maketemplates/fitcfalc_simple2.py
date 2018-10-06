#!/usr/bin/env python
import os, inspect, sys, glob, optparse, time
import numpy as np
import scipy as sp
import numpy as np
import pylab as pl
from scipy import optimize
from scipy.interpolate import interp1d, spline
from scipy import stats as spstats 
from mpmath import polyroots
import pickle as pkl
from random import shuffle

#WRITEPKL=True
WRITEPKL=False
READPKL=False
#READPKL=True


try:
     os.environ['SESNPATH']
     os.environ['SESNCFAlib']

except KeyError:
     print "must set environmental variable SESNPATH and SESNCfAlib"
     sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib"+"/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
from leastsqbound import leastsqbound



#pl.ion()
#LN10x2p5=5.75646273249
TEMPLATEFIT = False
UPDATETEMPLATE = False

if __name__=='__main__':
    parser = optparse.OptionParser(usage="fitcfalc.py <'filepattern'> ", conflict_handler="resolve")

    parser.add_option('--showonly', default=False, action="store_true", 
                      help='show only')
    parser.add_option('--showme', default=False, action="store_true", 
                      help='show plots')

    parser.add_option('-b', '--band', default='all' , type="string", 
                      help='band to be user:  /shooter (UBVRI); /mini (UBVRrIi); /kepler (UBVRrIi)')
    parser.add_option('-i', '--inst', default='all' , type="string", 
                      help='instrument to be used: shooter, mini, keploer, or all')

    parser.add_option('-t', '--sntype', default='all' , type="string", 
                      help='sn type you want to work on')
    parser.add_option('-p', '--template', default=False, action="store_true" , 
                      help='whether you want to run the template fit or not')

    parser.add_option('-m', '--mode', default='simple' , type="string", 
                      help='"iterative" or "simple" the mode in which the template is used: updated with each SN or kept at original ')
    parser.add_option('-s', '--spline', default=False, action="store_true" , 
                      help='if you want to run the cubic spline fit only')
    parser.add_option('-d', '--degree', default=3 , type="int", 
                      help='degree of polinomial fit')

    parser.add_option('-a', '--active', default=False, action="store_true" , 
                      help='active learning: show the fit before including in the training set -only if iterative is true')


    options, args = parser.parse_args()
    print options, args


    if len(args)>1:
        sys.argv.append('--help')    
        options, args = parser.parse_args()
        sys.exit(0)
        
    if len(args)>0:
         fall = glob.glob(os.environ['SESNPATH']+"/finalphot/*"+args[0]+"*.[cf]")+glob.glob(os.environ['SESNPATH']+"/literaturedata/phot/*"+args[0]+"*.[cf]")
    else:
         fall=glob.glob(os.environ['SESNPATH']+"/finalphot/s*[cf]")+glob.glob(os.environ['SESNPATH']+"/literaturedata/phot/s*[cf]")
    #print os.environ['SESNPATH'], fall
    #shuffle(fall)
    su=setupvars()
    templog=[]
    for b in su.bands:
         templog.append ( open('templog%s.tmp'%b, 'a'))
         
    if options.band == 'all':
         nbands = len(su.bands)
    else:
         su.bands = [options.band]
         nbands = 1
         #print su.bands
        
    if options.template:
         TEMPLATEFIT = True
         
    if options.mode == 'iterative':
         UPDATETEMPLATE = True
    else:
         if options.active :
              options.active = False

         if not options.mode == 'simple':
              print "modes available are 'simple' or 'iterative' for the construction of the template"
              sys.exit()
    
    if options.active:
         print "interactive plotting"
         pl.ion()
    if options.inst == 'all':
        ninsts = len(su.insts)

    if not os.path.isfile('lcvlog.dat'):
        logoutput=open("lcvlog.dat", 'w')
        print >> logoutput, "##lcv type                           band       inst    ndata rchisq deg mad  flagmissmax  flagmiss15 dm15 mjmx dm15lin"
    else:
        logoutput=open("lcvlog.dat", 'a')

    input_file = None
    
        
    showme=options.showme
    splinefit=options.spline

    figcounter = 1
    lccount = 0
    dalpha = 1.0/(len(fall)/3.0)
    myalpha = dalpha
    lcnames = []
    fig1=pl.figure(1)
    for b in su.bands:
     meanlcall = {}
     meanlc = []
     meanlccomponent = []

     if READPKL:
          meanlcall = pkl.load(open("%s_%s_lcv4template_gp.pkl"%(options.sntype, b), 'rb'))
          meanlc = np.array(meanlcall['lc'])
          meanlccomponent = meanlcall['names']
     else:
       for f in fall:
          bands=[b]#su.bands # su.photcodes.keys()
          fnir = True
          thissn=mysn(f)
          lc, mag, dmag, snname = thissn.loadsn(f, fnir, verbose=False)
          if '05eo' in snname: continue
          
          if input_file is None:
               try:
                    input_file = thissn.readinfofileall(verbose=True, earliest=False,
                                                        loose=True)
               except:
                    continue
          else:
               thissn.setVmaxFromFile(input_file, verbose=True, earliest=False, loose=True)
          thissn.setsn(thissn.metadata['Type'], thissn.Vmax)

          print (thissn.metadata['Type'], thissn.Vmax)

          if options.sntype and not options.sntype=='all':
               if not thissn.sntype == options.sntype:
                    print ("this SN type is not what you want: its a",
                           thissn.sntype, "and you want type", options.sntype)
                    
                    continue
               else:
                    print "\n\n\n FOUND A ", thissn.sntype, ":", thissn.snnameshort, thissn.Vmax
                    #raw_input()
          
          try:
               print "Vmax:", float(thissn.Vmax) 
          except:
               print "no date of V max"
               #raw_input()
               continue
          if float(thissn.Vmax) ==0:
               print "no date of V max"
               #raw_input()
               #continue
          try:
               thisebmv=su.ebmvs[thissn.snnameshort] +  su.ebmvhost[thissn.snnameshort]    
          except KeyError:
               try:
                    thisebmv=su.ebmvs[thissn.snnameshort] #+  su.ebmvcfa[thissn.snnameshort]
               except KeyError:
                    continue
          try:
               distpc=float(thissn.metadata['distance Mpc'])*1e6
          except:
               print "failed on distance:", snname#, thissn.metadata['distance Mpc']
               #continue
               #raw_input()
               
          #dm= 5.0*(np.log10(distpc)-1)
          thissn.setphot()
          print ("here")
          print b, thissn.filters[b]
          #raw_input()
          thissn.getphot(ebmv=thisebmv)

          #print thissn.photometry
          thissn.setphase()#verbose=True)
          try:
               print "herehere", b, thissn.snnameshort,
               print thissn.photometry[b]['phase'][0],
               print thissn.photometry[b]['mjd'][0], thissn.Vmax
          except:
               print "FAILED"
               #raw_input()

               continue
          print thissn.photometry[b]['phase']
          #print thissn.photometry
          #raw_input()
          thissn.getcolors(BmI=False)
          found, redo = 0, 1
          mylegslist=[]
          myfig = myopenfig(0, (23, 15))
          templatefigs=[]
          
          print "##################working on SN ", f, 
        
          snlog=open(thissn.name+'.log', 'w')
          #look for sn in big info file
          myphotcode = None 
#          donephotcodes = []
          minyall, maxyall=17, 17
          lph= len(thissn.photometry[b]['mag']) 
          if lph>0:
               print "maxphot:", min(thissn.photometry[b]['mag']), thissn.sntype
          else:
               print "no photometry in", b, "band"
               continue

          #thissn.photometry[b]['mag']-=dm
          
          minyall=max(minyall, max(thissn.photometry[b]['mag'])+0.5)
          maxyall=min(maxyall, min(thissn.photometry[b]['mag'])-0.5)
          #print thissn.photometry
          #figure = pl.figure(3)
          #ax=figure.add_subplot(111)
          try:
               thissn.photometry[b]['mag']-=thissn.getepochmags(b, phase=0, interpolate=True)[1]
               thissn.photometry[b]['dmag']=np.sqrt(thissn.photometry[b]['dmag']**2+thissn.getepochmags(b, phase=0, interpolate=True)[2]**2)
          #ax.errorbar(thissn.photometry[b]['phase'], thissn.photometry[b]['mag'], yerr=thissn.photometry[b]['dmag'], marker='o')
          #thissn.plotsn(photometry=True, band=b, fig=0, symbol='%so'%su.mycolors[b])
          except:
               continue
          meanlccomponent.append(thissn.snnameshort)
          #print "phase",thissn.photometry[b]['phase']
          meanlc.append([thissn.photometry[b]['phase'], thissn.photometry[b]['mag'], thissn.photometry[b]['dmag']])

          lcnames.append(thissn.snnameshort)
          #continue
          #pl.show()
          #figure.clf()
     if WRITEPKL:
         meanlcall['lc']=meanlc
         meanlcall['names']=meanlccomponent
         pkl.dump(meanlcall, open("%s_%s_lcv4template_gp.pkl"%(options.sntype,
                                                               b), 'wb'))
     meanlc=np.array(meanlc)
     print ""
     print meanlc[:][:]
     print ""
     print meanlc.shape, meanlccomponent
     
     try:
          print (meanlc.T[0].shape)
          tsmoothx = np.arange(np.ceil(min(np.concatenate(meanlc.T[0].flatten()))),
                               100, 0.5)
     except (IndexError, ValueError): continue
     #print tsmoothx
     #sys.exit()
     tsmooth=[]
     theta0=[2]#, 0.005, 50.0, 5]
     thetaL=[10]#, 10.0, 0.01, 5]
     thetaU=[1000.0]#, 100, 1, 1]

     fftall=np.ma.array(np.zeros((len(meanlc), len(tsmoothx)), float))
     fftweightsall=np.zeros((len(meanlc), len(tsmoothx)), float)
     
     from sklearn.gaussian_process import GaussianProcess
     np.random.seed(333)
     fig1.clf()
     for i, lc in enumerate(meanlc):
         ax1=fig1.add_subplot(211)
         fig4=pl.figure(4)
         fig4.clf()
         ax4=fig4.add_subplot(111)
         print i, meanlccomponent[i], min(lc[0]), max(lc[0])

         ax1.errorbar(lc[0], lc[1], yerr=lc[2], label=meanlccomponent[i])
         ax4.errorbar(lc[0], lc[1], yerr=lc[2], fmt = '.', label=meanlccomponent[i])
         tsmooth.append(spline(lc[0], lc[1], tsmoothx, 1.5))
         #ax.plot(tsmoothx, tsmooth[i])
         X= np.atleast_2d(lc[0]+np.random.randn(len(lc[0]))*0.01).T
         loop=1
         changed=1
         while changed==1:
              changed=0
              for ii, x in enumerate(X[:-1]):
                   if x==X[ii+1]:
                        X[ii]-=0.001
                        changed=1
                   
         gp = GaussianProcess(
              regr='linear', corr='squared_exponential', theta0=theta0[0], 
              #thetaL=thetaL[0], thetaU=thetaU[0], 
              nugget=(lc[2]/10)**2, random_state=0)

         if len(lc[1])<2: continue
         gp.fit(X, lc[1])
         #print np.atleast_2d(tsmoothx).T
         ff, MSE = gp.predict(np.atleast_2d(tsmoothx).T, eval_MSE=True)
         ff_err = np.sqrt(MSE)
         
         #print "ff", ff
         #print "ff-err", ff-3*ff_err
         #print "ff+err", ff+3*ff_err
         print "best-fit theta =", gp.theta_[0, 0]
         #pl.plot(tsmoothx, ff, '-', color='black')
         
         ff=np.ma.array(ff, mask = np.array([tsmoothx<lc[0].min()])+np.array([tsmoothx>lc[0].max()]))
         ax1.plot(tsmoothx, ff, 'k-')
         ax1.fill_between(tsmoothx, ff - 3.0 * ff_err, ff + 3.0 * ff_err, color='gray',
                          alpha=0.2)
         ax1.fill_between(tsmoothx, ff - 3.0 * ff_err, ff + 3.0 * ff_err, color='gray',
                          alpha=0.2)
         ax4.plot(tsmoothx, ff, 'k-')
         ax4.fill_between(tsmoothx, ff - 3.0 * ff_err, ff + 3.0 * ff_err, color='gray',
                          alpha=0.2)
         ax4.set_ylim(ax4.get_ylim()[1], ax4.get_ylim()[0])
         if showme: pl.show()
         fftall[i]=ff
         fftweightsall[i]=1.0/ff_err**2
         
         ax4.set_ylim(7, -3)#ax1.get_ylim()[1], ax1.get_ylim()[0])
         #ax1.set_xlabel("phase (days)")
         ax4.set_xlim(-20, 160)#
         ax4.set_ylabel("magnitude (peak=0)")
         ax4.set_title("%s %s, %d, %s"%(lcnames[i], b, len(meanlc), options.sntype))
 
         fig4.savefig("%s_%s_%s_templatelcv_gp.png"%(lcnames[i], options.sntype, b), dpi=150)
       

     ax1.set_ylim(7, -3)#ax1.get_ylim()[1], ax1.get_ylim()[0])
     #ax1.set_xlabel("phase (days)")
     ax1.set_xlim(-20, 160)#
     ax1.set_ylabel("magnitude (peak=0)")
     ax1.set_title("%s, %d, %s"%(b, len(meanlc), options.sntype))
    #fig2=pl.figure(2)
     ax2=fig1.add_subplot(212)
     #print fftall.mask, np.sum(fftall.mask, axis=0)
     lcaverage=np.ma.average(fftall, axis=0, weights=fftweightsall)
     from smooth import *
     #lcerrN=1/np.sqrt(np.sum(fftweightsall, axis=0))#*len(meanlc)/np.sum(~fftall.mask, axis=0)
     print ff_err
     V1 = np.sum(1.0/ff_err, axis=0)
     V2 = np.sum(fftweightsall, axis=0)
     lcerrN = np.sum(fftweightsall*(fftall-lcaverage)**2, axis=0) / (V1 - V2/V1)#/len(fftweightsall)
     #1/np.sqrt(np.sum(fftweightsall, axis=0))#*len(meanlc)/np.sum(~fftall.mask, axis=0)
     lcerr = np.std(fftall, axis=0)#1.0/np.sqrt(np.sum(fftweightsall, axis=0))
     print "here", lcerrN, lcerr, 
     #print fftweightsall
     #print len(meanlc), np.sum(fftweightsall), fftweightsall, np.sum(~fftall.mask, axis=0)
#0.5*np.sqrt(np.sum((fftweightsall*fftall)**2, axis=0))/np.sqrt(np.product(fftall, axis=0))
     #lcerr=np.sqrt(np.sum((fftweightsall)**2, axis=0))
     #print lcaverage.shape
     lcaverage_smooth = testGauss(tsmoothx, lcaverage, 10, 1000, sig=3)

     ax2.errorbar(tsmoothx, lcaverage_smooth-lcaverage_smooth[tsmoothx==0], yerr=lcerr, color='SteelBlue', alpha=0.5)
     ax2.errorbar(tsmoothx, lcaverage_smooth-lcaverage_smooth[tsmoothx==0], yerr=lcerrN, color='IndianRed', alpha=0.5)
     ax2.plot(tsmoothx, lcaverage, 'k--')
     ax2.plot(tsmoothx, lcaverage_smooth, 'k-')
     ax2.set_ylim(4, -1)#
     ax2.set_xlim(-20, 160)#
     ax2.set_xlim(-20, 30)#     
     #ax2.set_ylim(ax2.get_ylim()[1], ax2.get_ylim()[0])
     ax2.set_xlabel("phase (days)")
     ax2.set_ylabel("magnitude (peak=0)")
     #ax2.set_title("%s, %d, %s"%(b, i, options.sntype))
     #fig2.savefig("templatelcv_%s_%s.png"%(b, options.sntype))
     #for i, x in range(tsmoothx):
     ax1.legend(fontsize=8, ncol=2)
     for i in range(len(tsmoothx)):
          print tsmoothx[i], lcaverage_smooth[i]-lcaverage_smooth[tsmoothx==0][0],lcerr[i]

     fig1.savefig("%s_%s_templatelcv_gp.png"%(options.sntype, b), dpi=150)
     
