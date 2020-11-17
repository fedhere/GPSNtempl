#!/usr/bin/env python
from __future__ import print_function
import os
import inspect
import sys
import glob
import optparse
import time
import numpy as np
import scipy as sp
import numpy as np
import pylab as pl
from scipy import optimize
from scipy.interpolate import interp1d, splprep, splev
from scipy import stats as spstats 
from mpmath import polyroots
import pickle as pkl
from random import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor as GaussianProcess

GEORGE = True
WRITEPKL = False
WRITEPKL = True
READPKL = False
#READPKL = True
EXTCORR = True
EXTCORR = False
ABSMAG = True
ABSMAG = False
COLOR = True
COLOR = False

allcolors=["YellowGreen", "Aquamarine", "RoyalBlue", "Violet", "Tomato", "RosyBrown","Blue", "BlueViolet", "Brown", "BurlyWood", "CadetBlue", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Crimson", "Cyan", "DarkBlue", "DarkCyan", "DarkGoldenRod", "DarkGray", "DarkGreen", "DarkKhaki", "DarkMagenta", "DarkOliveGreen", "DarkOrange", "DarkOrchid", "DarkRed", "DarkSalmon", "DarkSeaGreen", "DarkSlateBlue", "DarkSlateGray", "DarkTurquoise", "DarkViolet", "DeepPink", "DeepSkyBlue", "DimGray", "DodgerBlue", "FireBrick", "Turquoise", "ForestGreen", "Fuchsia", "Gainsboro", "OliveDrab", "Gold", "GoldenRod", "Gray", "Green", "GreenYellow", "Wheat", "HotPink", "IndianRed", "Indigo",  "SteelBlue", "Khaki", "Lavender", "LavenderBlush", "LawnGreen","SpringGreen",  "Lime", "LimeGreen", "Linen", "Magenta", "Maroon", "MediumAquaMarine", "MediumBlue", "MediumOrchid", "MediumPurple", "MediumSeaGreen", "MediumSlateBlue", "MediumSpringGreen", "MediumTurquoise", "MediumVioletRed", "MidnightBlue", "MintCream", "MistyRose", "Moccasin", "Navy", "OldLace", "Olive", "OliveDrab", "Orange", "OrangeRed", "Orchid", "PaleGoldenRod", "PaleGreen", "PaleTurquoise", "PaleVioletRed", "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum", "PowderBlue", "Purple", "Red", "RosyBrown", "RoyalBlue", "SaddleBrown", "Salmon", "SandyBrown", "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue", "SlateGray", "SpringGreen", "SteelBlue", "Tan", "Teal", "Thistle", "Tomato", "Turquoise", "Violet", "Wheat","MistyRose", "Moccasin", "Navy", "OldLace", "Olive", "OliveDrab", "Orange", "OrangeRed", "Orchid", "PaleGoldenRod", "PaleGreen", "PaleTurquoise", "PaleVioletRed", "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum", "PowderBlue", "Purple", "Red", "RosyBrown", "RoyalBlue", "SaddleBrown", "Salmon", "SandyBrown", "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue", "SlateGray", "SpringGreen", "SteelBlue", "Tan", "Teal", "Thistle", "Tomato", "Turquoise", "Violet", "Wheat","MistyRose", "Moccasin", "Navy", "OldLace", "Olive", "OliveDrab", "Orange", "OrangeRed", "Orchid", "PaleGoldenRod", "PaleGreen", "PaleTurquoise", "PaleVioletRed", "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum", "PowderBlue", "Purple", "Red", "RosyBrown", "RoyalBlue", "SaddleBrown", "Salmon", "SandyBrown", "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue", "SlateGray", "SpringGreen", "SteelBlue", "Tan", "Teal", "Thistle", "Tomato", "Turquoise", "Violet", "Wheat"]


# checking env variables are declared
try:
     os.environ['SESNPATH']
     os.environ['SESNCFAlib']

except KeyError:
     print ("must set environmental variable SESNPATH and SESNCfAlib")
     sys.exit()
# loading SESNCFA library
cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib"+"/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *

wbreakout = ['03dh', '07d', '06aj', '13cq', '02ap']


#pl.ion()
#LN10x2p5=5.75646273249
TEMPLATEFIT = False
UPDATETEMPLATE = False

if __name__=='__main__':
     parser = optparse.OptionParser(usage="fitcfalc.py <'filepattern'> ",
                                    conflict_handler="resolve")

     parser.add_option('--showonly', default=False, action="store_true", 
                      help='show only')
     parser.add_option('--showme', default=False, action="store_true", 
                      help='show plots')

     parser.add_option('-b', '--band', default='all' , type="string", 
                      help='band to be user:  /shooter (UBVRI); /mini (UBVRrIi); ' +\
                       '/kepler (UBVRrIi)')
     parser.add_option('-i', '--inst', default='all' , type="string", 
                      help='instrument to be used: shooter, mini, keploer, or all')

     parser.add_option('-t', '--sntype', default='all' , type="string", 
                      help='sn type you want to work on')

     parser.add_option('-p', '--template', default=False, action="store_true" , 
                       help='whether you want to run the template fit or not')

     parser.add_option('-m', '--mode', default='simple' , type="string", 
                       help='"iterative" or "simple" the mode in which the template ' + \
                       'is used:  updated with each SN or kept at original ')

     parser.add_option('-s', '--spline', default=False, action="store_true" , 
                       help='if you want to run the cubic spline fit only')

     parser.add_option('-d', '--degree', default=3 , type="int", 
                       help='degree of polinomial fit')

     parser.add_option('-a', '--active', default=False, action="store_true" , 
                       help='active learning: show the fit before including in the ' + \
                       'training set -only if iterative is true')
     parser.add_option('-c', '--check', default=False, action="store_true" , 
                       help='just check how many good SNe we have')
     parser.add_option('--silent', default=False, action="store_true" , 
                       help='no print statements except error and good sn')
     parser.add_option('--george', default=False, action="store_true" , 
                       help='no print statements except error and good sn')


     options, args = parser.parse_args()
     print ("options", options, "arguments", args)

     if len(args)>1:
          sys.argv.append('--help')    
          options, args = parser.parse_args()
          sys.exit(0)
        
     # reading in lightcurves
     # if a SN name is passed only read the files for that SN
     if len(args)>0:
          fall = glob.glob(os.environ['SESNPATH']+"finalphot/*"+args[0]+"*.[cf]") + \
                 glob.glob(os.environ['SESNPATH']+"literaturedata/phot/*"+args[0]+"*.[cf]")
     # otherwise read all files in finalphot and literaturedata
     else:
          fall = glob.glob(os.environ['SESNPATH']+"finalphot/s*[cf]") + \
                 glob.glob(os.environ['SESNPATH']+"literaturedata/phot/s*[cf]")

     #print os.environ['SESNPATH'], fall

     # Turn shuffle on if you want to test if order matters. 
     #shuffle(fall)
     su = setupvars()

     # set photometric bands
     templog = []
     for b in su.bands:
          if b in ['i', 'r', 'u']:
               bname = b + 'p'
          else:
               bname = b
          templog.append ( open('logs/templog%s.tmp'%bname, 'a'))
         
     if options.band == 'all':
          nbands = len(su.bands)
     else:
          su.bands = [options.band]
          nbands = 1
          #print su.bands


     # fitting templates?     
     if options.template:
          TEMPLATEFIT = True
     # iterative templates? 
     if options.mode == 'iterative':
          UPDATETEMPLATE = True
     # normal mode: not fit and not update
     else:
          if options.active :
               options.active = False

          if not options.mode == 'simple':
               print ("modes available are 'simple' or 'iterative' " + \
                      "for the construction of the template")
               sys.exit()
    
     if options.active:
          if not options.silent:
               print ("interactive plotting")
          pl.ion()
     if options.inst == 'all':
          ninsts = len(su.insts)

     if not os.path.isfile('logs/lcvlog.dat'):
          logoutput = open("logs/lcvlog.dat", 'w')
          logoutput.write("##lcv type                           band       " + 
                          "inst    ndata rchisq deg mad  flagmissmax  flagmiss15 " +
                          "dm15 mjmx dm15lin")
     else:
          logoutput = open("logs/lcvlog.dat", 'a')

          
     if options.check:
          options.band = 'all'
          options.silent = True

     input_file = None
    
     showme = options.showme
     splinefit = options.spline

     # main figure with 2 panels for all lcvs and templates
     pl.figure()
     ngood = 0
     for ib, b in enumerate(su.bands):
          if b in ['i', 'r', 'u']:
               bname = b+'p'
          else:
               bname = b

          if not options.silent:
               print ("############################# band ############################",
                 b, "\n\n")

          if options.check:
               if ib>0 :
                    print ("number of good SNe of type", options.sntype, "in any filter: ", ngood)
                    sys.exit()
          # dictionary for pickling
          meanlcall = {}
          #list to host photometry
          meanlc = []
          #list to host SN names
          meanlccomponent = []
          meanlccolors = []          
          meanlcgeorge = []
          # read in the saved gaussian processes


          
          if READPKL:
               meanlcall = pkl.load(open("%s_%s_lcv4template_gp.pkl"%(options.sntype,
                                                                      bname), 'rb'))
               meanlc = np.array(meanlcall['lc'])
               meanlccomponent = meanlcall['names']

          # redo the gaussian processes
          else:
               if ib == 0:
                    if options.sntype == 'Ib/c':
                         os.system("mkdir outputs/Ibc_lcvs")
                    else:
                         os.system("mkdir outputs/%s_lcvs" % options.sntype)

               # for all SNe
               for f in fall:
                    if not options.silent:
                         print ("#############################  SN  ############################",
                           f, "\n\n")
                    bands=[b] #su.bands # su.photcodes.keys()

                    fnir = True
                    thissn = mysn(f, addlit=False, quiet=True)
                    
                    lc, mag, dmag, snname = thissn.loadsn(f, fnir, verbose=False)

                    if '05eo' in snname:
                         # its not a stripped SN
                         continue
                    # read in the big info file the first time and save it in input_file
                    if input_file is None:
                         try:
                              input_file, snn = thissn.readinfofileall(\
                                                          verbose=~options.silent,
                                                                  earliest=False,
                                                                  loose=True,
                                                                  quiet=True)
                         except:
                              continue
                    #if you already read the big info file just use it
                    else:
                         snn = thissn.setVmaxFromFile(input_file, verbose=False,
                                                earliest=False, loose=True,
                                                quiet=True)

                    # set Vmax and type
                    thissn.setsn(thissn.metadata['Type'], thissn.Vmax)

                    if not options.silent:
                         print (thissn.metadata['Type'], thissn.Vmax)

                    # select only the right type
                    if options.sntype and not options.sntype=='all':
                         if not thissn.type == options.sntype:
                              if not options.silent:
                                   print ("this SN type is not what you want: its a",
                                          thissn.type, "and you want type",
                                          options.sntype)
                              continue
                         else:
                              if not options.silent:
                                   print ("\n\n\n FOUND A ", thissn.type, ":",
                                          thissn.snnameshort, thissn.Vmax)
                    #raw_input()
                    # move on if Vmax is missing
                    try:
                         print ("Vmax:", thissn.snnameshort, float(thissn.Vmax))
                    except:
                         if not options.silent:
                              print ("no date of V max")
                         continue
                    
                    if float(thissn.Vmax) == 0 or np.isnan(thissn.Vmax):
                         if not options.silent:
                              print ("no date of V max")
                         continue

                    # correct for extinction
                    if EXTCORR:
                         try:
                              thisebmv = su.ebmvs[thissn.snnameshort] +\
                                         su.ebmvhost[thissn.snnameshort]    
                         except KeyError:
                              try:
                                   thisebmv = su.ebmvs[thissn.snnameshort] 
                                   #+  su.ebmvcfa[thissn.snnameshort]
                              except KeyError:
                                   continue
                    else:
                         thisebmv = 0.0
                    if ABSMAG:          
                         try:
                              distpc=float(thissn.metadata['distance Mpc'])*1e6
                         except:
                              if not options.silent:
                                   print ("failed on distance:", snname)
                                   #, thissn.metadata['distance Mpc']
                              continue
                              #raw_input()
               
                         dm = 5.0 * (np.log10(distpc) - 1)

                    # set up photometry
                    thissn.setphot()

                    if not options.check:
                         if thissn.filters[b]<3: continue
                    if not options.silent:
                         print ("band, datapoints for this SN and band",
                                b, thissn.filters[b])
                    thissn.getphot(ebmv=thisebmv)

                    # set up phase w respect to Vmax
                    thissn.setphase() #verbose=True)
                    thissn.sortlc()
                    thissn.printsn()
                    if not sum([thissn.filters[bb] for
                                bb in thissn.filters.keys()]):
                         continue
                    if options.check:
                         ngood +=1
                         continue
                    try:
                         print ("herehere", b, thissn.snnameshort)
                         print ("phase", thissn.photometry[b]['phase'][0])
                         print ("first mjd", thissn.photometry[b]['mjd'][0], thissn.Vmax)
                         print ("SN SUCCEEDED",  thissn.snnameshort)
                         #raw_input()
                    except:
                         print ("SN FAILED",  thissn.snnameshort)
                         #raw_input()
                         continue

                    if ib == 0:
                         thissn.plotsn(photometry=True)
                         if thissn.type == 'Ib/c':
                              pl.savefig("outputs/Ibc_lcvs/SN%s_all_lcv.png" % thissn.snnameshort)
                              pl.close()
                         else:
                              pl.savefig("outputs/%s_lcvs/SN%s_all_lcv.png" % (thissn.type,
                                                                               thissn.snnameshort))
                              pl.close()

                    if COLOR:
                         thissn.getcolors(BmI=False)
                         if ib == 0:
                              thissn.plotsn(color=True)
                              if thissn.type == 'Ib/c':
                                   pl.savefig("outputs/Ibc_lcvs/SN%s_all_color.png" % thissn.snnameshort)
                                   pl.close()
                              else:
                                   pl.savefig("outputs/%s_lcvs/SN%s_all_color.png" % (thissn.type,
                                                                                      thissn.snnameshort))
                                   pl.close()
                 
                    if not options.silent:
                         print ("################## working on SN ", f, "in band", b)
        
                    snlog=open('logs/'+thissn.name+'.log', 'w')

                    #look for sn in big info file
                    myphotcode = None 

                    minyall, maxyall=17, 17
                    # count the photometric points in this band
                    lph= len(thissn.photometry[b]['mag']) 

                    #if there are none continue
                    if not lph:
                         if not options.silent:
                              print ("no photometry in", b, "band")
                         continue

                    #correct relative to absolute photometry
                    if ABSMAG:
                         thissn.photometry[b]['mag'] -= dm
          
                    minyall=max(minyall, max(thissn.photometry[b]['mag'])+0.5)
                    maxyall=min(maxyall, min(thissn.photometry[b]['mag'])-0.5)

                    # setting photometry to 0 at Vmax
                    '''
                    if ib == 0:
                         for tmpb in su.bands:
                              thissn.photometry[tmpb]['mag'] -= \
                                        thissn.getepochmags(tmpb,
                                                            phase=0,
                                                            interpolate=True)[1]
                              thissn.photometry[tmpb]['dmag'] = \
                                        np.sqrt(thissn.photometry[tmpb]['dmag']**2 + \
                                                thissn.getepochmags(tmpb, phase=0,
                                                                    interpolate=True)[2]**2)

                              print (tmpb, thissn.getepochmags(tmpb,
                                                            phase=0,
                                                            interpolate=True)[1])
                    else:
                    '''
                    try:
                         thissn.photometry[b]['mag'] -= \
                                        thissn.getepochmags(b,
                                                            phase=0,
                                                            interpolate=True)[1]
                         thissn.photometry[b]['dmag'] = \
                                        np.sqrt(thissn.photometry[b]['dmag']**2 + \
                                                thissn.getepochmags(b, phase=0,
                                                                    interpolate=True)[2]**2)
                    except:
                         continue
                    

                    #append SN name
                    meanlccomponent.append(thissn.snnameshort)
                    meanlccolors.append(allcolors[list(snn)[0][0]])
                    #print "phase",thissn.photometry[b]['phase']
                    meanlc.append([thissn.photometry[b]['phase'],
                                   thissn.photometry[b]['mag'],
                                   thissn.photometry[b]['dmag']])
                    if GEORGE:
                         thissn.gpphot(b)
                         meanlcgeorge.append([thissn.gp['result'][b][0],
                                             thissn.gp['result'][b][1],
                                              thissn.gp['result'][b][2]])
               if WRITEPKL:
                         meanlcall['lc'] = meanlc
                         meanlcall['names'] = meanlccomponent
                         pkl.dump(meanlcall,
                                  open("%s_%s_lcv4template_gp.pkl"\
                                       %(options.sntype,
                                         bname),
                                       'wb'))
               meanlc=np.array(meanlc)
               if not options.silent:
                    print ("")
                    print ("data array shape", meanlc.shape, "SN to be used",
                      meanlccomponent)
               
               try:
                    # create the x array at intervals of 1/2 day between minimum epoch and 100
                    
                    #tsmoothx = np.arange(np.ceil(min(
                    #     np.concatenate(meanlc.T[0].flatten()))) - 1,
                    #                     100, 0.5)
                    tsmoothx  = np.arange(-15,100,0.5)
               except (IndexError, ValueError):
                    continue

               tind0 = np.where(tsmoothx==0)[0]
               tsmooth = []
               theta0 = [2]#, 0.005, 50.0, 5]
               thetaL = [10]#, 10.0, 0.01, 5]
               thetaU = [1000.0]#, 100, 1, 1]


               
               # create an empty array of size number of lcvs by number of regularized dps
               fftall = np.ma.array(np.zeros((len(meanlc), len(tsmoothx)),
                                             float))
               fftweightsall = np.zeros((len(meanlc), len(tsmoothx)), float)
               ffterrall = np.zeros((len(meanlc), len(tsmoothx)), float)
     
               np.random.seed(333)
               pl.clf()

               # iterate over saved lcvs
               for i, lc in enumerate(meanlc):
                    if isinstance(meanlcgeorge[i][0], float):
                         continue
                    if len(lc[1])<3: continue
                   
                    #subplot with lcvs
                    ax1 = pl.figure().add_subplot(211)
                    #lcvs figure
                    
                    fig4 = pl.figure()
                    fig4.clf()

                    ax4 = fig4.add_subplot(111)
                    if not options.silent:
                         print (i, meanlccomponent[i], min(lc[0]), max(lc[0]))

                    #plot the photometry 
                    ax1.errorbar(lc[0], lc[1], yerr=lc[2],
                                 label=meanlccomponent[i], color=meanlccolors[i])
                    ax4.errorbar(lc[0], lc[1], yerr=lc[2], fmt = '.',
                                 label=meanlccomponent[i], color=meanlccolors[i])

                    # add spline smoothed lcv
                    spline_curve = splprep(np.asarray([lc[0], lc[1]]), k=3)

                    tsmooth.append(splev(tsmoothx, spline_curve[0]))

                    # add a tiny offset so that no timestanps are identical
                    # make 2d and transpose
                    X = np.atleast_2d(np.log10(lc[0] + 1 - min(lc[0])
                                               + np.random.randn(len(lc[0]))
                                               * 0.01)).T
                    loop = 1
                    changed = 1
                    # one more check to make sure there are no identical points
                    while changed == 1:
                         changed = 0
                         for ii, x in enumerate(X[:-1]):
                              if x == X[ii + 1]:
                                   X[ii] -= 0.001
                                   changed=1
                    
                    if options.george:
                         tsmoothx = meanlcgeorge[i][0]
                         ff = meanlcgeorge[i][1]
                         ff_err = meanlcgeorge[i][2]
                         
                    else:
                         # do gaussian processes on the time series
                         gp = GaussianProcess()

                         print (lc[1])
                         gp.fit(X, lc[1])
                         
                         ff = gp.predict(np.atleast_2d(np.log10(tsmoothx +
                                                  1 - min(tsmoothx))).T, return_std=True, return_cov=False)

                         ff_err = np.sqrt(ff[1])

                         ff = ff[0]

#TODO: We should find out how we can find the best params of the new version of GaussianProcessRegressor
                         # if not options.silent:
                         #      print ("best-fit theta =", gp.theta_[0, 0])
         
                    mask = np.ones_like(tsmoothx) * False
                    mask[tsmoothx<lc[0].min()] = True
                    mask[tsmoothx>lc[0].max()] = True
                    if meanlccomponent[i] in wbreakout:
                         mask[tsmoothx<0] = True
                    ff = np.ma.array(ff,
                                     mask = mask)

                         
                    # plotting gps
                    ax1.plot(tsmoothx, ff, 'k-')
                    ax1.fill_between(tsmoothx, ff - 3.0 * ff_err,
                                     ff + 3.0 * ff_err,
                                    color='gray', alpha=0.2)

                    ax4.plot(tsmoothx, ff, 'k-')
                    ax4.fill_between(tsmoothx, ff - 1.0 * ff_err, ff + 1.0 * ff_err,
                                     color='gray', alpha=0.2)
                    ax4.fill_between(tsmoothx, ff - 3.0 * ff_err, ff + 3.0 * ff_err,
                                     color='gray', alpha=0.2)
                    ax4.set_ylim(ax4.get_ylim()[1], ax4.get_ylim()[0])

                    try:
                         ft0 = ff[tind0][0]
                    except IndexError:
                         ft0 = 0
                    fftall[i] = ff - ft0
                    ffterrall[i] = ff_err
                    fftweightsall[i] = 1.0 / ff_err**2
         
                    ax4.set_xlim(-20, 160)#
                    ax4.plot([0,0],[ax4.get_ylim()[1], ax4.get_ylim()[0]], 'k--')
                    ax4.set_ylabel("magnitude (peak=0)")
                    ax4.set_title("%s %s, %d, %s"%(meanlccomponent[i], b,
                                                   len(meanlc), options.sntype)) 

                    fig4.savefig("templatelcv/%s_%s_%s_templatelcv_gp.png"%(meanlccomponent[i],
                                                                options.sntype,
                                                                bname), dpi=150)
                    pl.close(fig4)
               if showme:
                    pl.show()

       

          ax1.set_ylim(ax1.get_ylim()[1], ax1.get_ylim()[0])
          #ax1.set_xlabel("phase (days)")
          ax1.set_xlim(-20, 160)
          ax1.set_ylabel("magnitude (peak=0)")
          ax1.set_title("%s, %d, %s"%(b, len(meanlc)-1, options.sntype))
          #pl.show()
          
          ax2 = pl.add_subplot(212)

          lcaverage = np.ma.average(fftall, axis=0, weights=fftweightsall)
          from smooth import *
          #lcerrN=1/np.sqrt(np.sum(fftweightsall, axis=0))#*len(meanlc)/np.sum(~fftall.mask, axis=0)

          #weithgs one are the inverse of the gp uncertainties
          #V1 = np.sum(1.0 / ff_err, axis=0)
          #weights two are the inverse of the gp undcertainties squared.
          #V2 = np.sum(fftweightsall, axis=0)

          #The square resudual over the sum of the uncertainties
          lcerrN = np.sqrt(np.sum(ffterrall**2, axis=0)) #/ (fftall.shape[0] -
                                               #fftall[:].mask.sum(axis=0))
          # * (fftall - lcaverage)**2, axis=0) #/ (V1 - V2 / V1)
          #/len(fftweightsall)
          #1/np.sqrt(np.sum(fftweightsall, axis=0))#*len(meanlc)/np.sum(~fftall.mask, axis=0)

          # the standard deviation 
          lcerr = np.std(fftall, axis=0)
          #1.0/np.sqrt(np.sum(fftweightsall, axis=0))
          #0.5*np.sqrt(np.sum((fftweightsall*fftall)**2, axis=0))/np.sqrt(np.product(fftall, axis=0))

          # smoothing with a Gaussian
          lcaverage.mask = [fftall[:].mask.sum(axis=0) ==
                            max(fftall[:].mask.sum(axis=0))]

          lcaverage_smooth = testGauss(tsmoothx, lcaverage, lcaverage.mask,
                                       10, 1000, sig=5)

          if WRITEPKL:
               picklestemplate = {}
               picklestemplate['phase'] = tsmoothx
               picklestemplate['lc'] = lcaverage
               picklestemplate['lcSmooth'] = lcaverage_smooth
               picklestemplate['lcerrN'] = lcerrN
               picklestemplate['lcstd'] = lcerr
               
               pkl.dump(picklestemplate,
                        open("templatelcv/%s_%s_templatelcv_gp.pkl"%(options.sntype, bname),
                             'wb'))
                        
          try:
               lcat0 = lcaverage[tsmoothx==0][0]
          except IndexError:
               lcat0 = 0

          ax2.errorbar(tsmoothx, lcaverage - lcat0,
                       yerr=lcerr, color='SteelBlue', alpha=0.5)
          try:
               lcast0 = lcaverage_smooth[tsmoothx==0][0]
          except IndexError:
               lcast0 = 0
               
          ax2.errorbar(tsmoothx, lcaverage_smooth - lcat0,
                       yerr=lcerrN, color='IndianRed', alpha=0.5)

          ax2.plot(tsmoothx, lcaverage - lcat0, 'k--')
          ax2.plot(tsmoothx, lcaverage_smooth - lcast0,
                   'k-')
          #ax2.set_ylim(4, -1)
          ax2.set_xlim(-20, 160)
          ax2.set_xlim(-20, 160) #30)     
          ax2.set_ylim(ax2.get_ylim()[1] + 0.5, ax2.get_ylim()[0] - 0.5)
          ax2.set_xlabel("phase (days)")
          ax2.set_ylabel("magnitude (peak=0)")
          #ax2.set_title("%s, %d, %s"%(b, i, options.sntype))
          #fig2.savefig("templatelcv_%s_%s.png"%(b, options.sntype))

          ax1.legend(fontsize=8, ncol=3)
          for i in range(len(tsmoothx)):
              if not options.silent:
                   print(tsmoothx[i],
                         lcaverage_smooth[i] - lcast0,
                     lcerr[i])
          #pl.show()
          pl.savefig("templatelcv/%s_%s_templatelcv_gp.png"%(options.sntype,
                                                   bname),
                       dpi=150)
          pl.close()

