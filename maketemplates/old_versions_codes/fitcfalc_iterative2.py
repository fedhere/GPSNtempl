#!/usr/bin/env python
import os,inspect,sys,glob, optparse,time
import numpy as np
import scipy as sp
import numpy as np
import pylab as pl
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats 
from mpmath import polyroots
import pickle as pkl
from random import shuffle

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

    parser.add_option('-b','--band', default='all' , type="string",
                      help='band to be user:  /shooter (UBVRI); /mini (UBVRrIi); /kepler (UBVRrIi)')
    parser.add_option('-i','--inst', default='all' , type="string",
                      help='instrument to be used: shooter, mini, keploer, or all')

    parser.add_option('-t','--sntype', default=None , type="string",
                      help='sn type you want to work on')
    parser.add_option('-p', '--template',default=False,action="store_true" ,
                      help='whether you want to run the template fit or not')

    parser.add_option('-m','--mode', default='simple' , type="string",
                      help='"iterative" or "simple" the mode in which the template is used: updated with each SN or kept at original ')
    parser.add_option('-s', '--spline',default=False,action="store_true" ,
                      help='if you want to run the cubic spline fit only')
    parser.add_option('-d','--degree', default=3 , type="int",
                      help='degree of polinomial fit')

    parser.add_option('-a','--active', default=False,action="store_true" ,
                      help='active learning: show the fit before including in the training set -only if iterative is true')


    options,  args = parser.parse_args()
    print options,args


    if len(args)>1:
        sys.argv.append('--help')    
        options,  args = parser.parse_args()
        sys.exit(0)
        
    if len(args)>0:
         fall = glob.glob(os.environ['SESNPATH']+"/finalphot/*"+args[0]+"*.[cf]")
    else:
         fall=glob.glob(os.environ['SESNPATH']+"/finalphot/s*[cf]")
    #print os.environ['SESNPATH'],fall
    shuffle(fall)
    su=setupvars()
    templog=[]
    for b in su.bands:
         templog.append ( open('templog%s.tmp'%b,'a'))
         
    if options.band == 'all':
         nbands = len(su.bands)
    else:
         su.bands = [options.band]
         nbands = 1
         print su.bands
        
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
        logoutput=open("lcvlog.dat",'w')
        print >>logoutput,"##lcv type                           band       inst    ndata rchisq deg mad  flagmissmax  flagmiss15 dm15 mjmx dm15lin"
    else:
        logoutput=open("lcvlog.dat",'a')


    
        
    showme=options.showme
    splinefit=options.spline

    figcounter=1
    lccount=0
    dalpha=1.0/(len(fall)/3.0)
    myalpha = dalpha

    for f in fall:
          bands=su.bands # su.photcodes.keys()
          print bands
          fnir = True
          thissn=mysn(f)
          lc,mag,dmag, snname = thissn.loadsn(f,fnir, verbose=False)
          if '05eo' in snname: continue
          thissn.readinfofileall(verbose=False, earliest=False, loose=True) 
          thissn.setsn(thissn.metadata['Type'],thissn.Vmax)
          if options.sntype:
               if not thissn.sntype == options.sntype:
                    print "this SN type is not what you want: its a", thissn.sntype, "and you want type", options.sntype
                    
                    continue
               else:
                    print "\n\n\n FOUND A ",thissn.sntype,":",thissn.snnameshort
          
          
          try:
               print "Vmax:", float(thissn.Vmax) 
               
          except:
               print "no date of V max"
               continue
          if thissn.snnameshort not in su.ebmvhost and  thissn.snnameshort not in su.ebmvcfa:
               print "no extinction correction"
               continue
          try:
               thisebmv=su.ebmvs[thissn.snnameshort] +  su.ebmvhost[thissn.snnameshort]    
          except KeyError:
               thisebmv=su.ebmvs[thissn.snnameshort] +  su.ebmvcfa[thissn.snnameshort]            
          try:
               distpc=float(thissn.metadata['distance Mpc'])*1e6
          except:
               print "failed on distance:", snname, thissn.metadata['distance Mpc']
               continue
               
          dm= 5.0*(np.log10(distpc)-1)
          thissn.setphot()
          thissn.getphot(ebmv=thisebmv)
          thissn.getcolors(BmI=False)
          found,redo = 0,1
          mylegslist=[]
          myfig = myopenfig(0,(23,15))
          templatefigs=[]
          
          print "##################working on SN ", f,
        
          snlog=open(thissn.name+'.log','w')
          #look for sn in big info file
          myphotcode = None 
#          donephotcodes = []
          minyall,maxyall=17,17
          for b in su.bands:
             for fig in range(0,figcounter+1):
                    pl.figure(fig)
                    pl.clf()
             for  fig in range(10,10+figcounter+1):
                    pl.figure(fig)
                    pl.clf()
             figcounter=1
          

             print b, thissn.filters[b]
             lph= len(thissn.photometry[b]['mag']) 
             if lph>0:
                  print "maxphot:", min(thissn.photometry[b]['mag']), thissn.sntype
             else:
                  print "no photometry in", b, "band"
                  continue

             thissn.photometry[b]['mag']-=dm

             minyall=max(minyall,max(thissn.photometry[b]['mag'])+0.5)
             maxyall=min(maxyall,min(thissn.photometry[b]['mag'])-0.5)
          continue
          for b in thissn.filters.iterkeys():
                    inst='any'
#               for inst in su.insts:
##############################setting up band#########################
#                    if (inst == 'shooter' and b not in ['U','B','V','R','I']) or ((inst == 'kepler' or inst == 'mini') and b not in ['U','B','V','r','i']) :
#                         print "skip filter/instrument combination %s/%s" %(b,inst)
#                         continue
#                    print "band ",b, " instrument ",inst
#                    
#                    try:
#                         myphotcode=su.photcodes[b]
#                         print "my photcode: ",myphotcode
#                         if myphotcode in donephotcodes:
#                              continue
#                    except:
#                         print "invalid photecode"
#                         continue
#
#                    donephotcodes.append(myphotcode)


##############################starting   fitting####################
####################################################################

##############################polynomial fitting####################
                    if thissn.filters[b]>2:
                         itercount,redo = 1,1
                         deg=options.degree
                         solution=[]
                         rchis=[]
                         print "number of datapoints with this filter:", thissn.filters[b]
                         #pars= np.polynomial.polynomial.polyfit(x,y, options.degree, rcond=None, full=False)          
                         x = thissn.photometry[b]['mjd'].astype(np.float64) 
                         y = thissn.photometry[b]['mag'].astype(np.float64)
                         print x,y
                         #raw_input()
                         yerr=thissn.photometry[b]['dmag']

                         xp=np.linspace(min(x),max(x),10000)
                         
                         raw_input()
                         if splinefit:
                              thissn.snspline[b] = interp1d(x,y, kind='linear', bounds_error=False)
                              pl.figure(0)
                              thissn.plotsn(photometry=True,band=b,fig=0,symbol='%so'%su.mycolors[b])
#                              pl.plot(np.arange(min(x),max(x)), thissn.snspline[b](np.arange(min(x),max(x))))
                              pl.draw()
                         

                              myfig=pl.figure(figcounter)

                              #raw_input("got here")
#                              thissn.plotsn(photometry=True,band=b,fig=figcounter,symbol='%so'%su.mycolors[b])

                              #thissn.plotsn(photometry=True,band=b, fig=figcounter, show=showme, save=False, symbol='.',title='%s band: %s instrument: %s spline'%(thissn.name,b,inst), plotpoly=False, plotspline=True, verbose=True, plottemplate=False)
                              fullrange=np.arange(min(x),max(x),0.1)
                              pl.plot(fullrange-53000,thissn.snspline[b](fullrange),'k-')
                              pl.show()

#                              sys.exit()
                              myfig.savefig("%s.%s.%s_spline.png"%(thissn.name,b,inst))
                              print "done"
                              continue
                         while redo:
                              if itercount > 1:
                                   print "redoing fit that was subpar"
                              #pars0 = myrobustpolyfit(x,y,deg,1.0/lc['dmag'][indx])
                              pars = np.polynomial.polynomial.polyfit(
                                   x,y,deg=deg, w=1.0/yerr)                        
                              polysol=np.poly1d(np.array(pars[::-1]))

                              ######single band figure
                              myfig=pl.figure(figcounter)                          
                              myplot_err(xp,polysol(xp))
                              thissn.plotsn(photometry=True,band=b,fig=figcounter,show=True)
                    
                              #all = optimize.fmin_powell(sumsqres, pars, args=(x,y,yerr, 'poly',b), full_output=1)
                              all = optimize.curve_fit(mypolynomial, x,y , p0=pars)#, sigma=1.0/lc['dmag'][indx])
                              #all = optimize.leastsq(residuals, pars, args=(x,y,lc['dmag'][indx], 'poly',b), full_output=1, epsfcn=0.00001,  ftol=1.49012e-38)
                              lsq,success=all[0],all[1]#/[0.1,10.0,1000.,10000]
                              dev = y-polysol(x)
                              rchisq = sum((dev**2)/yerr)/(len(x)-deg-2)
                              solution.append({'sol':polysol,'deg':deg, 'pars':pars})
                              rchis.append(rchisq)
                              print "rchisq, deg, iteration count:",rchisq, deg, itercount 
                              if rchisq < 0.2 and deg > 3 and itercount < 3: 
                                   deg = deg-1
                                   itercount=itercount+1
                                   thissn.stats[b].flagmissmax,thissn.stats[b].flagmiss15=0,0
                                   pl.figure(figcounter)
                                   pl.clf()
                                   continue
                              if rchisq >1.6 and deg < 6 and itercount < 6: 
                                   deg = deg+1
                                   itercount=itercount+1
                                   pl.figure(figcounter)
                                   pl.clf()
                                   thissn.stats[b].flagmissmax,thissn.stats[b].flagmiss15=0,0
                                   continue

                              redo=0

                              sol=np.where(abs(np.array(rchis)-1 ) == np.min( abs(np.array(rchis)-1)))[0]
                              if len(sol)>1: 
                                   sol = sol[0]
                                   print "WARNING: some fits are equivalent. very suspicious"
                              thissn.solution[b] = solution[sol]
                              thissn.stats[b].polyrchisq = rchis[sol]
                              thissn.stats[b].polydeg=thissn.solution[b]['deg']
                              thissn.stats[b].polyresid = np.array(dev)
                              thissn.getstats(b)
                              thissn.stats[b].printstats()

                              lsq,success=all[0],all[1]#/[0.1,10.0,1000.,10000]
                              thissn.polysol[b]=np.poly1d(np.array(lsq[::-1]))
                              dev = y-thissn.polysol[b](x)
                              mad,chisq = np.median(np.abs(dev))/0.6745,sum((dev**2)/yerr)
                              rchisq=chisq/(len(x)-deg-2)
                              rchis.append(thissn.stats[b].polyrchisq)
                              
                              pl.figure(0)
                              thissn.plotsn(photometry=True,band=b,fig=0,symbol='%so'%su.mycolors[b])
                              
                              myfig=pl.figure(figcounter)
                              thissn.plotsn(photometry=True,band=b, fig=figcounter, show=showme, verbose=False, save=False, symbol='',title='%s band: %s instrument: %s'%(thissn.name,b,inst), Vmax=thissn.Vmax, plotpoly=True)
                              #                        myplot_err(xp,thissn.solution[b]['sol'](xp),symbol='k-')

                              if  thissn.stats[b].flagmiss15 == 0 and thissn.stats[b].flagmissmax == 0:
                             
                                   thissn.stats[b].dm15lin= np.mean(thissn.stats[b].maxjd[1])-np.mean(thissn.stats[b].m15data[1])
                                   #                            print dm15lin
                              else:
                                   thissn.stats[b].dm15lin = -1000
                
                              if thissn.printlog(b,inst,logoutput) == -1:
                                   continue
                        
                              myplot_txtcolumn(0.6,0.8,0.05,['mjd: %.2f'%(thissn.stats[b].maxjd[0]+2453000.0), 'dm15 - fit: %.2f'%(thissn.stats[b].dm15),  'dm15 - linear: %.2f'%(thissn.stats[b].dm15lin), 'median deviation: %.2f'%(mad),'chi-square: %.2f'%(thissn.stats[b].polyrchisq), 'degrees of poly fit: %.2f'%(thissn.solution[b]['deg'])],myfig)
                              myplotarrow(thissn.stats[b].maxjd[0], thissn.stats[b].maxjd[1]-0.8,"maxjd")
                              myplotarrow(thissn.stats[b].maxjd[0]+15, thissn.stats[b].maxjd[1]-0.8,"maxjd+15")
                              myfig.savefig("%s.%s.%s.png"%(thissn.name,b,inst))
                              
                              pl.figure(0)
                              myplot_setlabel(xlabel='JD - 2453000.00',ylabel='Mag',title='%s '%(thissn.name))
                              myplot_err(x,y,symbol='%s%s'%(su.mycolors[b],su.myshapes[b]))
                              leg,=myplot_err(xp,thissn.solution[b]['sol'](xp),xlim=(min(xp)-10,max(xp)+10),ylim=(minyall,maxyall),symbol='%s-'%su.mycolors[b])


                         mylegslist.append(leg)
                         myfig=pl.figure(0)
                         figcounter+=1                

                         ################################fitting with template
                         if not TEMPLATEFIT:
                              continue

                              #    pl.plot (template.x,template.tempfuncy()(template.x))
                              #    pl.plot (template.x,template.tempfuncstd()(template.x))
                              #    pl.show()
                              #    pkl_file.close()
                              #    sys.exit()
                         if not b in ['V','R']:
                              print "templates only available in V and R for now"
                              continue

                         if UPDATETEMPLATE:
                              pl.figure(10000)
                              pl.ylim(3,-0.5)          
                              pl.xlim(-5,40)          

                         if lccount == 0 or not UPDATETEMPLATE:
                              sne,templates=loadlitlist(b)
                              sne=splinetemplates(sne)
                              if sne==-1: 
                                   print "all failed. continuing..."
                                   continue
                              if lccount==0:
                                   thistemplate=Mytempclass()
                                   thistemplate.loadtemplate(b, x=sne[0].new_x,mean=smoothListGaussian(spstats.stats.nanmean([s.new_y for s in sne],axis=0)),median=smoothListGaussian(spstats.stats.nanmedian([s.new_y for s in sne],axis=0)),std=smoothListGaussian(spstats.stats.nanstd([s.new_y for s in sne],axis=0)))
                                   pl.plot ( thistemplate.template[b].x,thistemplate.template[b].median,'k-',alpha=1.0)
                              


                         fullxrange=np.arange(thissn.Vmax-2400000.0-10.,thissn.Vmax-2400000.0+40.0,0.1)
                         myfig=pl.figure(figcounter+10)
                         myplot_setlabel(xlabel='JD - 2453000.00',ylabel= 'Mag',title='%s band: %s instrument: %s'%(thissn.name,b,inst))
                         myplot_err(x,y,yerr=None,xlim=None,ylim=(minyall+1,maxyall-1), symbol='ro')
                         myplot_err(x,y, yerr=yerr, symbol='%so'%su.mycolors[b])
                         
                         templspline = interp1d(x,y, kind='linear', bounds_error=False)
                         temp=templspline(np.arange(min(x),max(x), 0.1))
                         #                    print min(x),max(x), templspline(min(x)+1),templspline(max(x)-1)                    
                         pars = (1.0,np.arange(min(x),max(x), 0.1)[np.where(temp == np.nanmin(temp))[0][0]], np.nanmin(temp))#,0.5)
                         print "\n\n\n#############################"
                         print "starting minimization with template"
                         
                         try:
                              bounds = [(None,None),(pars[1]-5,pars[1]+5),(None,None)] 
                              all=leastsqbound(residuals, pars, args=(x,y,yerr, 'template',b,thistemplate), bounds= bounds,full_output=1, epsfcn=0.00001,  ftol=1.49012e-1)#3)#)8)
                              #                    all = optimize.fmin_powell(sumsqres, pars, args=(x,y,lc['dmag'][indx], 'template'), ftol=0.1, full_output=1)
                              #                    all = optimize.fmin(sumsqres, pars, args=(x,y,yerr, 'template',b), ftol=0.01, full_output=1, maxiter=100)
                              #                    all = optimize.leastsq(residuals, pars, args=(x,y,lc['dmag'][indx], 'template',b), full_output=1, epsfcn=0.00001,  ftol=1.49012e-38)
                              pars = all[0]
                              print "parameters at minimization: ",pars
                              
                              #                    print fullxrange-pars[1], pars
                              #                    pl.ylim(min(mytemplate(fullxrange,pars)),max(mytemplate(fullxrange,pars)))
                         except:
                              pass

                         print thissn.stats[b].templatefit
                         thissn.stats[b].templatefit['stretch']=pars[0]
                         thissn.stats[b].templatefit['xoffset']=pars[1]
                         thissn.stats[b].templatefit['xstretch']=pars[0]
                         thissn.stats[b].templatefit['yoffset']=pars[2]
                         
                         thissn.templsol[b]=mytemplate
                         myplot_err(fullxrange,thissn.templsol[b](fullxrange,pars,b), symbol='k--') 
                         if options.active:
                              pl.draw()
                              input_var = raw_input("is the fit ok? (y/n) ")
                              if input_var.lower().startswith('n') :
                                   continue

                         dev = y-thissn.templsol[b](x,pars,b)
                         mad,chisq = np.median(np.abs(dev))/0.6745,sum((dev**2)/yerr)                    
                         thissn.stats[b].templrchisq=chisq/(len(x)-len(pars)-2)
                         thissn.stats[b].templresid=np.array(dev)
                         templatefigs.append((figcounter+10,b,inst, np.nanmax(thissn.templsol[b](fullxrange,pars,b)),np.nanmin(thissn.templsol[b](fullxrange,pars,b)),pars))

                         #the next line is so that if i passed only 3 pars the print wont crush (and the fourth par is the same as the first then: stretch=luminosity)
                         pars=np.append(pars,pars[0])              
                         print >>templog[su.bandsindex[b]], thissn.name, thissn.sntype, pars, time.strftime("%a, %d %b %Y %H:%M:%S +0000",  time.localtime(time.time()))
                         #                    templog.flush()
                         #                    all = optimize.fmin_powell(sumsqres, pars, args=(x,y,lc['dmag'][indx], 'template',b), full_output=1)
                         #, epsfcn=0.000001,  ftol=1.49012e-18)
                         
                         
                         print thissn.stats[b].templatefit
                         pl.figure(10000)
                         #                    for s in sne:         
                         #                         pl.errorbar(s['phot'][0]-s['mjdmax'],s['normphot'],s['phot'][2],fmt='ko')
                         #                        pl.plot(s['new_x'],s['new_y'], 'k-')
                         #                  fullxrange=np.arange(-10,40,0.01)
                         #                    pl.plot (fullxrange,reversetemplate(fullxrange,thissn.stats[b].templatefit,templspline, thissn.Vmax-2400000.0),'.')
                         thissntemplate=templatesn(thissn.name)
                         thissntemplate.sntype=thissn.sntype
                         thissntemplate.mjdmax=thissn.Vmax-2400000.0
                         thissntemplate.new_y=reversetemplate(thissntemplate.new_x,thissn.stats[b].templatefit,templspline, thissn.Vmax-2400000.0)
                         thissntemplate.yfunc=templspline
                    
                         #                    thistemplate=Mytempclass()
                         sne.append(thissntemplate)

                         thistemplate.loadtemplate(b, x=sne[0].new_x,mean=smoothListGaussian(spstats.stats.nanmean([s.new_y for s in sne],axis=0)),median=smoothListGaussian(spstats.stats.nanmedian([s.new_y for s in sne],axis=0)),std=smoothListGaussian(spstats.stats.nanstd([s.new_y for s in sne],axis=0)))
                    
                         print len(sne)
                         #                    pl.plot ( thistemplate.template[b].x,thistemplate.template[b].mean+3,'k-', alpha=myalpha)
                         print "myalpha: ",myalpha
                         if lccount>1:
                              pl.plot ( thistemplate.template[b].x,thistemplate.template[b].median,'g-',alpha=myalpha)
                         myalpha=max(myalpha+dalpha, 1.0)
                         # pl.plot ( thistemplate.template[b].x,thistemplate.template[b].mean-thistemplate.template[b].std+3,'b-',linewidth=1)
                         # pl.plot ( thistemplate.template[b].x,thistemplate.template[b].mean+thistemplate.template[b].std+3,'b-',linewidth=1)
                         lccount = lccount+1
                    else:
                         print "no datapoints for filter and instrument selection"
                         continue
          if UPDATETEMPLATE and b in ['V','R']:
               pl.figure(10000)
               if options.sntype and lccount>1:
                    mytemplatefile=open("tmp%d/mytemplate"%dirindex+b+"_"+options.sntype+".dat",'w')
                    print >> mytemplatefile, "#phase, mean, median, stdev"
                    for i,x in enumerate(thistemplate.template[b].x):
                         print >> mytemplatefile, x,thistemplate.template[b].mean[i],thistemplate.template[b].median[i],thistemplate.template[b].std[i]
                    myplot_setlabel(xlabel='phase',ylabel='normalized magnitude',title='%s (%d sne)'%(options.sntype, lccount))
                    pl.xlim(-5,40)          
                    pl.savefig("tmp%d/"%dirindex+options.sntype+"%04d.png"%lccount)
             
               elif lccount>1:
                    pl.savefig("tmp.png")

        
          myfig=pl.figure(0)
          if thissn.stats[b].success==0:
             continue
          myplotarrow(float(thissn.Vmax)-2400000,pl.ylim()[1]+0.5,label="V max")
        
          print >> snlog, f
          print >> snlog, "number of bands", sum([thissn.stats[b].success for b in su.bands if thissn.stats[b].flagbadfit == 0 ])
          print >> snlog, "bands available:"
          print >> snlog, "\t",
          mybands,mymjds,mydm15s,mydegs,mychisqs,mychisqs,mybandstring,mytchisqs=\
                                                                                  '','','','','','','',''
          mybandslist=[]
          for b in su.bands:
             if thissn.stats[b].flagbadfit > 0 or thissn.stats[b].success==0:
                  continue
             mybandslist.append(b)
             mybandstring+=b+" "
             mymjds+="%.2f"%(thissn.stats[b].maxjd[0])+" "#-2453000.0)+" " 
             mydm15s+="%.2f"%thissn.stats[b].dm15+" "
             mychisqs+="%.2f"%thissn.stats[b].polyrchisq+" "
             mydegs+="%d"%thissn.stats[b].polydeg+" "
             mytchisqs+="%.2f"%thissn.stats[b].templrchisq+" "
             
             print >>snlog, b+"-"+inst," ",
             print >>snlog, ""
             print >> snlog, "\t", b,inst, "\t datapoints: ",thissn.filters[b], "\t flags: ",thissn.stats[b].flagmissmax," ",thissn.stats[b].flagmiss15," ",thissn.stats[b].flagbadfit   ,"\t chisq: ",thissn.stats[b].polyrchisq, 


          thissn.printsn(template=TEMPLATEFIT, extended=True)

          print >>snlog ,""
          
          myplot_txtcolumn(0.35,0.85,0.05,['mjds: %s'%mymjds],myfig)
          myplot_txtcolumn(0.5,0.80,0.05,['dm15s: %s'%mydm15s,'reduced chi-squares: %s' %mychisqs,'degrees of poly fits: %s'%(mydegs), "template rchisq %s"%(mytchisqs)],myfig)
          pl.legend(mylegslist,mybandslist, loc=3, ncol=1,prop={'size':8})  
          myfig.savefig("%s.png"%(thissn.name))
          
        
          for i in templatefigs:
             myfig=pl.figure(i[0])
             pl.ylim(i[-3],i[-2])
             myplot_txtcolumn(0.5,0.85,0.05,['mjds: %s'%mymjds,'dm15s: %s'%mydm15s,'chi-squares: %s' %mychisqs,'parameters: %2.2f %2.2f %2.2f %2.2f'%(i[-1][0],i[-1][1],i[-1][2],i[-1][0])],myfig)             
             myfig.savefig("%s.%s.%s_template.png"%(thissn.name,i[1],i[2]))

#    pl.show()
    if not splinefit:
         pklfile = open(thissn.name+'.pkl', 'wb')
         pkl.dump(thissn,pklfile)    

                                  
