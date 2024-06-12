import glob 
import os
import pylab as pl
import sys
import pickle as pkl
import pandas as pd
import numpy as np
import traceback
import seaborn as sns

#s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
#pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import matplotlib as mpl
mpl.use('agg')

avoid=["03dh"]

pl.rcParams['figure.figsize']=(10,10)
pl.rcParams['figure.figsize']=(10,10)
pl.rcParams['font.family'] = 'serif'
pl.rcParams['font.serif'] = ['Times New Roman'] + pl.rcParams['font.serif']



# # Loading CfA SN lightcurves

#setting parameters for lcv reader
#use literature data (if False only CfA data)
LIT = True
#use NIR data too
FNIR = True

SNTYPE = 'Ic-bl'

#pl.ion()
readgood = pd.read_csv("goodGPs.csv", header=None)
# print('readgood is ' ,readgood)
# meansmooth = lambda x : np.zeros_like(x)
#print readgood
#sys.exit()


DEBUG = False

tcorlims = {
     'R':{'tmin':10, 'tmax':20},
     'V':{'tmin':10, 'tmax':20},     
     'r':{'tmin':10, 'tmax':20},
     'g':{'tmin':10, 'tmax':20},                    
     'U':{'tmin':20, 'tmax':0},
     'u':{'tmin':20, 'tmax':0},                     
     'J':{'tmin':20, 'tmax':10},
     'B':{'tmin':20, 'tmax':5},
     'H':{'tmin':15, 'tmax':20},
     'I':{'tmin':10, 'tmax':20},
     'i':{'tmin':10, 'tmax':20},                    
     'K':{'tmin':20, 'tmax':20},
     'm2':{'tmin':20, 'tmax':20},
     'w1':{'tmin':20, 'tmax':20},
     'w2':{'tmin':20, 'tmax':20}}                    
                    
bands2 = ['U','B','V', 'g', 'R', 'I', 'rp','ip','up','J','H','K','m2','w1','w2']

if __name__ == '__main__':

     #uncomment for all lcvs to be read in
     allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                          "/SESNessentials.csv")['SNname'].values
     #set a plotcolor for each SN by assigning each to a number 0-1
     snecolors = {}


     for i,sn in enumerate(allsne):
          snecolors[sn] = i * 1.0 / (len(allsne) - 1)

     NUM_COLORS = len(allsne)
     LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
     NUM_STYLES = len(LINE_STYLES)

     sns.reset_orig()  # get default matplotlib styles back
     clrs = sns.color_palette('Spectral', n_colors=NUM_COLORS)

     # if len(sys.argv) > 1:
     #      if sys.argv[1] in ['Ib','IIb','Ic','Ic-bl', 'Ib-c', 'Ibn']:
     #           SNTYPE = sys.argv[1]
     #      else:
     #           allsne = [sys.argv[1]]

     SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']
     
     #set up SESNCfalib stuff
     su = templutils.setupvars()
     printtable = False

     if len(sys.argv) > 2:
          bands = [sys.argv[2]]
     else:
          bands = su.bands
          keys = bands
          printtable = True
     nbands = len(bands)

     #errorbarInflate = {"93J":30, 
     #                   "05mf":1}


     # SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']
     # bands = ['U','B','V', 'g', 'R', 'I', 'rp','ip','up','J','H','K','m2','w1','w2']
     colorTypes = {'IIb':'FireBrick',
                  'Ib':'SteelBlue',
                  'Ic':'DarkGreen',
                  'Ic-bl':'DarkOrange',
                  'Ibn':'purple'}
     outliers = {}
     counter = {}
     for SNTYPE in SNTYPES:
          outliers[SNTYPE] = {}
          counter[SNTYPE] = dict(zip(keys, [0]*len(keys)))

          tmpl = {}
          for bb in bands2:

               tmpl[bb] = {}



               tmpl[bb][SNTYPE] = {}



               path = "outputs/GPalltemplfit_%s_%s_V0.pkl"%(SNTYPE,bb)
               tmpl_ = pkl.load(open(path, "rb"))

               #         print(tmpl_['rollingMedian'])



               if np.nansum(tmpl_['rollingMedian']) == 0:
                 # print(bb, SNTYPE)
                 continue
               tmpl[bb][SNTYPE] = tmpl_


          axs_com2 = {}

          figs_com2 = []


          for j,b in enumerate(bands):
               outliers[SNTYPE][b] = []

               fv4, axs_com2[b] = pl.subplots(2,1,sharex=True, figsize=(22,20))
               fv4.subplots_adjust(hspace=0.05)
               figs_com2.append(fv4)

          #pl.ion()
          dt = 0.5
          t = np.arange(-15,50,dt)

          #set up arrays to host mean, mean shifted by peak, standard dev, and dtandard dev shifted
          mus = np.zeros((len(allsne), len(bands), len(t))) * np.nan
          musShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan
          stds = np.zeros((len(allsne), len(bands), len(t))) * np.nan
          stdsShifted = np.zeros((len(allsne), len(bands), len(t))) * np.nan
          
          c = 0


          for i, sn in enumerate(allsne):

               # read and set up SN and look for photometry files
               try:
                    thissn = snstuff.mysn(sn, addlit=True)
               except AttributeError:
                    print('1 removed sn: ', sn)
                    continue
               if len(thissn.optfiles) + len(thissn.fnir) == 0:
                    print ("bad sn")
               # read metadata for SN
               thissn.readinfofileall(verbose=False, earliest=False, loose=True)
               #thissn.printsn()
               if not thissn.type == SNTYPE:
                    continue
               if thissn.snnameshort in avoid:
                    continue
               # check SN is ok and load data
               if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
                    print ("bad sn")
               print (" starting loading ")
               print (os.environ['SESNPATH'] + "/finalphot/*" + \
                      thissn.snnameshort.upper() + ".*[cf]")
               print (os.environ['SESNPATH'] + "/finalphot/*" + \
                      thissn.snnameshort.lower() + ".*[cf]")

               print( glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                                thissn.snnameshort.upper() + ".*[cf]") + \
                      glob.glob(os.environ['SESNPATH'] + "/finalphot/*" + \
                                thissn.snnameshort.lower() + ".*[cf]") )

               lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
               thissn.setphot()
               thissn.getphot()
               thissn.setphase()
               thissn.sortlc()
               thissn.printsn()



               # see if SN is in the list of good SNe with ok GP fits
               goodbad = readgood[readgood[0] == thissn.name]

               # print('goodbad is ', goodbad)

               if len(goodbad)==0:
                    if DEBUG:
                         # raw_input()
                         print('Removed because gp fit is not ok:', sn)
                    continue

               #check that it is k
               if np.array([n for n in thissn.filters.values()]).sum() == 0:
                    print ("bad sn")
                    continue

               # c = 0

               for j,b in enumerate(bands):

                    # print(b)


                    if b == 'i':
                       bb = 'ip'
                    elif b == 'u':
                       bb = 'up'
                    elif b == 'r':
                       bb = 'rp'
                    else:
                         bb = b

                    tcore = (t>-tcorlims[b]['tmin']) * (t<tcorlims[b]['tmax'])

                    if DEBUG:
                         print (goodbad)
                    if len(goodbad[goodbad[1]==bb]) == 0:
                         if DEBUG:
                              print ("no",b)
                              # raw_input()
                              print('2 Removed sn', sn)
                         continue

                    if goodbad[goodbad[1]==bb][2].values[0] == 'n':
                         if DEBUG:
                              print (goodbad[goodbad[1]==bb][2].values[0])
                              print ("no good",b)
                              # raw_input()
                              print('3 Removed sn', sn)
                         continue
                    if thissn.filters[b] == 0:
                         if DEBUG:
                              print ("no band ",b)
                              # raw_input()
                              print('4 Removed sn', sn)
                         continue


                    xmin = thissn.photometry[b]['mjd'].min()
                    # x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5

                    if xmin - thissn.Vmax < -1000:
                     x = thissn.photometry[b]['mjd'] - thissn.Vmax + 2400000.5

                    elif xmin - thissn.Vmax > 1000:
                     x = thissn.photometry[b]['mjd'] - thissn.Vmax - 2400000.5
                    else:
                     x = thissn.photometry[b]['mjd'] - thissn.Vmax


                    y = thissn.photometry[b]['mag']
                    photmin = y.min()

                    y = photmin - y

                    yerr = thissn.photometry[b]['dmag']

                    # print(x)

                    if b == 'g':
                         low_lim = -20
                         up_lim = 50
                    elif b == 'I':
                         low_lim = -20
                         up_lim = 100
                    elif b == 'i':
                         low_lim = -20
                         up_lim = 100

                    elif b == 'U':
                         low_lim = -15
                         up_lim = 40

                    elif b == 'u':
                         low_lim = -25
                         up_lim = 30
                    else:
                         low_lim = -20
                         up_lim = 100


                    y = y[np.where(np.array((x<up_lim)&(x>low_lim)))[0]]
                    yerr = yerr[np.where(np.array((x<up_lim)&(x>low_lim)))[0]]
                    x = x[np.where(np.array((x<up_lim)&(x>low_lim)))[0]]

                    # y = y[np.where(np.array((x<100)&(x>-20)))[0]]
                    # yerr = yerr[np.where(np.array((x<100)&(x>-20)))[0]]
                    # x = x[np.where(np.array((x<100)&(x>-20)))[0]]

                    # print(b,x)

                    t_sn = t[(t >= x.min()) * (t <= x.max())]

                    # print('We are printing x and y',x,y)



                    pklf = "outputs/GPfit%s_%s.pkl"%(sn,b + 'p' if b in ['u', 'r', 'i']
                                                         else b)
                    if not os.path.isfile(pklf):
                         print ("missing file ", pklf)
                         # raw_input()
                         print('5 Removed sn', sn)
                         continue

                    ygp, gp, tmplm = pkl.load(open(pklf, "rb"))
                    meansmooth = lambda x : np.zeros_like(x)#-tmplm(x) + tmplm(0)
                    try:
                         mu, cov = gp.predict(y + meansmooth(x), np.log(t_sn+30))
                    except ValueError:

                         if DEBUG:
                              traceback.print_exc()
                              # print ("error")
                              # raw_input()
                              print('6 Removed sn', sn)
                         continue
                    # print(mu)


                    std = np.sqrt(np.diag(cov))
                    if (np.abs(mu)<0.1).all():
                         continue

                    ind_start_mu = np.where(t == t_sn.min())[0][0]
                    ind_end_mu = np.where(t == t_sn.max())[0][0]

                    # print(ind_start_mu, ind_end_mu)

                    # print(len(mus[i][j][ind_start_mu:ind_end_mu+1]), len(mu - mu[np.abs(t_sn)==np.abs(t_sn).min()]))

                    mus[i][j][ind_start_mu:ind_end_mu+1] = mu - mu[np.abs(t_sn)==np.abs(t_sn).min()]

                    stds[i][j][ind_start_mu:ind_end_mu+1]  = std

                    # print(len(t), len(mus[i][j]), len(std), len(meansmooth(t)))


                    if (sum(~np.isnan(np.array(mus[i][j] + meansmooth(t))[tcore])))<4:
                     continue
                    # print((sum(~np.isnan(np.array(mus[i][j] + meansmooth(t))))))
                    # print((np.array(mus[i][j] + meansmooth(t))[tcore]))
                    truemax = np.where(np.array(mus[i][j] + meansmooth(t)) ==
                                       np.nanmax(np.array(mus[i][j] + meansmooth(t))[tcore]) )[0][0]

                    if truemax < 5:
                         minloc = np.where(np.array(mus[i][j] + meansmooth(t )) == np.nanmin(np.array(mus[i][j] + meansmooth(t ))[tcore]))[0][0]
                         if minloc > 0 and minloc <len(tcore):
                              tcore = (t >-10) * (t <tcorlims[b]['tmax'])
                              # print ((~np.isnan(np.array(mus[i][j] + meansmooth(t ))[tcore])))
                              if not (sum(~np.isnan(np.array(mus[i][j] + meansmooth(t ))[tcore]))) == 0:
                               truemax = np.where(np.array(mus[i][j] + meansmooth(t)) ==
                                         np.nanmax(np.array(mus[i][j] + meansmooth(t))[tcore]))[0][0]

                         if np.abs(truemax - np.where(t == t[tcore][0])[0][0]) < 2:
                              truemax = np.where(t == 0)[0][0]

                    t2 = t - t[truemax]
                    t2_sn = t_sn - t[truemax]
                    t20 = np.where(t2==0)[0][0]

                    yoffset = (mus[i][j] + meansmooth(t))[t20]


                    tmin, tmax = t2.min(), t2.max()


                    # After shifiting the mus, we remove the parts of the light curve out of the original interval
                    # And we put the empty part of the light curve within that interval equal to nan.


                    if (thissn.snnameshort == '16hgs' and b == 'g') or\
                       (thissn.snnameshort == '13ge' and b == 'U') or\
                       (thissn.snnameshort == '13cq' and b == 'I') or\
                       (thissn.snnameshort == '06aj' and b == 'U') or\
                       (thissn.snnameshort == '06aj' and b == 'B') or\
                       (thissn.snnameshort == '13cq' and b == 'V') :
                       print('exceptions are: ',thissn.snnameshort)
                       yoffset = 0
                       musShifted[i][j] = (mus[i][j] + meansmooth(t))- yoffset
                       stdsShifted[i][j] = stds[i][j]


                    else:

                         if t.min()>tmin:
                          # print(t_sn.max(), t[truemax])
                          ind_max = np.where(t == t.max()-t[truemax])[0][0]
                          musShifted[i][j][:ind_max] = (mus[i][j] + meansmooth(t))[-ind_max:] - yoffset
                          stdsShifted[i][j][:ind_max] = stds[i][j][-ind_max:]

                         elif t.min()<tmin:
                          ind_min = np.where(t == t.min()-t[truemax])[0][0]
                          musShifted[i][j][ind_min:] = (mus[i][j] + meansmooth(t))[:-ind_min]- yoffset
                          stdsShifted[i][j][ind_min:] = stds[i][j][:-ind_min]

                         else:

                          musShifted[i][j] = (mus[i][j] + meansmooth(t))- yoffset
                          stdsShifted[i][j] = stds[i][j]

                    # Plot individual SESNe

                    counter[SNTYPE][b] = counter[SNTYPE][b] + 1

                    axs_com2[b][0].plot(t, musShifted[i][j] + meansmooth(t), lw=5,
                                        label=thissn.snnameshort, color=clrs[i], linestyle=LINE_STYLES[i % NUM_STYLES])
                    #Detect outliers
                    try:
                         t_sn = t
                         y_sn = musShifted[i][j] + meansmooth(t)
                         t_tmpl = tmpl[bb][SNTYPE]['t']
                         if max(t_sn)<max(t_tmpl):
                              max_t = max(t_sn)
                         else:
                              max_t = max(t_tmpl)

                         if min(t_sn)>min(t_tmpl):
                              min_t = min(t_sn)
                         else:
                              min_t = min(t_tmpl)

                         y_sn_limited = y_sn[(t<max_t) & (t>min_t)]
                         y_tmpl_limited_low = tmpl[bb][SNTYPE]['rollingPc25'][(t_tmpl<max_t) & (t_tmpl>min_t)]
                         y_tmpl_limited_up = tmpl[bb][SNTYPE]['rollingPc75'][(t_tmpl < max_t) & (t_tmpl > min_t)]
                         dp_out = len(y_sn_limited[(y_sn_limited>y_tmpl_limited_up) | (y_sn_limited<y_tmpl_limited_low)])
                         frac = dp_out/len(y_sn_limited)
                         if frac>0.5:
                              outliers[SNTYPE][b].append(thissn.snnameshort)
                              axs_com2[b][1].plot(t, musShifted[i][j] + meansmooth(t),  lw=5,
                                            label=thissn.snnameshort,color =  clrs[i],linestyle = LINE_STYLES[i%NUM_STYLES])
                    except:
                         pass


          #Plot the templates
          for j, b in enumerate(bands):

               if b == 'i':
                    bb = 'ip'
               elif b == 'u':
                    bb = 'up'
               elif b == 'r':
                    bb = 'rp'
               else:
                    bb = b

               try:
                    axs_com2[b][0].plot(tmpl[bb][SNTYPE]['t'], tmpl[bb][SNTYPE]['rollingMedian'], 'k',linewidth = 5, label = 'Rolling Median', zorder=10)
                    axs_com2[b][0].fill_between(tmpl[bb][SNTYPE]['t'],
                                          tmpl[bb][SNTYPE]['rollingPc25'],
                                          tmpl[bb][SNTYPE]['rollingPc75'],
                                          color= 'black', alpha=0.3,
                                          label="75 Percentile", zorder=9)
                    axs_com2[b][1].plot(tmpl[bb][SNTYPE]['t'], tmpl[bb][SNTYPE]['rollingMedian'], 'k', linewidth=5,
                                     label='Rolling Median', zorder=10)
                    axs_com2[b][1].fill_between(tmpl[bb][SNTYPE]['t'],
                                             tmpl[bb][SNTYPE]['rollingPc25'],
                                             tmpl[bb][SNTYPE]['rollingPc75'],
                                             color='black', alpha=0.3,
                                             label="75 Percentile", zorder=9)
               except:
                    pass

               axs_com2[b][0].set_title(SNTYPE + ", " + b, size=45)
               axs_com2[b][1].set_xlabel("phase (days)", size=30)
               # axs_com2[b].set_ylabel("mag", size = 30)
               axs_com2[b][0].legend(loc = 'lower left' ,ncol=4, prop={'size':20})
               axs_com2[b][1].legend(loc = 'lower left' ,ncol=4, prop={'size': 20})


               axs_com2[b][0].set_xlim(-25,55)
               axs_com2[b][0].grid(True)
               axs_com2[b][1].grid(True)

               axs_com2[b][0].tick_params(axis="y", direction="in", which="major", \
                            right=True, size=7, labelsize=55, width=2)
               axs_com2[b][1].tick_params(axis="both", direction="in", which="major", \
                                          right=True, top=True, size=7, labelsize=55, width=2)

               figs_com2[j].text(0.04, 0.5, 'Relative Magnitude', va='center', rotation='vertical', size=30)

               figs_com2[j].savefig("outputs/gp_templates/GPMedtemplfit_%s_%s_outlier2.pdf"%(SNTYPE,b + 'p' if b in ['u', 'r', 'i']
                                                         else b),  bbox_inches='tight')
               print(SNTYPE, b, outliers[SNTYPE][b])

     # df = pd.DataFrame({})
     df = pd.DataFrame.from_dict(counter, orient='index').reset_index(drop=True)
     df = df.rename(index = {0:'Ib',1:'IIb',2:'Ic',3:'Ic-bl', 4:'Ibn'})
     df.to_csv("outputs/counter_SN_lightcurves_per_band_per_subtype.csv")
     # print(b , outliers)



