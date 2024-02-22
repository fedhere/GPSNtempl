#This version was modified by Somayeh 1/13/2022 to make the table with SNR more compact, 
#adding one column of N, [min, max], SNR for each band

import sys
import os
import pandas as pd
import matplotlib as mpl
import pylab as pl
import numpy as np
import json
import datetime 
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import datetime
import traceback
import seaborn as sns


# try:
os.environ['SESNPATH']
os.environ['SESNCFAlib']
os.environ['DB']
# except KeyError:
#      print ("must set environmental variable SESNPATH and SESNCfAlib")
#      sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib")+"/templates"
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
import readinfofile as ri

today_year = int(str(datetime.date.today().year)[-2:])

su=setupvars()



s = json.load(open("./../fbb_matplotlibrc.json", "r"))
mpl.rcParams.update(s)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    "SteelBlue",
    "IndianRed",
    "Green",
    "DarkOrange",
    "MediumTurquoise",
    "gold",
    "#E24A33"
    ])
mpl.rcParams.update({"axes.formatter.limits" : (-7, 7),
                     'xtick.direction' : 'in',
                     'ytick.direction' : 'in'
                     })
mpl.rcParams.update({"legend.frameon" : True,
                     "legend.fancybox" : True,
                     "legend.framealpha" : 0.5,
                     "axes.facecolor" : "white",
                     "axes.edgecolor": "#bcbcbc"})
#,
#                     'xtick.top' : False,
#                     'xtick.bottom' : False,
#                     'ytick.left' : False,
#                     'ytick.right' : False})
mpl.rcParams.update({'font.size': 21,
                     'axes.titlesize': 21,
                     'xtick.labelsize': 21,
                     'ytick.labelsize': 21
                     })
mpl.rcParams['date.autoformatter.year'] = '%Y'
s = json.load(open("./../fbb_matplotlibrc.json", "r"))
mpl.rcParams.update(s)
pl.rcParams.update({'axes.grid':False})

### data munging
#snname = pd.read_csv("AllPhotOptTable_z.tex", sep="&", header=None)[0]
#snname.head()

# Fix the discovery dates to have four digits instead of two to avoid confusing the decades.
def fixdate (x):
    if int(x.split('/')[-1]) <= today_year:
        return "/".join(x.split("/")[:-1])+"/20"+x.split("/")[-1]
    else:  
        return "/".join(x.split("/")[:-1])+"/19"+x.split("/")[-1]

def read_allSNeOpenSNCat(dropnophot = True):

    # SESNe_1_phot.csv: SESNe with at least one photometric data in any bands downloaded 
    # on Aug 27, 2021.
    allOpenSNe = pd.read_csv("SESNe_1_phot.csv")
    print ("initial size", allOpenSNe.shape)

    print (allOpenSNe[allOpenSNe.duplicated(subset=["Dec.","R.A."])])
    #print unique subtypes
    #print(allOpenSNe.Type.drop_duplicates())
    
    ##drop duplicated by RA and Dec
    #removed cause there are none that are not NaN
    #allOpenSNe = allOpenSNe.drop_duplicates(subset=["Dec.","R.A."])

    #drop CasA
    allOpenSNe = allOpenSNe.drop(allOpenSNe[[nm == "SN1667A" for nm in allOpenSNe.Name.values]].index)    

    print ("initial size without duplicates and CasA", allOpenSNe.shape)

    # remove SLSN
    nonSLSNOpenSNe = allOpenSNe.drop(allOpenSNe[["SLSN" in snt for snt in allOpenSNe.Type.values]].index)
    
    # remove Blazar, Variable, Microlensing, BL
    notSN = ["Blazar", "Variable", "Microlensing", "BL"]
    nonSLSNOpenSNe = nonSLSNOpenSNe.drop(nonSLSNOpenSNe[[snt in notSN for snt in nonSLSNOpenSNe.Type.values]].index)

    print ("after removing wrong types", nonSLSNOpenSNe.shape)
    
    #nonSLSNOpenSNe.dropna(inplace=True)
    print(nonSLSNOpenSNe.Type.drop_duplicates())
    print ('All SESNe with modified types and no limit on phot and spectra: ',nonSLSNOpenSNe.shape)

    # select SNe w some photometry
    if dropnophot:
        nonSLSNOpenSNe.drop((nonSLSNOpenSNe[nonSLSNOpenSNe["Phot."]<=0]).index, inplace=True)
    print ("All SESNe with at least 1 photometric point", nonSLSNOpenSNe.shape)
    
    # manipulate discovery date
    try:
        nonSLSNOpenSNe["Disc. Date"] = [fixdate(x) for x in nonSLSNOpenSNe["Disc. Date"].values]
        print('Dates fixed....')
    except:
        print('Did not fixed Dates....')
        traceback.print_exc()
        pass
        
    nonSLSNOpenSNe['Date'] = pd.to_datetime(nonSLSNOpenSNe["Disc. Date"].map(lambda x: "2016" if isinstance(x, float) and np.isnan(x) else x.split(",")[0]))
    nonSLSNOpenSNe['year'] = pd.to_datetime(nonSLSNOpenSNe['Date'].map(lambda x:  x.year), format= "%Y")

    # print('DateDateDate',min(nonSLSNOpenSNe.Date), max(nonSLSNOpenSNe.Date))
    nonSLSNOpenSNe.info()
    print ("w >=5 phot and >1 spec",
           nonSLSNOpenSNe[(nonSLSNOpenSNe["Phot."]>=5) &
                   (nonSLSNOpenSNe["Spec."]>0)].shape)
    return nonSLSNOpenSNe

#1 plot disovery

def plotDiscovery(df, df2 = None, verbose=False):
    fig, (ax1, ax2) = pl.subplots(2, sharex=True, figsize=(12,12))
    mpl.rcParams.update({"legend.frameon" : True,
                     "legend.fancybox" : True,
                     "legend.framealpha" : 0.5,
                     "axes.facecolor" : "white",
                     "axes.edgecolor": "#bcbcbc"})
    #print (pd.cut(df["year"], bins = 14))
    #print(max(df2["year"]))
    pd.to_datetime(df["year"]).hist(bins = 22, ax=ax1,
                                    label="all stripped SNe with photometry",
                                    color="SteelBlue")
    print((pd.to_datetime(df["year"])))
    count,divisions = pd.cut(pd.to_datetime(df["year"]), 22,retbins=True)

    if not df2 is None: 
        pd.to_datetime(df2["year"]).hist(bins = divisions, ax=ax1,
                                    label="our template sample",
                                    color="IndianRed")
    ax1.legend(loc=2)
    ax1.set_yticks([0,100,200,300, 400, 500])
    ax1.set_yticklabels(["","100","200","300", "400", "500"])
    ax1.set_ylim(0,500)    
    
    
    to_timestamp = np.vectorize(lambda x: pd.to_timedelta(x -
                                            pd.to_datetime("1940/1/1",
                                                           format="%Y/%m/%d")))

    #to_timestamp( pd.to_datetime(nonSLSNOpenSNe[\
        #                                        nonSLSNOpenSNe["Phot."]>10]\
        #                                          ["year"]))
    #ax2.grid('off')

    #select subset with 1 spectrum and 10 photometric dps at least
    wellobs = df[(df["Phot."]>=10) &
                 (df["Spec."]>0)]
    years = pd.DatetimeIndex(wellobs["year"].values).year
    print ("with 10 photometric measurements and 1 spectrum at least",
           wellobs.shape)
    pd.to_datetime(wellobs["year"]).hist(bins = 20,
                                    ax=ax2,
                                    label=(r"with $\geq 10$ " +
                                           "photometric datapoints & $\geq 1$ spectrum"), color="SteelBlue")

    #pl.hist(years, bins=14)
    
    ax2.set_yticks([0,25,50])
    ax2.set_yticklabels(["","25","50"])
    ax2.set_title(" ", )
    ax2.set_ylim(0,55)
     
    pl.legend(loc='upper center',  fancybox=True, framealpha=0.5)
    fig.subplots_adjust(hspace=0.05)


    # crete color band that represents average photometric datapoints
    if verbose:
        print ((wellobs.groupby(\
                     ["year", pd.Grouper(freq='A',key='Date')]).mean())\
               .reset_index())
    tmp = ((wellobs.groupby(\
                     ["year", pd.Grouper(freq='A',key='Date')]).mean())\
        .reset_index().year.values,
        (wellobs.groupby(\
                     ["year", pd.Grouper(freq='A',key='Date')]).mean())\
        .reset_index()["Phot."].values)

    # create a moc x axis for interpolation
    x =  ((np.concatenate(\
        [np.array([np.datetime64(datetime.datetime(1950,1,1))]),
         tmp[0],
         np.array([np.datetime64(datetime.datetime(2020,1,1))])]) -\
           np.datetime64(datetime.datetime(1950,1,1))).astype(float) /\
          (tmp[0][0]- np.datetime64(datetime.datetime(1950,1,1))).astype(float))
    y = np.concatenate([np.array([0]), tmp[1], np.array([0])])

    f = interp1d(x, np.log(y+1))

    
    print ("max number of DPs:", max(y))
    print ("earliest", (min(df["Date"]).date()),
           (min(wellobs["Date"])).date())
    print ("latest", (max(df["Date"])).date(),
           (max(wellobs["Date"])).date())
    xnew = np.linspace(0, x.max(), 100, endpoint=True)
    colors = cm.viridis(f(xnew) / f(xnew).max())
    i = 0
    ax3 = ax2.twiny()
    for yy, c in zip(np.ones(len(xnew)) * 110, colors):
        if verbose:
            print (yy,c,xnew[i])
        c[3] = 0.5
        pl.scatter(xnew[i], yy, color=c, marker='s')
        i = i + 1
    ax3.set_xticks(ax3.get_ylim())
    print((ax2.get_xticklabels()))
    print((ax3.get_xticklabels()))
    ax3.set_xticklabels(["" for i in ax3.get_xticklabels()])
    ax3.set_xlim(xnew[0], xnew[-1])
    ax3.set_ylim(0, 140)
    
    ax1.set_ylabel("SNe discovered", fontsize=21)
    
    ax2.set_ylabel("SNe discovered", fontsize=21)
    
    ax2.set_xlabel("Discovery Year", fontsize=21)
    #pl.figure()
    #ax4 = fig.add_subplot(313)
    #pl.plot (x,y, 'o')
    #pl.plot (xnew,f(xnew), '-')
    #ax4.set_xlim(xnew[0], xnew[-1])
    
    #print(pl.hist(years, bins=16, color='red'))
    #pl.xlim(1949.9,2019.9)

    #pl.show()
    pl.savefig("SESSNDiscoveryByYear.pdf",  bbox_inches='tight')
    # os.system("pdfcrop SESSNDiscoveryByYear.pdf ../figs/SESSNDiscoveryByYear.pdf")
    
def mkzfloat(x):
    try:
        xx = float(x)
    except ValueError:
        xx = str(x).split(',')[0]
    return xx


def zhist(df, ax = None, cut=True, color="SteelBlue", label = ""):
    if ax is None:
        fig = pl.figure(figsize=(8,6))
        ax = fig.add_subplot(211)
    ax.set_xlabel(r"$z$", fontsize=21)
    ax.set_ylabel("Number of SE SNe", fontsize=21)
   
    #\n in Open SN Catalog")
    wellobs = df[(df["Phot."]>=10) &
                 (df["Spec."]>0)]    
    if cut:
        df['zfloat'] = df.z.map(mkzfloat ).\
                       astype(float)
    df['zfloat'].hist(bins = np.arange(-0.1,1.2,0.05), ax=ax, color=color,
                      label = label)
    

    #pl.show()
    return ax
    # make up an x axis
    #print(np.concatenate([np.array([np.datetime64(datetime.datetime(1950,1,1))]),
    #                      tmp[0] - np.datetime64(datetime.datetime(1950,1,1)),
    #                      np.array([np.datetime64(datetime.datetime(2020,1,1))])]))
    #,
    #                      np.array([np.datetime64(datetime.datetime(2020,1,1))])]))

def plotzWbox(df, cut=True, df2 = None):
    f, (ax1, ax2) = pl.subplots(2,1,
                                gridspec_kw = {'height_ratios':[2.5, 1]},
                                sharex=True)
    pl.subplots_adjust( hspace=0.0 )
    zhist(df, ax1, cut=cut, label="all SE SN with photometry")
    
    sns.set_style("whitegrid")
    print ("df1 percentiles -16:{0:.5f} 50:{1:.5f} 16:{2:.5f}, 99.7:{3:.5f}".\
           format(*np.percentile(df['zfloat'].dropna().values, [16., 50., 84., 99.7])))
    
    if not df2 is None:
        print ("df1 percentiles -16:{0:.5f} 50:{1:.5f} 16:{2:.5f}, 99.7:{3:.5f}".\
               format(*np.percentile(df2['zfloat'].dropna().values, [16., 50., 84., 99.7])))
    
        
        zhist(df2, ax1, cut=False, color="IndianRed",
              label="our template sample")
        ax1.legend(loc = 1, frameon=True, fancybox=True, framealpha=0.5)
        ax1.set_yscale("log")     
           
        sns.boxplot(data = [df.zfloat, df2.zfloat],
                    orient="h", ax=ax2, flierprops={"marker": "o", "ms":3}),
        
        ax2.set_yticklabels(["", ""])
        
    else:
        sns.stripplot(df.zfloat, ax=ax2, color="#aaaaaa")
        ax2.set_yticklabels([""])        
    #ax2.set_ylim(0.01, ax2.get_ylim()[1])
    pl.xlabel("z")
    
    #pl.show()
    pl.savefig("SESSNzDistrib.pdf")
    # os.system("pdfcrop SESSNzDistrib.pdf ../figs/SESSNzDistrib.pdf")
    
if __name__ == "__main__":
    # read in
    nonSLSNOpenSNe = read_allSNeOpenSNCat()

    sesndf = pd.read_csv(os.getenv("SESNCFAlib") +"/SESNessentials.csv", comment="#", encoding = "ISO-8859-1")
    # print (sesndf.Type)
    #sesndf.replace(to_replace="<0000000", value=np.nan, inplace=True)
    for key in sesndf.columns[2:]:
        sesndf[key] = pd.to_numeric(sesndf[key], errors='coerce')

    sesndf = sesndf.dropna(subset=[u'finalmaxVjd',
       u'CfA VJD bootstrap',
       u'CfA BJD bootstrap', u'CfA RJD bootstrap',
       u'CfA IJD bootstrap', u'D11Vmaxdate'], how='all')

    sesndf['Name'] = sesndf.SNname.apply(lambda x: x.replace("sn1","SN1").\
                                         replace("sn2","SN2").\
                                         replace("SDSS-II","SDSS-II SN "))

    # print (nonSLSNOpenSNe.columns)
    # print (sesndf.columns)
    
    sesndfmerged = sesndf.merge(nonSLSNOpenSNe, how="left",
                                on="Name", indicator=True)
    # print('Type_x is:'+ sesndfmerged['Type_x'])
    print (" not in OSN but in my DB:",
           sesndfmerged[sesndfmerged._merge == 'left_only'].Name)

    sesndfmerged['Type'] = sesndfmerged.Type_x
    sesndfmerged['zfloat'] = sesndfmerged.z.map(mkzfloat ).\
                       astype(float)
    sesndfmerged['z'] = sesndfmerged.zfloat.map(lambda x: "%.2f"%x)
    sesndfmerged.index = sesndfmerged.Name
    
    for s in sesndfmerged.SNname.values:
        print ("%s"%s.replace("sn20","").replace("sn19",""),)
    print ("")

    '''
    #print if types are different in 2 tables
    tmp = [a+" "+b+" "+c+"\n" for a,b,c in
           sesndfmerged[['Name', 'Type_x', 'Type_y']].values
           if not b==c ]
    for t in tmp:
         print t
    '''
    
    print ("z>0.2")
    print (sesndfmerged[sesndfmerged.zfloat>0.2])
    sesndfmerged['Vmax'] = np.zeros(len(sesndfmerged.SNname))
    sesndfmerged['Vmax err'] = np.zeros(len(sesndfmerged.SNname))
    if "largetable" in sys.argv[1:]:
        for b in su.bands:
            sesndfmerged[b] = ['0'] * len(sesndfmerged.SNname)
            sesndfmerged[b+' SNR'] = np.zeros(len(sesndfmerged.SNname)) 
            # sesndfmerged[b+' SNR Median'] = np.zeros(len(sesndfmerged.SNname)) 
            # sesndfmerged[b+' SNR [min,max]'] = ['-'] * len(sesndfmerged.SNname)           
            sesndfmerged[b+'[min,max]'] = ['-'] * len(sesndfmerged.SNname)
            sesndfmerged[b+ ': N, [min,max], SNR'] = ['-'] * len(sesndfmerged.SNname)


    for sn in sesndfmerged.SNname.index:
        thissn = mysn(sesndfmerged.loc[sn].SNname, addlit=True)
        thissn.readinfofileall(verbose=False, earliest=False, loose=True)
        
        # thissn.loadsn2()
        lc, flux, dflux, snname = thissn.loadsn2(verbose=True)
        lc_mag = lc['ccmag'][lc['dmag'] >= 0.01]
        lc_photcode = lc['photcode'][lc['dmag'] >= 0.01]
        lc_dmag = lc['dmag'][lc['dmag'] >= 0.01]

        flux = 10 ** (-lc_mag / 2.5) * 5e10
        dflux = flux * lc_dmag / LN10x2p5

        snr = flux/dflux
        photcode = [lc_photcode[i].decode().split('l')[0] for i in range(len(lc_photcode))]
        thissn.setphot()
        thissn.getphot()
        thissn.setVmax()
        # thissn.printsn()
        if all(value == 0 for value in thissn.filters.values()):
            sesndfmerged.drop(sn)
            
        sesndfmerged.at[sn, 'Vmax'] =  thissn.Vmax
        if  np.isnan(thissn.dVmax):
            thissn.dVmax = 2
        sesndfmerged.at[sn, 'Vmax err'] = thissn.dVmax
        # print(sesndfmerged['Type'][sn])
        sesndfmerged['Type'] = sesndfmerged.Type.map(lambda x:
                                                 x.replace('BL','bl'))
        sesndfmerged['Type'] = sesndfmerged.Type.map(lambda x:
                                                     x.replace('Ca-rich','Ca'))
        sesndfmerged['Type'] = sesndfmerged.Type.map(lambda x:
                                                     x.replace('Ibc','Ib/c'))


        if "largetable" in sys.argv[1:]:
            # thissn.getphot()
            thissn.setphase()
            print (sn, thissn.filters)
            for b in su.bands:
                sesndfmerged.at[sn, b] = '%d'%thissn.filters[b]
                if thissn.filters[b]:
                    sesndfmerged.at[sn, b+'[min,max]'] = '[%.1f,%.1f]'%(thissn.photometry[b]['phase'].min(),\
                                      thissn.photometry[b]['phase'].max())
                    sesndfmerged.at[sn, b+' SNR'] =\
                            '%.1f'%np.round(np.nanmedian(snr[np.asarray(photcode) == b]),2)
                    # sesndfmerged.at[sn, b+' SNR[min,max]'] = '[%.1f,%.1f]'%(snr[np.asarray(photcode) == b].min(),\
                    #                   snr[np.asarray(photcode) == b].max())
                    sesndfmerged.at[sn, b+': N, [min,max], SNR'] = '%d'%thissn.filters[b] + ', '+\
                                                           '[%.1f,%.1f]'%(thissn.photometry[b]['phase'].min(),\
                                                                       thissn.photometry[b]['phase'].max())+ ', '+\
                                                           '%.1f'%np.round(np.nanmedian(snr[np.asarray(photcode) == b]),2)

                if b in ['J', 'H', 'K']:   
                  print(sn, b, sesndfmerged.at[sn, b])       
    sesndfmerged.sort_values('Vmax', inplace=True)
    if "table" in sys.argv[1:]:        
        #editing        
        sesndfmerged['Vmax'] = sesndfmerged.Vmax.map(lambda x: "%7.1f"%x)
        sesndfmerged['Vmax err'] = sesndfmerged['Vmax err'].map(lambda x:
                                                                "%1.1f"%x)
        print  ("final sample size", sesndfmerged.shape)
        tname = os.getenv("DB")\
        + "papers/SESNtemplates.working/tables/photsample_data_sk_add_Ibns.tex"
        sesndfmerged.to_latex(buf=tname,
                              columns=[ u'Type', 'Vmax', 'Vmax err', 'z'],
                              longtable=True)
        os.system("sed -e '1,13d' < %s > tmptable"%(tname))
        os.system("sed '$d' tmptable > %s"%(tname))
        
    if "zhist" in sys.argv[1:]:
        zhist(nonSLSNOpenSNe)
    if "zbox" in sys.argv[1:]:
        plotzWbox(nonSLSNOpenSNe, df2=sesndfmerged)
    if "discovery" in sys.argv[1:]:
        plotDiscovery(nonSLSNOpenSNe, df2=sesndfmerged)

    if "largetable" in sys.argv[1:]:

        sesndfmerged.drop([u'source', u'E(B-V) SF2011', u'finalmaxVjd',
                           u'finalmaxVjderr', u'CfA VJD bootstrap',
                           u'CfA VJD bootstrap error',
                           u'CfA BJD bootstrap', u'CfA BJD error',
                           u'CfA RJD bootstrap',
                           u'CfA RJD error', u'CfA IJD bootstrap',
                           u'CfA IJD error',
                           u'D11Vmaxdate', u'D11Vmaxdateerr',
                           u'MaxVdate', u'MaxVJD',
                           u'Vmax_CfAmethod', u'Vmax_CfAmethod_err',
                           u'MaxVMag', u'MaxVmagerr',
                           u'luminosity distance Mpc', u'comment',
                           u'Name', u'Disc. Date', u'R.A.',
                           u'Dec.', u'z', u'Type_y', u'Phot.',
                           u'Spec.', u'Date',
                           u'year'], axis=1, inplace=True)
        

        sesndfmergedUBVRI = sesndfmerged.drop([sn for sn in
                                    sesndfmerged.SNname.index
                if sesndfmerged.reindex(index=[sn], columns=['U','B','V',
                                             'R','I']).values.astype(float).astype(int).sum() == 0])
        sesndfmergedUBVRI.to_latex(buf=os.getenv("DB")
                    + "papers/SESNtemplates.working/tables/allphotUBVRI_snr3.tex",
                              columns=[ u'Type',
                                        'U: N, [min,max], SNR',
                                        'B: N, [min,max], SNR',
                                        'V: N, [min,max], SNR',
                                        'R: N, [min,max], SNR',
                                        'I: N, [min,max], SNR'],
                                   longtable=True)
        
        sesndfmergedugri = sesndfmerged.drop([sn for sn in
                                    sesndfmerged.SNname.index
                if sesndfmerged.reindex(index=[sn],columns=['u',
                                             'r','i']).values.astype(float).astype(int).sum() == 0])
        
        print(sesndfmergedugri.reindex(index=[sn], columns=['g','g'+ '[min,max]']))

        sesndfmergedugri.reindex(columns=['Type','u: N, [min,max], SNR','g: N, [min,max], SNR',
                                          'r: N, [min,max], SNR','i: N, [min,max], SNR']).rename(columns={
                                         'u: N, [min,max], SNR':"u': N, [min,max], SNR",
                                         'g: N, [min,max], SNR':"g': N, [min,max], SNR",
                                         'r: N, [min,max], SNR':"r': N, [min,max], SNR",
                                         'i: N, [min,max], SNR':"i': N, [min,max], SNR"}
                                ).to_latex(buf=os.getenv("DB")
                    + "papers/SESNtemplates.working/tables/allphotugri_snr3.tex",
                              columns=[ u'Type', "u': N, [min,max], SNR",
                                        "g': N, [min,max], SNR",
                                        "r': N, [min,max], SNR",
                                        "i': N, [min,max], SNR"],
                                           longtable=True)

        sesndfmergedJKHUV = sesndfmerged.drop([sn for sn in
                                    sesndfmerged.SNname.index
                if sesndfmerged.reindex(index=[sn], columns=['J','H',
                                             'K', 'w1',
                                             'w2','m2']).values.astype(float).astype(int).sum() == 0])
        print(sesndfmergedJKHUV.index)
        sesndfmergedJKHUV.to_latex(buf=os.getenv("DB")
                    + "papers/SESNtemplates.working/tables/allphotJHKUV_snr3.tex",
                              columns=[ u'Type', "J: N, [min,max], SNR",
                                        "H: N, [min,max], SNR",
                                        "K: N, [min,max], SNR",
                                        "w1: N, [min,max], SNR", 
                                        "w2: N, [min,max], SNR",
                                        "m2: N, [min,max], SNR"],
                                   longtable=True)

        # sesndfmergedUV = sesndfmerged.drop([sn for sn in
        #                             sesndfmerged.SNname.index
        #         if sesndfmerged.reindex(index=[sn], columns=['w1',
        #                                      'w2','m2']).values.astype(float).astype(int).sum() == 0])
        # print(sesndfmergedUV.index)
        # sesndfmergedUV.to_latex(buf=os.getenv("DB")
        #             + "papers/SESNtemplates.working/tables/allphotUV_snr2.tex",
        #                       columns=[ u'Type', "w1: N, [min,max], SNR", 
        #                                 "w2: N, [min,max], SNR",
        #                                 "m2: N, [min,max], SNR",],
        #                            longtable=True)

    print (sesndfmerged.groupby("Type")["SNname"].count())
    print (np.sum(sesndfmerged.groupby("Type")["SNname"].count().values))
    pd.DataFrame({'Type':sesndfmerged.groupby("Type")["SNname"].count().index,
              'Count':sesndfmerged.groupby("Type")["SNname"].count().values}).to_latex(buf=os.getenv("DB")
                    + "papers/SESNtemplates.working/tables/typesCount.tex",
                              columns=[ u'Type', "Count"],
                                   longtable=True)
    fout = open("osnSESN.dat", "w")
    for nm in sesndfmerged.index.values:
        fout.write(nm+"\n")
    fout.close()
    
    

