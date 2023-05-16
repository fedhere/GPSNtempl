import numpy as np
import glob
import os,inspect,sys
import pandas as pd
try:
     os.environ['SESNPATH']
     os.environ['SESNCFAlib']

except KeyError:
     print ("must set environmental variable SESNPATH and SESNCfAlib")
     sys.exit()

RIri = False

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib")+"/templates"
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff

def prepsn(sn):
     #     thissn = snstuff.mysn(f[0], addlit=True)
    thissn = snstuff.mysn(sn, addlit=True, fnir=True)
    if len(thissn.optfiles) + len(thissn.fnir) == 0:
        return -1
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)

    if thissn.Vmax is None or thissn.Vmax == 0 or np.isnan(thissn.Vmax):
        return -1

    #raw_input()
    lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
    thissn.setphot()
    thissn.getphot()

    if np.array([n for n in thissn.filters.itervalues()]).sum() == 0:
        return -1
    
    #thissn.plotsn(photometry=True)
    thissn.setphase()
    #thissn.printsn()
 
    return thissn


def makephotable():
    inputSNe = pd.read_csv(os.getenv("SESNCFAlib") + "/SESNessentials.csv")['SNname'].values

    print (inputSNe)

    bands1 = ['U','u','B','V','R','r','I','i']
    bands2 = ['w2','m2','w1','H','J','K']
    
    tmp1={}
    i=1
    tmp1['%dType'%i] = {}
    for b in bands1:
        #print (b)
        i=i+1
        tmp1["%d"%i+b]={}
        i=i+1
        tmp1["%d"%i+b+"[min,max]"] = {}
    raw_input()
    tmp2={}
    i=1
    tmp2['%dType'%i] = {}
    for b in bands2:
        i=i+1
        tmp2["%d"%i+b]={}
        i=i+1
        tmp2["%d"%i+b+"[min,max]"] = {}
    i=i+1 
    tmp2["%d"%i+'Any'] = {}
    i=i+1
    tmp2["%d"%i+"Any[min,max]"] = {}
    for f in inputSNe:
        print (f       )
        thissn = prepsn(f)
        if thissn == -1: 
             #print "here"
             continue
        else: 
             print (thissn.photometry[b]['phase'])
        add2DF(thissn, tmp1, tmp2, bands1, bands2)
    print (tmp1)
    
    bands = []
    for b in bands1:
         bands.append(b)
         bands.append(b+"[min,max]")

    tabletex = "../../papers/SESNexplpars/tables/AllPhotOptTable.tex"
    add2table(tmp1, bands, tabletex)
    

    bands = []
    for b in bands2:
         bands.append(b)
         bands.append(b+"[min,max]")

    tabletex = "../../papers/SESNexplpars/tables/AllPhotUVNIRTable.tex"
    add2table(tmp2, bands, tabletex)
        
             
def add2DF(thissn, DF1,  DF2, bands1, bands2):  
        try:
             DF1['1Type']
        except:
             DF1['1Type'] = {}
        DF1['1Type'][thissn.name] = thissn.type
        i = 1
        for b in bands1:
            i=i+1
            try:
                 DF1["%d"%i+b]
            except KeyError:
                 DF1["%d"%i+b] = {}
            DF1["%d"%i+b][thissn.name]=  thissn.filters[b]
            i=i+1
            try:
                 DF1["%d"%(i-1)+b]
            except KeyError:
                 DF1["%d"%i+b] = {}
            try:
                 DF1["%d"%(i-1)+b+"[min,max]"]
            except KeyError:
                 DF1["%d"%i+b+"[min,max]"] = {}
          
            if DF1["%d"%(i-1)+b][thissn.name]>0: DF1["%d"%i+b+"[min,max]"][thissn.name] = [thissn.photometry[b]['phase'].min(),
                                                                                           thissn.photometry[b]['phase'].max()]
            else: DF1["%d"%i+b+"[min,max]"][thissn.name] = "-"
        #tmp1 = tmp1[[bands1]]
        try:
             DF2['1Type']
        except KeyError:
             DF2['1Type'] = {}
             
        DF2['1Type'][thissn.name]=thissn.type              
        i=1
        for b in bands2:
             i=i+1
             try:
                  DF2["%d"%i+b]
             except KeyError:
                  DF2["%d"%i+b] = {}
             try:
                  DF2["%d"%i+b+"[min,max]"]
             except KeyError:
                  DF2["%d"%i+b+"[min,max]"] = {}
             try:
                  DF2["%d"%(i-1)+b+"[min,max]"]
             except KeyError:
                  DF2["%d"%(i-1)+b+"[min,max]"] = {}
             DF2["%d"%i+b][thissn.name]=  thissn.filters[b]
             i=i+1
             try:
                  DF2["%d"%i+b]
             except KeyError:
                  DF2["%d"%i+b] = {}
             try:
                  DF2["%d"%i+b+"[min,max]"]
             except KeyError:
                  DF2["%d"%i+b+"[min,max]"] = {}
             try:
                  DF2["%d"%(i-1)+b+"[min,max]"]
             except KeyError:
                  DF2["%d"%(i-1)+b+"[min,max]"] = {}
                  
             if DF2["%d"%(i-1)+b][thissn.name]>0: DF2["%d"%i+b+"[min,max]"][thissn.name] = [thissn.photometry[b]['phase'].min(),thissn.photometry[b]['phase'].max()]
             else: DF2["%d"%i+b+"[min,max]"][thissn.name] = "-"
        i=i+1
        try:
             DF2["%d"%i+'Any']
        except:
             DF2["%d"%i+'Any'] = {}             
        try:
             DF2["%d"%(i-1)+'Any']
        except:
             DF2["%d"%(i-1)+'Any'] = {}
        try:
             DF2["%d"%(i+1)+'Any']
        except:
             DF2["%d"%(i+1)+'Any'] = {}
        try:
             DF2["%d"%(i+1)+'Any[min,max]']
        except:
             DF2["%d"%(i+1)+'Any[min,max]'] = {}

        
        DF2["%d"%i+'Any'][thissn.name]=  np.array([thissn.filters[b] for b in bands1+bands2]).sum()
        i=i+1
        if DF2["%d"%(i-1)+'Any'][thissn.name]>0: DF2["%d"%i+"Any[min,max]"][thissn.name] = \
           [np.array([thissn.photometry[b]['phase'].min() for b in bands1+bands2 if thissn.filters[b]>0]).min(),
            np.array([thissn.photometry[b]['phase'].max() for b in bands1+bands2 if thissn.filters[b]>0]).max()]
        else: DF2["%d"%i+"Any[min,max]"][thissn.name] = "-"
        

def add2table(DF1, bands, tabletex):
        #"../../papers/SESNexplpars/tables/AllPhotOptTable.tex",

 
        pd.DataFrame(DF1).to_latex(open(tabletex, "w"))

        f = open(tabletex, "r")
        lines = f.readlines()
        f.close()
        f = open(tabletex, "w")
        for line in lines[4:-2]:
             f.write(line)
        f.close()

       
if __name__ == '__main__':
     makephotable()
