# Read in ZTF forced photometry files and save in our desired format


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
import sys, os

def read_ztf_txt(filepath):
    file1 = open(filepath, 'r')
    Lines = file1.readlines()

    for line in Lines:
        if line.startswith('#'):
            continue
        if line.startswith(' index'):
            columns = line.split()

    for i,c in enumerate(columns):
        columns[i] = c.strip().split(',')[0]

    df = pd.DataFrame(columns=columns)

    for line in Lines:
        if line.startswith('#'):
            continue
        if line.startswith(' index'):
            continue
        df.loc[len(df.index)] = line.split()

    df['forcediffimflux'][df['forcediffimflux']=='null']= np.nan
    df['forcediffimfluxunc'][df['forcediffimfluxunc']=='null']= np.nan
    df['forcediffimsnr'][df['forcediffimsnr']=='null']= np.nan
    df['zpdiff'][df['zpdiff']=='null']= np.nan

    df['jd'] = df['jd'].astype(float)
    df['forcediffimflux'] = df['forcediffimflux'].astype(float)
    df['forcediffimfluxunc'] = df['forcediffimfluxunc'].astype(float)
    df['forcediffimsnr'] = df['forcediffimsnr'].astype(float)
    df['zpdiff'] = df['zpdiff'].astype(float)


    df['mag'] = np.zeros(len(df))
    df['magerr'] = np.zeros(len(df))

    return df 

def apply_snr(df, snr):
    df = df[(df['forcediffimsnr']>snr)]
    df['mag'] = df['zpdiff'] - 2.5*np.log10(df['forcediffimflux'])
    df['magerr'] = 2.5/np.log(10)*np.abs(df['forcediffimfluxunc']/df['forcediffimflux'])

    return df


def format_fixer(name, df):
    
    if name.split('20')[1] == '':
        shortname = 'sn20'+ name.split('20')[2]
    else:
        shortname = 'sn'+ name.split('20')[1]
        
    fileout = open(os.getenv("SESNPATH") + "literaturedata/phot/slc." + shortname + ".f", "w")
    bands = [df.groupby('filter').size().keys()[i].split('_')[1] for i in range(len(df.groupby('filter').size().keys()))]
    
    for i, b in enumerate(bands):
        print(b, len(df[df['filter'] == 'ZTF_'+b]))
        
        if b in ['UVW1', 'UVW2', 'UVM2', 'H', 'Ks', 'J']:
            continue
        
        
        for j, t in enumerate(df[df['filter'] == 'ZTF_'+b]['jd']):
            
            dm = df[df['filter'] == 'ZTF_'+b]['magerr'].reset_index(drop=True)[j]
            
        
            fileout.write('{0} {1} {2} {3} {4} {5}\n'.format(b + 'l', t, 'nan', 'nan', dm,
                                                             df[df['filter'] == 'ZTF_'+b]['mag'].reset_index(drop=True)[j]))
    fileout.close()

def plot_me(sn_name, df, bands, peak = 0):
    plt.figure()

    for b in bands:
        if b == 'r':
            color = 'red'
        elif b== 'g':
            color = 'green'
        elif b == 'i':
            color = 'purple'
        plt.errorbar(df['jd'][(df['filter']=='ZTF_'+b)]-2400000.5-peak, 
                     df['mag'][(df['filter']=='ZTF_'+b)],
                     yerr = df['magerr'][(df['filter']=='ZTF_'+b)],
                     marker='.', ls='none',
                     color=color,
                     label = b)
    plt.xlabel('Time (MJD)')
    plt.ylabel('Magnitude')
    plt.title(sn_name)
    plt.axvline(peak)
    plt.legend()
    if peak != 0:
        plt.xlim(-25, 100)

    plt.gca().invert_yaxis()
    plt.savefig(os.getenv("SESNPATH")+ "maketemplates/more_plots/%s.png"%sn_name)



if __name__ == "__main__":
    plot = True

    if len(sys.argv)<3 or len(sys.argv)>5:
        print("Usage: python read_ztf_file.py <sn_name> <ztf_path> <snr-optional> <MJD_peak-optional>")
        sys.exit(1)

    sn_name = sys.argv[1]
    ztf_path = sys.argv[2]

    if len(sys.argv)>3:
        snr = int(sys.argv[3])
    else:
        snr = 3

    if len(sys.argv)>4:
        peak = float(sys.argv[4])
    else:
        peak = 0

    df = read_ztf_txt(ztf_path)
    df = apply_snr(df, snr)
    format_fixer(sn_name, df)

    bands = [df.groupby('filter').size().keys()[i].split('_')[1] 
             for 
             i 
             in 
             range(len(df.groupby('filter').size().keys()))]

    if plot:
        plot_me(sn_name, df, bands, peak = peak)




