'''
If you want to add a SN to the sample of this paper, you should use this code to convert
your photometry into our required format.
This function takes in:
- The name of the SN like SN1994I
- a pandas dataframe containing the photometry with columns:
	- time
	- mag
	- err
	- band
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import ascii
import sys, os

def format_fixer(name, df):
    
    if name.split('20')[1] == '':
        shortname = 'sn20'+ name.split('20')[2]
    else:
        shortname = 'sn'+ name.split('20')[1]
        
    fileout = open(os.getenv("SESNPATH") + "literaturedata/slc." + shortname + ".f", "w")
    bands = [df.groupby('band').size().keys()[i].split('_')[1] for i in range(len(df.groupby('filter').size().keys()))]
    
    for i, b in enumerate(bands):
        print(b, len(df[df['filter'] == 'ZTF_'+b]))
        
        if b in ['UVW1', 'UVW2', 'UVM2', 'H', 'Ks', 'J']:
            continue
        
        
        for j, t in enumerate(df[df['filter'] == 'ZTF_'+b]['jd']):
            
            dm = df[df['filter'] == 'ZTF_'+b]['magerr'].reset_index(drop=True)[j]
            
#             if float(dm) > 90:
#                 continue
#                 dm = 1
        
            fileout.write('{0} {1} {2} {3} {4} {5}\n'.format(b + 'l', t, 'nan', 'nan', dm,
                                                             df[df['filter'] == 'ZTF_'+b]['mag'].reset_index(drop=True)[j]))
    fileout.close()

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: running <SN name> <path to photomtry - it should be a csv file.>")
        sys.exit(1)

    name = sys.argv[1]
    df = pd.read_csv(sys.argv[2])

    format_fixer(name, df)
