
# author Federica B Bianco
# validates the gaussian process for a single SN lightcurve in a single band.
# makes a plot and expects a yes/no validation answer

import glob as glob
import os
from PIL import Image
import pylab as pl
import sys
import pandas as pd

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

# set savefig to true to overwrite plots
SAVEFIG = True
# SAVEFIG = False
pl.ion()
flist = sorted(glob.glob("outputs/test1/GPfit*_opt_test.png"))#GPfit*sk_2022_opt*.png"))#medians.png")
# flist = flist.sort()

data = Image.open(flist[0])

fig = pl.figure().add_subplot(111)
im = fig.imshow(data)
fout = open("goodGPs_parameter_selection_test", "w")
allsne = pd.read_csv(os.getenv("SESNCFAlib") +
                          "/SESNessentials.csv", encoding = "ISO-8859-1")
for c,f in enumerate(flist):
    fnames = f.replace("GPfit","").replace("_test.png","").split("_")
    print (c+1, '/',len(flist),' ',fnames[0].split('/')[2])
    fname = fnames[0].split('/')[2] #"_".join(fnames[:1])
    band = fnames[1] #fnames[-1].split('.')[0]
    type = allsne.Type[allsne['SNname'] == fname].values[0]


    im.set_data(Image.open(f))

    good = input("is this good? Y/n") #raw_input("is this good? Y/n")
    print (good)
    if good == '' :
        good='y'
    elif good.startswith("n"):
        good='n'
    else:
        good='y'
    print(fname + "," + band + "," + type + "," + good + "\n")
    # fname = fname.split('/')[1]

    if SAVEFIG:
        fout.write(fname+","+band+","+type+","+good+"\n")
    
