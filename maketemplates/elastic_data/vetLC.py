
# author Federica B Bianco
# validates the gaussian process for a single SN lightcurve in a single band.
# makes a plot and expects a yes/no validation answer

import glob as glob
import os
from PIL import Image
import pylab as pl


# set savefig to true to overwrite plots
SAVEFIG = True
# SAVEFIG = False
pl.ion()
subtype = "Ib"
flist = sorted(glob.glob("plots/"+subtype+"/*.png"))#medians.png")
# flist = flist.sort()
data = Image.open(flist[0])

fig = pl.figure().add_subplot(111)
im = fig.imshow(data)
fout = open("goodlc_"+subtype+".csv", "w")
for c,f in enumerate(flist):
    print (c+1, '/',len(flist),' ','SNID: '+f)
    im.set_data(Image.open(f))

    good = input("is this good? Y/n") #raw_input("is this good? Y/n")
    print (good)
    if good == '' :
        good='y'
    elif good.startswith("n"):
        good='n'
    else:
        good='y'
    print(f+","+good+"\n") 
    if SAVEFIG:
        fout.write(f+","+good+"\n")
    
