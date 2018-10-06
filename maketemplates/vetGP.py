import glob as glob
import os
from PIL import Image
import pylab as pl

pl.ion()
flist = glob.glob("outputs/GPfit*png")#medians.png")
data = Image.open(flist[0])

fig = pl.figure().add_subplot(111)
im = fig.imshow(data)
fout = open("goodGPs.csv", "w")
for f in flist:
    fnames = f.replace("GPfit","").replace("_medians.png","").split("_")
    print (fnames)
    fname = "_".join(fnames[:1])
    band = fnames[-1]

    im.set_data(Image.open(f))

    good = raw_input("is this good? Y/n")
    print (good)
    if good == '' :
        good='y'
    elif good.startswith("n"):
        good='n'
    else:
        good='y'
    print(fname.replace("outputs/","")+","+band+","+good+"\n")    
    fout.write(fname+","+band+","+good+"\n")
    
