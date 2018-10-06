from __future__ import print_function
import os
import glob

datadir = "OSNCdownloads"
tablespath = os.getenv("GPTBLPATH")
sne = open(tablespath + "/osnSESN.test.dat").readlines()
sne = [sn.strip() for sn in sne]

print("SNe to download")
print (sne)

cmd0 = "wget https://sne.space/astrocats/astrocats/supernovae/output/json/"
N = len(sne)
for i,sn in enumerate(sne):
    print (i, "/", N)
    cmd = cmd0  + sn.replace(" ","%20") + ".json"
    
    #print (cmd)
    os.system(cmd)
    cmd =  "mv " +  sn.replace(" ","%20") + ".json" + " " + datadir
    print(cmd)
    os.system(cmd)
    
downloaded = glob.glob(datadir + "/*.json")

print ("downloaded SNe:\n", downloaded)
print ("looking for missing ones")
for i,sn in enumerate(sne):
    if not datadir + "/" + sn + ".json" in downloaded:
        print ("not got", sn.strip() + ".json")


#this last bit seems wrong
print ("looking for duplicates")        
for i,sn in enumerate(downloaded):
    if not sn.replace(".json","").strip(datadir + '/') in sne:
        print ("duplicate?", sn.replace(".json","").strip(datadir + '/'))

