from __future__ import print_function
import os
import glob

sne  = open("../../papers/SESNtemplates/tables/osnSESN.dat").readlines()
sne= [sn.strip() for sn in sne]
cmd0 = "wget https://sne.space/astrocats/astrocats/supernovae/output/json/"
N = len(sne)
for i,sn in enumerate(sne):
    print (i, "/", N)
    cmd = cmd0  + sn.replace(" ","%20") + ".json"
    #print (cmd)
    #os.system(cmd)

downloaded = glob.glob("*.json")

print ("looking for missing ones")
for i,sn in enumerate(sne):
    if not sn + ".json" in downloaded:
        print ("not got", sn.strip() + ".json")

print ("looking for duplicates")        
for i,sn in enumerate(downloaded):
    if not sn.replace(".json","") in sne:
        print ("duplicate?", sn.replace(".json",""))

