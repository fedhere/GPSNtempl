import os
import pickle as pkl
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    os.environ['SESNPATH']
    os.environ['SESNCFAlib']

except KeyError:
    print ("must set environmental variable SESNPATH and SESNCfAlib")
    sys.exit()

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib") + "/templates"
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)

from scipy.optimize import curve_fit


def gaussian(x, mean, amplitude, standard_deviation):
    return amplitude * np.exp( - (x - mean)**2 / (2*standard_deviation ** 2))
    

all_params = pkl.load(open("outputs/all_params_scipy_opt.pkl", "rb"))

readgood = pd.read_csv("goodGPs_parameter_selection.csv", header=None)

par1 = []
par2 = []

perType = True
tps = ['Ib',
        'Ibn',
        'IIb',
        'Ic',
        'Ic-bl']

tp = tps[0]


if perType:
    sel_list = readgood[readgood[2] == tp].reset_index(drop=True)
else:
    sel_list = readgood



for i, sn in enumerate(sel_list[0]):
    if sel_list[3][i] == 'y':
        
        if sel_list[1][i].endswith('p'):
            band = sel_list[1][i][0]
        else:
            band = sel_list[1][i]
        
        
        if len(all_params[sn][band]) != 0:
#             print(all_params[sn][readgood[1][i]])
            par1.append(all_params[sn][band][0][0])
            par2.append(all_params[sn][band][0][1])
        else:
            print(sn, band)

print('Median of parameter 1: ', np.median(par1))
print('Median of parameter 2: ', np.median(par2))

print('Average of parameter 1: ', np.mean(par1))
print('Average of parameter 2: ', np.mean(par2))

f, axs = plt.subplots(1,2, figsize=(25,10))
f.subplots_adjust(wspace=0.1)

x = par1
bin_heights, bin_borders, _ = axs[0].hist(x, bins=100, label='${\Theta}_1$ across all SN')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.])

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
axs[0].plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt),linewidth=4,color='red', label='Gaussian fit')
axs[0].legend(prop={'size':20})
axs[0].text(0.02,0.8,'$\mu$ = '+ str(round(popt[0],2)),size=25, transform=axs[0].transAxes)
axs[0].text(0.02,0.7,'Median = '+ str(round(np.median(par1),2)),size=25, transform=axs[0].transAxes)
axs[0].text(0.02,0.6,'Mean = '+ str(round(np.mean(par1),2)),size=25, transform=axs[0].transAxes)


x = par2
bin_heights, bin_borders, _ = axs[1].hist(x, bins=100, label='${\Theta}_2$ across all SN')
bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
popt, _ = curve_fit(gaussian, bin_centers, bin_heights, p0=[1., 0., 1.], maxfev = 18000)

x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)
axs[1].plot(x_interval_for_fit, gaussian(x_interval_for_fit, *popt),linewidth=4,color='red', label='Gaussian fit')
axs[1].legend(prop={'size':20})

axs[1].text(0.8,0.8,'$\mu$ = '+ str(round(popt[0],2)),size=25, transform=axs[1].transAxes)
axs[1].text(0.68,0.7,'Median = '+ str(round(np.median(par2),2)),size=25, transform=axs[1].transAxes)
axs[1].text(0.7,0.6,'Mean = '+ str(round(np.mean(par2),2)),size=25, transform=axs[1].transAxes)



for tick in axs[0].xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in axs[1].xaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in axs[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
for tick in axs[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
    
if perType:
    f.savefig("outputs/find_final_params_"+tp+".pdf", bbox_inches='tight')
else:
    f.savefig("outputs/find_final_params.pdf", bbox_inches='tight')



