import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import gzip
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import pickle as pkl
from numpy import math
import traceback
import sys
from scipy import interpolate

from tqdm import tqdm
from scipy.interpolate import InterpolatedUnivariateSpline

from matplotlib.ticker import AutoMinorLocator
import os

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
	sys.path.insert(0, cmd_folder)
	   
import templutils as templutils
import snclasses as snstuff

lc_direc = os.getenv("SESNPATH") + './../Somayeh_contributions/main/ELASTICC_lc/'

SNTYPES = ['Ib','IIb','Ic','Ic-bl', 'Ibn']

bands = ['R','V','r','g','U','u','J','B','H','I','i','K','m2','w1','w2']
colorTypes = {'IIb':'FireBrick',
		   'Ib':'SteelBlue',
		   'Ic':'DarkGreen',
		   'Ic-bl':'DarkOrange',
		   'Ibn':'purple'}

def read_Ibc_tmpl(bands):

	templates = {}
	for b in bands:

		templates[b] = {}

		path = os.getenv("SESNPATH") + "maketemplates/outputs/Ibc_template_files/UberTemplate_%s.pkl" %\
								 (b + 'p' if b in ['u', 'r', 'i'] else b)
		templates_ = pkl.load(open(path, "rb"))

		templates[b] = templates_

		if np.nansum(templates_['med_smoothed']) == 0:
		   print(b + ' band has no Ibc templates.')
		   continue

	return templates




def read_gp_tmpl(bands, SNTYPE):
	tmpl = {}

	for b in bands:
		if b in ['u', 'r', 'i']:
			bb = b + 'p'  
		else: 
			bb = b

		tmpl[bb] = {}

		for SNTYPE in SNTYPES:
		   

			path = os.getenv("SESNPATH") + "maketemplates/outputs/GP_template_files/GPalltemplfit_%s_%s_V0.pkl"%(SNTYPE,bb)
			if os.path.isfile(path):
				tmpl_ = pkl.load(open(path, "rb"))
			else:
				print('No GP template for type %s in band %s'%(SNTYPE,bb))
				continue
			tmpl[bb][SNTYPE] = {}
			tmpl[bb][SNTYPE] = tmpl_

	return tmpl

def cal_metric(t,
			tmpl_med,
			tmpl_med_up,
			tmpl_med_low,
			x, y):


	t2 = t[(t <= x.max()) * (t >= x.min())]
	u = interpolate.splrep(x, y, s=0.5)
	y2 = interpolate.splev(t2, u)
	y2_min = np.nanmin(y2)
	y = y2_min - y
	y2 = y2_min - y2

	if sum(y2 - tmpl_med_low[(t <= x.max()) * (t >= x.min())])<0:
		IQR = np.abs(tmpl_med - tmpl_med_low)
		metric = np.abs(y2 - tmpl_med_low[(t <= x.max()) * (t >= x.min())])/\
				 IQR[(t <= x.max()) * (t >= x.min())]
	if sum(y2 - tmpl_med_up[(t <= x.max()) * (t >= x.min())])>0:
		IQR = np.abs(tmpl_med - tmpl_med_up)
		metric = np.abs(y2 - tmpl_med_up[(t <= x.max()) * (t >= x.min())])/\
				 IQR[(t <= x.max()) * (t >= x.min())]

	
	metric_min = np.nanmin(metric)
	metric_max = np.nanmax(metric)
	return x, y, t2, y2, metric_min, metric_max




if __name__ == "__main__":

    
	if len(sys.argv)<4 or len(sys.argv)>5:
		print("Usage: python single_SN_dist_from_tmpl.py <SN name> <tmpl type> <b=?,?,...> <plot>")
		sys.exit(1)

	sn_name = sys.argv[1]
	tmpl_type = sys.argv[2]
	if sys.argv[3].startswith('b='):
		bands = sys.argv[3].split('=')[1].split(',')
	else:
		print("Usage: python single_SN_dist_from_tmpl.py <SN name> <tmpl type> <b=?,?,...> <plot>")
		sys.exit(1)


	if len(sys.argv) == 5:
		plot = True
	else:
		plot = False


	thissn = snstuff.mysn(sn_name, addlit=True)
	lc, flux, dflux, snname = thissn.loadsn2(verbose=False)
	thissn.setphot()
	thissn.getphot()
	thissn.setphase()
	thissn.sortlc()
	input_file, snn = thissn.readinfofileall(verbose=False)
	SNTYPE = thissn.type

	for b in bands:
		print (b)

		xmin = thissn.photometry[b]['mjd'].min()

		x = thissn.photometry[b]['mjd']    
		y = thissn.photometry[b]['mag']
		yerr = thissn.photometry[b]['dmag']

		x = x - x[np.argmin(y)]

		idx = np.lexsort([x, y])
		out = np.sort(idx[np.unique(x[idx], return_index=1)[1]])
		x = x[out]
		y = y[out]
		# y = np.min(y)-y

		


		if tmpl_type == 'Ibc':
			tmpl = read_Ibc_tmpl(bands)
			t = templates[b]['phs']
			tmpl_med = -1*templates[b]['med_smoothed']
			tmpl_med_up = -1*templates[b]['pc25_smoothed']
			tmpl_med_low = -1*templates[b]['pc75_smoothed']
			tmpl_med = tmpl_med - np.nanmax(tmpl_med)
			tmpl_med_up = tmpl_med_up - np.nanmax(tmpl_med)
			tmpl_med_low = tmpl_med_low - np.nanmax(tmpl_med)
			t = t - t[np.nanargmax(tmpl_med)]

			x, y, t2, y2, metric_min, metric_max = cal_metric(t,
												tmpl_med,
												tmpl_med_up,
												tmpl_med_low,
												x, y)
			print ('Distance from %s template for SN %s in %s band= %.0f to %.0f x IQR'%(tmpl_type,
																  sn_name,
																  b, 
																  metric_min,
																  metric_max))
			if plot:

				label_templ = '%s template in %s band'%(tmpl_type, b)
				plt.figure()
				plt.plot(x, y, '.', label=sn_name + ' in %s band'%b)
				plt.plot(t2, y2, label=sn_name + ' interpolation in %s band'%b)
				plt.plot(t, tmpl_med, color = colorTypes[SNTYPE], label=label_templ)
				plt.fill_between(t, tmpl_med_low, tmpl_med_up, alpha=0.3, color = colorTypes[SNTYPE])
				plt.legend(loc = 4)

				plt.xlabel('Phase (time-peak)')
				plt.ylabel('Relative Magnitude')
				plt.text(0.5, 0.95, 
					    'Distance from template= %.0f to %.0f x IQR'%(metric_min,
															metric_max), 
					    transform=plt.gca().transAxes)
				plt.savefig(os.getenv("SESNPATH") + 
						  "maketemplates/more_plots/%s_%s_dist_metric_%s_tmpl"%(sn_name,
																	b,
																	tmpl_type))  

		elif tmpl_type == 'GP':
			tmpl = read_gp_tmpl(bands, SNTYPE)
			t = tmpl[b][SNTYPE]['t']
			tmpl_med = tmpl[b][SNTYPE]['rollingMedian']
			tmpl_med_up = tmpl[b][SNTYPE]['rollingPc75']
			tmpl_med_low = tmpl[b][SNTYPE]['rollingPc25']

			x, y, t2, y2, metric_min, metric_max = cal_metric(t,
												tmpl_med,
												tmpl_med_up,
												tmpl_med_low,
												x, y)
			print ('Distance from %s template for SN %s in %s band= %.0f to %.0f x IQR'%(tmpl_type,
																  sn_name,
																  b, 
																  metric_min,
																  metric_max))
			if plot:

				label_templ = '%s %s template in %s band'%(SNTYPE, tmpl_type, b)
				plt.figure()
				plt.plot(x, y, '.', label=sn_name + ' in %s band'%b)
				plt.plot(t2, y2, label=sn_name + ' interpolation in %s band'%b)
				plt.plot(t, tmpl_med, color = colorTypes[SNTYPE], label=label_templ)
				plt.fill_between(t, tmpl_med_low, tmpl_med_up, alpha=0.3, color = colorTypes[SNTYPE])
				plt.legend(loc = 4)

				plt.xlabel('Phase (time-peak)')
				plt.ylabel('Relative Magnitude')
				plt.text(0.5, 0.95, 
					    'Distance from template= %.0f to %.0f x IQR'%(metric_min,
															metric_max), 
					    transform=plt.gca().transAxes)
				plt.savefig(os.getenv("SESNPATH") + 
						  "maketemplates/more_plots/%s_%s_dist_metric_%s_tmpl"%(sn_name,
																	b,
																	tmpl_type))




