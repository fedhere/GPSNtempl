import numpy as np
from Functions import read_snana_fits
from tqdm import tqdm

def read_lc(directory, num = 40):

	sne = []
	for c in tqdm(range(num)):
		
		n = c + 1
		
		if n <10:
			head = directory + 'ELASTICC_TRAIN_NONIaMODEL0-000'+ str(n) +'_HEAD.FITS'
			phot = directory + 'ELASTICC_TRAIN_NONIaMODEL0-000'+ str(n) +'_PHOT.FITS'
		else:
			head = directory + 'ELASTICC_TRAIN_NONIaMODEL0-00'+ str(n) +'_HEAD.FITS'
			phot = directory + 'ELASTICC_TRAIN_NONIaMODEL0-00'+ str(n) +'_PHOT.FITS'
			


		data = read_snana_fits(head, phot, n = None)
		sne.append(data)

	return sne

def select_lc(sne, max_dist = 5, high_SN_ratio_threshold = 10, least_num_high_SN = 5, ref_band = 'r ', redshift_threshold = 0.2):

	selected_lc = {}

	b = ref_band
	count = 0

	for c,data in enumerate(sne):

		

		# print(len(data))
		
		for i in range(len(data)):
			
			if data[i].meta['REDSHIFT_HELIO'] > redshift_threshold:
	#             print (i, ': Redshift is larger than 0.2!')
				continue
			
			

			t = np.asarray(data[i]['MJD'][np.asarray(list(data[i]['BAND'])) == b])
			f = np.asarray(data[i]['FLUXCAL'][np.asarray(list(data[i]['BAND'])) == b])
			ferr = np.asarray(data[i]['FLUXCALERR'][np.asarray(list(data[i]['BAND'])) == b]) 
			SN = f/ferr

			if (np.argmax(f) == 0) or (np.argmax(f) == -1):
				continue
				
			try:
				if ((t[np.argmax(f)]-t[np.argmax(f)-1]) > max_dist) or ((t[np.argmax(f)+1]-t[np.argmax(f)]) > max_dist):
					continue
			except:
				continue
			

			if np.sum(SN > high_SN_ratio_threshold) > least_num_high_SN:
				
				selected_lc[str(count)] = {}
				selected_lc[str(count)]['u']= {}
				selected_lc[str(count)]['g']= {}
				selected_lc[str(count)]['r']= {}
				selected_lc[str(count)]['i']= {}

				
				
				selected_lc[str(count)]['r']['t'] = t
				selected_lc[str(count)]['r']['f'] = f
				selected_lc[str(count)]['r']['ferr'] = ferr
				
				
				t = np.asarray(data[i]['MJD'][np.asarray(list(data[i]['BAND'])) == 'u '])
				f = np.asarray(data[i]['FLUXCAL'][np.asarray(list(data[i]['BAND'])) == 'u '])
				ferr = np.asarray(data[i]['FLUXCALERR'][np.asarray(list(data[i]['BAND'])) == 'u '])
				
				selected_lc[str(count)]['u']['t'] = t
				selected_lc[str(count)]['u']['f'] = f
				selected_lc[str(count)]['u']['ferr'] = ferr
				
				t = np.asarray(data[i]['MJD'][np.asarray(list(data[i]['BAND'])) == 'g '])
				f = np.asarray(data[i]['FLUXCAL'][np.asarray(list(data[i]['BAND'])) == 'g '])
				ferr = np.asarray(data[i]['FLUXCALERR'][np.asarray(list(data[i]['BAND'])) == 'g '])
				
				selected_lc[str(count)]['g']['t'] = t
				selected_lc[str(count)]['g']['f'] = f
				selected_lc[str(count)]['g']['ferr'] = ferr
				
				t = np.asarray(data[i]['MJD'][np.asarray(list(data[i]['BAND'])) == 'i '])
				f = np.asarray(data[i]['FLUXCAL'][np.asarray(list(data[i]['BAND'])) == 'i '])
				ferr = np.asarray(data[i]['FLUXCALERR'][np.asarray(list(data[i]['BAND'])) == 'i '])
				
				selected_lc[str(count)]['i']['t'] = t
				selected_lc[str(count)]['i']['f'] = f
				selected_lc[str(count)]['i']['ferr'] = ferr
				
				count = count + 1

	return selected_lc