This repository contains the codes, data, plots, tables and results of the paper "Multi-filter UV to NIR Data-driven Light Curve Templates for Stripped Envelope Supernovae" published in ApJS at https://iopscience.iop.org/article/10.3847/1538-4365/ad7eaa.

------------------------------------------------------------------------------------------------------------------------------------------------

The final Ibc and GP templates can be found in this address folder ["maketemplates/share_tmpls/all_templates.pkl"](https://github.com/fedhere/GPSNtempl/blob/main/maketemplates/share_tmpls/all_templates.pkl).
The instruction on how to read and use the final templates can be found here: ["maketemplates/share_tmpls/Reading Ibc and GP templates.ipynb"](https://github.com/fedhere/GPSNtempl/blob/main/maketemplates/share_tmpls/Reading%20Ibc%20and%20GP%20templates.ipynb).

------------------------------------------------------------------------------------------------------------------------------------------------

Instructions for running the codes to regenerate the templates:

This repo works with a few utility codes, which are stored in https://github.com/fedhere/SESNCfAlib and https://github.com/fedhere/fedsastroutils.


First, set the following environmental variables (assuming bash syntax):

export SESNPATH = "path_to_main_directory/GPSNtempl/maketemplates/"
export SESNCFAlib = "path_to_library/SESNCFAlib"
export UTILPATH = "path_to_randomutils/fedastroutils/"


The data is downloaded from the OSNC and is stored in the "literature" folder in CfA SN data format. Any added SN photometry should be saved in this format and saved in the "literature" folder. An example of the few first lines of the data files are shown below:

Ul 53734.43359375 nan nan 0.15399999916553497 19.402000427246094
Ul 53736.46875 nan nan 0.18400000035762787 19.582000732421875
Ul 53737.44140625 nan nan 0.25600001215934753 19.552000045776367
Ul 53739.51171875 nan nan 0.3310000002384186 19.94300079345703
Bl 53733.3828125 nan nan 0.03400000184774399 18.895000457763672
Bl 53734.4296875 nan nan 0.03700000047683716 18.937999725341797
Bl 53736.46484375 nan nan 0.039000000804662704 19.045000076293945
Bl 53737.43359375 nan nan 0.041999999433755875 19.10700035095215
Bl 53739.50390625 nan nan 0.06499999761581421 19.302000045776367
Bl 53742.45703125 nan nan 0.07100000232458115 19.47800064086914
Bl 53743.47265625 nan nan 0.0860000029206276 19.608999252319336

Columns:

column 1: PHOTCODE (filter and phot system identifier)

column 2: MJD

column 3: ?

column 4: ?

column 5: ERROR

column 6: MAG (Natural system)


For reproducing the process of generating the templates, for example, if new data is added, follow the instructions below:

1) After converting your light curves in the format above, put it in the folder "literaturedata/phot" if it has optical photometry and in "literaturedata/nirphot" if it has NIR and UV photometry. Make sure the format of the file name is the same as the existing files. For reference, you can use the notebook at "/maketemplates/add_any_new_SN_lc.py" to find an example of how format conversion is done.
2) The next step is to regenerate the Ibc templates for each band. This template will be used as an average template to subtract from individual light curves so we can fit the residual with GPs. Use the code at "maketemplates/Ibc_templates.py" to generate the Ibc templates. If you give the band name as an argument, it will only generate a Ibc template for that band. If no bands are given, it will generate Ibc templates for all bands.
3) Here, we run the code at "maketemplates/GPfitstrippedLCVS.py" to generate GP fits to the light curves in our sample. We first do the fits while optimizing for the best hyperparameters. So, at this step, you should put FITGP=True inside the code. You have the option to do it for one SN or all of the SNe in your sample. If no arguments are given, this will be done for all SNe. If the name of the SN is given as an argument, only the fit for that SN in all of its available bands will be done. You also have the option to do the fits for one subtype at a time. To do so, inside the code, put perType = True and put tp = your selected subtype. All of the best-fit hyperparameters will be saved in "maketemplates/outputs/GP_template_files/all_SNe_data_and_GP_fits.pkl".
4) Next, we analyze the GP fits visually, using a user interface. Running the code at "maketemplates/vetGP.py", plots of GP fits will be shown to you. If you find them acceptable, you type "Y", and if not, you type "N". The results will be saved in a csv file at "maketemplates/good_GP_OPT2.csv".
5) Next, we review the distribution of the hyperparameters per subtype of SESNe for the ones that received "Y" when they were being vetted. The Jupiter notebook at "maketemplates/get_final_gp_hyperparams.py" can be used for this purpose.
6) To be completed.... 


