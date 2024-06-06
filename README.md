Stripped envelope supernovae (SESNe) are a family of supernovae for which progenitor and explosion channels are still poorly constrained. While their spectroscopic classification scheme is clear, photometric differences between classes also remain elusive. We know that they originate from massive stars that lost their envelopes of Hydrogen and Helium. A detailed mapping of each SESN subtype (IIb, Ib, Ic, Ib-BL)  to its stellar progenitor remain uncertain, as are the mechanism for envelope loss, and the relationships among subtypes and with long duration Gamma Ray Bursts (GRB) that are occasionally seen in conjunctions with Ic-BL.
Photometric surveys, like Vera C. Rubin’s Legacy Survey of Space and Time, discovering tens of thousands of transients each night, offer an incredible opportunity to improve our knowledge of these supernova subtypes, but increasing emphasis has to be placed on photometric classification and characterization, as spectroscopic resources will only enable follow up of a small fraction of observed transients. We have generated data-driven photometric templates for SESNe subtypes using machine learning techniques (Gaussian processes) and a comprehensive  multi-survey dataset of all open-access data from the Open Supernova Catalog. We assess the photometric diversity of SESNe,
among and within subtypes, setting the stage for studies aimed at relating the explosion properties to their stellar progenitors.  Our templates can help evaluate the current photometric simulations used to develop classification methods and identify peculiar behavior.


This repository contains the codes, data, plots, tables and results of the paper "Multi-filter UV to NIR Data-driven Light Curve Templates for Stripped Envelope Supernovae" that is submitted to ApJS and the preprint can be found at https://arxiv.org/abs/2405.01672

The final Ibc and GP templates can be found in the folder "share_tmpls".



Instructions for running the codes:

This repo works with a few utility codes which are stored in https://github.com/fedhere/SESNCfAlib and https://github.com/fedhere/fedsastroutils.


First, set the following environmental variables (assuming bash syntax):

export SESNPATH = "path_to_main_directory/GP_templates_SES/"
export SESNCFAlib = "path_to_library/SESNCFAlib"
export UTILPATH = "path_to_randomutils/fedastroutils/"


The data is downloaded from the OSNC and is sotered in the "literature" folder in CfA SN data format. Any added SN photometry should be saved in this format and saved in the "literature" folder. An example of the few first lines of the data files are shown below:

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


For reproducing the process of generating the templates for example if new data is added, follow the instructions below:

To be completed soon...



