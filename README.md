Readme to accompany “Transfer Learning for Protein Structure Classification at Low Resolution’, an MSc dissertation prepared by Alexander Hudson, under the supervision of Professor Shaogang Gong and available at https://arxiv.org/abs/2008.04757

Please cite use of this code as below:

Alexander Hudson and Shaogang Gong. Transfer Learning for Protein Structure Classification at Low Resolution. arXiv, 2020. URL https://arxiv.org/abs/2008.04757

BibTex:
@article{Hudson2020transfer,
abstract = {},
archivePrefix = {arXiv},
arxivId = {1608.06993},
author = {Hudson, Alexander and Gong, Shaogang},
eprint = {2008.04757},
journal = {arXiv},
pmid = {},
title = {{Transfer Learning for Protein Structure Classification at Low Resolution}},
year = {2020},
url = {https://arxiv.org/abs/2008.04757},
}

1. OVERVIEW

This folder contains 6 executable files:
- Sourcing.py: creates input images for a given atom selection from input .pdb
-Traintest.py: identifies domain subsets for HR, LR or NMR subsets, reshapes matrices and saves to the working directory as .h5 files
-DenseNet121.py trains DenseNet121 on training data, and/or evaluates on test data
-Ensemble.py evaluates weighted ensemble of four DenseNet 121 models on test data
-GetPFP.py Computes homogeneity score and generates t-SNE plots of protein fingerprints (PFPs)
-GetHistory.py Prints training history and computes maximum training and validation accuracies from loss /accuracy over all epochs

The following additional files are included in the folder:
-AMBER.dat / AMBER.names forcefield specifications for ABS-PDB2PQR (see below)
-CATHdict.pickle: a dictionary mapping domain names to their associated CATH label and auxiliary information: PDB ID; chain; resolution; enzyme classification; experimental methodology; category and sequence)
-CATHonehot.pickle: a dictionary mapping CATH labels to a one hot index
- Dummy data files (random example instances):
	-	dompdb.zip (.pdb structure files)
	-	CATH/PQR.zip (.pqr structure files)
	- 	CATH/ca/raw.zip (.npz unprocessed images)
	-	CATH/ca/preprocessed/HR.zip (.npz HR processed images)
	-	CATH/ca/preprocessed/NMR.zip (.npz LR processed images)
	-	CATH/ca/preprocessed/LR.zip (.npz NMR processed images)
	- 	CATH/ca/preprocessed/HRCA_TRAINTEST.h5 (.h5 HR dataset)
	- 	CATH/ca/preprocessed/LRCA_TRAINTEST.h5 (.h5 LR dataset)
	- 	CATH/ca/preprocessed/NMRCA_TRAINTEST.h5 (.h5 NMR dataset)

** Available on request: 4 .ckpt files containing pre-trained DenseNet121 weights, required for Ensemble.py. Please contact a.o.hudson@se18.qmul.ac.uk for further information.

2. EXECUTABLE FILES

2.1 Dependencies

All executables were developed using MacOS 10.15.5 for Mac or Linux operating environments.

Package                Version
---------------------- ---------
absl-py                0.10.0
astunparse             1.6.3
biopython              1.77
cachetools             4.1.1
certifi                2020.6.20
chardet                3.0.4
cycler                 0.10.0
gast                   0.3.3
google-auth            1.20.1
google-auth-oauthlib   0.4.1
google-pasta           0.2.0
grpcio                 1.31.0
h5py                   2.10.0
idna                   2.10
joblib                 0.16.0
Keras                  2.4.3
Keras-Preprocessing    1.1.2
kiwisolver             1.2.0
Markdown               3.2.2
matplotlib             3.3.1
numpy                  1.18.5
oauthlib               3.1.0
opencv-python          4.4.0.42
opt-einsum             3.3.0
pandas                 1.1.1
Pillow                 7.2.0
pip                    20.2.2
ProDy                  1.10.11
propka                 3.3.0
protobuf               3.13.0
pyasn1                 0.4.8
pyasn1-modules         0.2.8
pyparsing              2.4.7
python-dateutil        2.8.1
pytz                   2020.1
PyYAML                 5.3.1
requests               2.24.0
requests-oauthlib      1.3.0
rsa                    4.6
scikit-learn           0.23.2
scipy                  1.4.1
setuptools             49.6.0
six                    1.15.0
sklearn                0.0
tensorboard            2.3.0
tensorboard-plugin-wit 1.7.0
tensorflow             2.3.0
tensorflow-estimator   2.3.0
termcolor              1.1.0
threadpoolctl          2.1.0
urllib3                1.25.10
Werkzeug               1.0.1
wheel                  0.35.1
wrapt                  1.12.1

2.2 Sourcing.py

This program takes as its input a folder of .pdb files, converts them into .pqr files, and extracts distance, NB and ANM representations before preprocessing and saving to the working directory.

Downloads and installation

CATH: For sourcing.py to run, you must have downloaded the S20 and S40 non-redundant datasets from the CATH repository at ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/, and merged into a single folder (‘dompdb’) in your working directory. Example .pdb files are included in dompdb.zip.

PDB2PQR: Requires user registration at http://www.poissonboltzmann.org/. Once registered, download the latest version of ABS from https://github.com/Electrostatics/apbs-pdb2pqr/releases. Take note of the location of the ABS/pdb2pqr/pdb2pqr.py file, which is required as an input to Sourcing.py. If the programme is unable to locate the AMBER forcefield parameters, ensure AMBER.dat and AMBER.names are copied from ABS/pdb2pqr/dat into the working directory.

To run the program, you must enter the desired atom selection (one of “ca”,”bb” or “heavy”), the location of the ABS pdb2pqr executable (usually in /Volumes/xxxx/Applications/ABS/pdb2pqr/pdb2pqr.py) and (optionally) the desired working directory (default = current working directory) and height x width for the reshaped image (default =255). For example: 

“python sourcing.py -a ca -p /Volumes/XXX/Applications/apbs-3.0.0/pdb2pqr/pdb2pqr.py -s 255”

Example .pqr files are included in CATH/PQR.zip.

2.3 Traintest.py

This program loads the preprocessed distance matrix, ANM and NB matrices from the specified folder, one-hot encodes the labels, and outputs an .h5 file as an input to the DenseNet model. 
To run the file, specify the input folder and (optionally) the proportion of test instances to set aside (default =0.1), for example:

“python Traintest.py -i CATH/ca/preprocessed/NMR -tr 1”

Setting the test ratio to 1.0 will assign all instances to the test set. 

2.4 DenseNet121.py

This program takes as its input preprocessed image files and associated labels, trains and/or evaluates DenseNet121 on the test set. 

To run the model:
	- Ensure you have the correct version of Tensorflow installed for your operating system and configuration. The model uses a distributed strategy that will identify available gpus when Tensorflow-gpu is installed.
	- Specify the input file (-i) and the number of epochs (-e). For evaluation only, select 0 epochs. 

- Optional parameters: 
	-b (batch size, default = 32); 
	-o (output layer size, default = 512); 
	-lr (learning rate, default = 0.001)
	-lf (load weights from pre-trained model); 
	-r (save accuracy to Results/results.csv); 
	-f1 (save F1 to Results/F1.csv); 
	-ch (save model checkpoints every 10 epochs);
	-fr (freeze all non-custom layers);
	-pf (save protein fingerprints to Features folder)

For example:

“ python DenseNet121.py -i CATH/ca/preprocessed/HRCA_TRAINTEST.h5 -e 10 -b 16 -lr 0.002 -o 128 -r True -f1 True -ch True -pf True”

2.5 Ensemble.py

This program takes as its input preprocessed image files and associated labels, uses a weighted ensemble of pre-trained DenseNet121 models to make predictions from the inputs, and evaluates performance of the ensemble.
To run the model, you must select the input file and (optionally) save performance in Results/f1.csv. You can also alter the run speed of the evaluation to suit the memory capacity of your system by selecting the batch-size (default = 32). 

For example:

“python Ensemble.py -i CATH/ca/preprocessed/HRCA_TRAINTEST.h5 -f1 True -b 16”

NB: For this model to run properly, you will need to request the learned model weights from a.o.hudson@se18.qmul.ac.uk.

2.6 GetPFP.py

Computes homogeneity scores for PFPs across C, A, T and H tasks and generates tSNE transformation plots.
To run the program, select the input .npz file from the Features folder. You can (optionally) select the number of tasks for which to compute homogeneity score, as computation of T and H tasks can be prohibitively slow for large datasets, and print t-SNE plots with the predicted as well as the true labels. 

For example:

“python InspectPFP.py -i  'Features/20200825_1423_densenet121_HRCA_TRAINTEST_PFP.npz' -n 4 -gp True”

2.7 GetHistory.py

Prints training plots and computes the maximum accuracy for training and validation sets across C, A, T and H tasks.
To run the program, select the input history file from the History folder, for example:

“python PlotHistory.py  -i History/20200825_1533_densenet121_HRCA_TRAINTEST.pickle”
	















