# Parkinson’s disease classification and the use of the tunable Q-factor wavelet transform
## Data description
This is a project done for the Scientific Programming course of the Maastricht university master Systems Biology and Bioinformatics.
The goal of this Project is to make a classifier system based on the [data](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) of [a study at the Department of Neurology in Cerrahpaşa Faculty of Medicine, Istanbul University](https://doi.org/10.1016/j.asoc.2018.10.022). The data used in this study were gathered from 188 patients with PD (107 men and 81 women). The control group comprises 64 healthy individuals (23 men and 41 women). During the data collection process, the microphone was set to 44.1 KHz, and following the physician’s examination, the sustained phonation of the vowel /a/ was collected from each subject with three repetitions. Various speech signal processing algorithms, including Time-Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform Features, Vocal Fold Features, and TWQT features, have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment.

The study tries to find the efficacy of using the tunable Q-factor wavelet transform in classification algorithms for Parkinson's Disease. It used a leave-one-subject-out (LOSO) cross-validation technique to validate the generalization ability of the classification models. For this project, I will set up a 20% test to validate the model I will make.

## Technical details
Analysis was done in Python 3.12.3.
Requirements will be installed in the first code block of the Datachecking.ipynb with pip. 
If it's still not working, run: ```pip install -r requirements.txt``` in the command prompt
The code looks for files in the same folder as the IPYNB files. Therefore, it is recommended that all files be saved in the same folder labeled with the project name.

## The use of these files.
The original CSV file is pd_speech_features.csv and is saved the data folder. This file will first be loaded into Datachecking.ipynb (First row of image below), where it is cleaned and saved as the outliercorrected_pd_speech_features.csv. This new file will be used in Classification.ipynb, where the actual classification algorithm is stored and the model validation is done.

![image](https://github.com/user-attachments/assets/5a8b7030-7703-421f-9584-995a84861319)  
Image 1: The workflow of the two ipynb files. The first row is described in Datachecking and the rest is done in Classification.ipynb
### PKL files.
The pickle files are saved variables in the data folder from the classification file since some steps take 10-15 minutes to process. To see the full extent of these steps, you can delete them before running the code.


