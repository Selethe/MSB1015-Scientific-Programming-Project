# Parkinson’s disease classification and the use of the tunable Q-factor wavelet transform
## Data description
The goal of this Project is to make a classifier system based on the [data](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) of [a study at the Department of Neurology in Cerrahpaşa Faculty of Medicine, Istanbul University](https://doi.org/10.1016/j.asoc.2018.10.022). The data used in this study were gathered from 188 patients with PD (107 men and 81 women). The control group comprises 64 healthy individuals (23 men and 41 women). During the data collection process, the microphone was set to 44.1 KHz, and following the physician’s examination, the sustained phonation of the vowel /a/ was collected from each subject with three repetitions. Various speech signal processing algorithms, including Time-Frequency Features, Mel Frequency Cepstral Coefficients (MFCCs), Wavelet Transform Features, Vocal Fold Features, and TWQT features, have been applied to the speech recordings of Parkinson's Disease (PD) patients to extract clinically useful information for PD assessment.

The study tries to find the efficacy of using the tunable Q-factor wavelet transform in classification algorithms for Parkinson's Disease. The study used a leave-one-subject-out (LOSO) cross-validation technique to validate the generalization ability of the classification models. For this project, I will set up an 20% test set to validate the model I will be making.

## The use of these files.
The original CSV file is the pd_speech_features.csv. This file will be first loaded Datachecking.ipynb, where it is cleaned and saved as the outliercorrected_pd_speech_features.csv. This new file will be used in the Classification.ipynb, where the actual classification algorithm is stored and the model validation is done.

### PKL files.
The pickle files are saved variables from the classification file since some steps take 10-15 minutes to process. To see the full extend of these steps delete them before running the code.
