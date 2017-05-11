# svm-chirp-classification
Exploratory data analysis in Python on how SVM can be used to identify chirp in interferometric cross correlation measurements

## generate test data 
The script autocorrelation.py can be used to generate a test data set of interferometric crosscorrelation measurements. 
This script requires CUDA and PyCUDA.

To generate data just run ``` python3 autocorrelation.py ```

A folder called "data" is generated with the simulated data (0.txt, 1.txt,... nb_samples.txt) and a file called "labels.txt"
with the true class labels.
