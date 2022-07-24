# BAYES-CLASSIFICATION-USING-CONDITIONAL-PROBABILITY-APPROXIMATIONS


We have provided the datasets used for testing in the zip file named datasets.zip
These datasets were acquired from the UCI Machine Learning Repository.
https://archive.ics.uci.edu/ml/datasets.php

The citations to each dataset are listed below. 



experiments.py and experiments.ipynb are the same and was used to produce the results in the paper results. 
We have provided experiments.ipynb and the datasets for convenience in reproducing our resutls. 


To use experiments.ipynb


1/ Download datasets.zip and extract


2/ Set the location of the folder in line 461
LOCATION ='./datasets' 


3/ Select the you want to run in line 462
FILE = 'file_name' 


4/ Experiments.ipynb allows you to select the features to be evaluated, this can be done in line 469
final_features = "string_list_of_features' eg. ['0','1','5']


5/ Finally youcan set the parameter kappa for the model in line 471
kappa = INT eg. 42


The files proposed_bayesian_classifier_cpu.py and proposed_bayesian_classifier_gpu.py are stand-alone cpu and gpu implenetations of the proposed classifier respectively.

When using proposed_bayesian_classifier_cpu.py or proposed_bayesian_classifier_gpu.py, the implementation resembles that of scikit-learn methods. 

PC = ProposedClassifier(kappa)

y_PC_pred = PC.fit(X_train, y_train).predict(X_test)
