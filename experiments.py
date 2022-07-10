import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
import time
import matplotlib.pyplot as plt
import math
from os import listdir
from os.path import isfile, join
from collections import Counter
import operator
import numpy as np
from sklearn.inspection import permutation_importance
import csv
import itertools
import random
import warnings as warnings
import matplotlib.pyplot as plt
import random
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import RepeatedKFold
warnings.filterwarnings('ignore')


class TimeError(Exception):
    """Custom exception class for timer"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        return(elapsed_time)
#         print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        


class MNBClassifier(ClassifierMixin, BaseEstimator):
    """ 
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, kappa=20):
        self.kappa = kappa

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ 
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        #X is test data
        #self.X_ is the training data
        
        #prior probabilities
        #access cth label using z[cth]
        #Z are class prior probabilities
        y_pred=[]
        z = np.zeros((np.shape(self.classes_)[0],1 ))
        for i in range(len(self.X_)):
            z[int(self.y_[i])] += 1  

        z = z/sum(z)
            

        
        R = np.zeros((np.shape(self.X_)[1],1))
        for k in range(0, np.shape(self.X_)[1]): 
            R[k] = np.amax(self.X_[:,k]) - np.amin(self.X_[:,k])
        

        
        for i in range(len(X)):
            v = np.zeros((np.shape(self.classes_)[0],1))
            c = np.zeros((np.shape(self.classes_)[0],1))
            p = np.zeros((np.shape(self.classes_)[0],1))
            for j in range(len(self.X_)):
                d=0
                for k in range(0,np.shape(self.X_)[1]):
                    d += ((X[i,k] - self.X_[j,k])/R[k])**2
                v[int(self.y_[j])] += 1/((1 + math.sqrt(d))**self.kappa)
                c[int(self.y_[j])] += 1
            p = z*v/c

            y_pred.append(np.argmax(p))
        
        return y_pred

    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
def load_data(data_file):
    df = pd.read_csv('{}'.format(data_file),header=None)
#     df = pd.read_csv('datasets3/iris.data.csv'.format(data_file),header=None)
    
    #UCI has ? as missing data
    df = df[~df.eq('?').any(1)]
    df.dropna(axis = 1, how ='all', inplace = True)
    df.dropna(axis = 0, how ='all', inplace = True)
    df.to_csv('cleaned_frame.csv'.format(data_file),index=False)
    
    
    
    #reread for appropriate column dtypes
    df = pd.read_csv('cleaned_frame.csv',header=0)
    result_summary[data_file]["data_type"]=Counter(df.dtypes.tolist())
    
    #convert Y to integer type
    df[df.columns[-1]]=df[df.columns[-1]].astype('str')
    df[df.columns[-1]]=df[df.columns[-1]].str.strip()
    df[df.columns[-1]]=df[df.columns[-1]].astype('category')
    df[df.columns[-1]]=df[df.columns[-1]].cat.codes
    
    #get the columns that needed to be label encoded
    conversion_idx=[]
#     print(df.dtypes.values)
    for idx,d_type in enumerate(df.dtypes.values):
        if "object" == d_type:
            conversion_idx.append(idx)

    
    
    #clean categories of white space
    for idx in conversion_idx:
        df.iloc[:,idx]=df.iloc[:,idx].astype('str')
        df.iloc[:,idx]=df.iloc[:,idx].str.strip()
        df.iloc[:,idx]=df.iloc[:,idx].astype('category')
    
    #encode strings as numbers
    for idx in conversion_idx:
        labels = df.iloc[:,idx].astype('category').cat.categories.tolist()
        replace_map_comp = {idx : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
        df.iloc[:,idx].replace(replace_map_comp[idx], inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    return df

def feature_selection(model,X,y,kf,num_CV,acc,removed_col):

    number_of_repeats = 10
    rkf = RepeatedKFold(n_splits=num_CV, n_repeats=number_of_repeats, random_state=6920)
    #Use state to break while loop if acc decreases. 
    state = True
    important_features={}
    highest_score = [0,acc]
    print("new_loop_initial_scores {}".format(highest_score))
    print(X.columns)
    for col in list(X.columns.values):
        important_features[col]=0
        
    for train_index , test_index in rkf.split(X,y):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
        
        
        model = model.fit(X_train, y_train)
#         r = permutation_importance(model, X_test, y_test,
#                            n_repeats=1,
#                             scoring= 'accuracy')
        r = permutation_importance(model, X_test, y_test)
        imp_features=[]
#         print("importance means {}".format(r.importances_mean))
        for imp_idx,i in enumerate(r.importances_mean):
            important_features[list(X.columns.values)[imp_idx]] += i

        
        y_pred_t = model.fit(X_train, y_train).predict(X_test)
        highest_score[0] += (1-((y_test != y_pred_t).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)
    
    print("Score for this run {}".format(highest_score[0]))
    sorted_features = {k: v for k, v in sorted(important_features.items(), key=lambda item: item[1],reverse = True)}
    if highest_score[0]>highest_score[1]:
                state =True
                print("score was higher")
                print("features and importance in dict{}".format(sorted_features))
                cols = list(sorted_features.keys())
                print("all featues for current iteration{}".format(cols))
                if len(cols)>1:
                    removed_col = cols.pop()
                print("featues for next iteration {}".format(cols))
    
 
    else:
                state =False
                print("ending while loop")
                print("score was lower")
                print("features and importance in dict{}".format(sorted_features))
                cols = list(sorted_features.keys())
                print("all featues for this iteration {}".format(cols))
                cols.append(removed_col)
                print("features selected for evaluation {}".format(cols))
    


    return (max(highest_score[0],highest_score[1]),cols,state,removed_col)



def naive_bayes_analysis(data_file,result_summary={},final_features=[0],kappa=1):
    
    #set up data collection 


    result_summary[data_file]={}
    result_summary[data_file]["data_file"]=data_file
    result_summary[data_file]["Gaussian"]=0
    result_summary[data_file]["Laplacian"]=0
    result_summary[data_file]["kNN"]=0
    result_summary[data_file]["kNN_20"]=0
    result_summary[data_file]["MNB_optimal"]=0
    result_summary[data_file]["MNB_optimal_kappa"]=[]
    result_summary[data_file]["MNB_20"] = 0
    result_summary[data_file]["MNB_60"] = 0
    result_summary[data_file]["MNB_time"]=0
    result_summary[data_file]["Laplace_time"]=0
    result_summary[data_file]["MNB_optimal_kappa"].append(kappa)
    
    #feature selection
    number_of_repeats = 10
    num_CV= 10
    acc = 0
    removed_column = 10000
    kf = KFold(num_CV,shuffle=True,random_state=6920)
    rkf = RepeatedKFold(n_splits=num_CV, n_repeats=number_of_repeats, random_state=6920)
    #data cleaning happens in load data
    df = load_data(data_file)
    
    #separate into x|Y
    X = df[df.columns[0:len(df.columns)-1]]
    y= df[df.columns[-1]]
    print(X.columns)
    

    
    
    result_summary[data_file]["selected_features"]=final_features
    X = df[df.columns[0:len(df.columns)-1]].loc[:,final_features]
            
            
    labels = y.unique()
    
    
    # determine hyperparameters

    params_grid={'n_neighbors':range(1,100,1)}
    search_KNN = GridSearchCV(KNeighborsClassifier(), param_grid=params_grid,
                              n_jobs=-1,cv=kf,scoring='accuracy').fit(X, y)

    

    print("{}-NN".format(search_KNN.best_params_['n_neighbors']))



    t = Timer()

    for train_index , test_index in rkf.split(X,y):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y.iloc[train_index] , y.iloc[test_index]






        #KNN

        neigh = KNeighborsClassifier(n_neighbors=search_KNN.best_params_['n_neighbors'])
        y_kNN_pred = neigh.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["kNN"]+=(1-((y_test != y_kNN_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)


        #KNN_20

        neigh = KNeighborsClassifier(20)
        y_kNN_pred = neigh.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["kNN_20"]+=(1-((y_test != y_kNN_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)


        #Gaussian Naive Bayes

        t.start()
        gnb = GaussianNB()
        y_GNB_pred = gnb.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["Gaussian"]+=(1-((y_test != y_GNB_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)
        t.stop()

#       
        # proposed approximated bayesian classification


        t.start()
        MNB = MNBClassifier(kappa)
        y_MNB_pred = MNB.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["MNB_time"]=t.stop()/num_CV
        result_summary[data_file]["MNB_optimal"]+=(1-((y_test != y_MNB_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)


        MNB = MNBClassifier(20)
        y_MNB_pred = MNB.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["MNB_20"]+=(1-((y_test != y_MNB_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)

        MNB = MNBClassifier(60)
        y_MNB_pred = MNB.fit(X_train, y_train).predict(X_test)
        result_summary[data_file]["MNB_60"]+=(1-((y_test != y_MNB_pred).sum()/(X_test.shape[0])))/(num_CV*number_of_repeats)





#             Laplacian
        prob={}
        y_MNB = []
        t.start()
        Z={}
        for label in labels:
            Z[label] = (y_train.values == label).sum()/y.shape[0]
        for i,row in X_test.iterrows():
            row = row.tolist()
            class_sample_size = 0
            zero_freq=0

            #create prob dict
            for label in labels:
                prob[label] = 0

            for k, col in enumerate(X_train.columns):
                #casting to avoid iloc str error
                col = int(col)
                hosein_estimate = 1
                v = 10
                class_sample_size={}
                x_col = X_train.iloc[:,k]


                occ_score={}
                for label in labels:
                    occ_score[label] = 0
                    class_sample_size[label] = x_col[y_train==label].shape[0]

                for label in labels:
                    v = (x_col[y_train==label].values == row[k]).sum()

                    if (v == 0):
                        occ_score[label] = 1
                    else:
                        occ_score[label] = v + 1




                for label in labels:
                    prob[label] += math.log(occ_score[label]/(class_sample_size[label]+len(x_col.unique())))


            for label in labels:
                prob[label] += math.log(Z[label])



            y_MNB.append(max(prob.items(), key=operator.itemgetter(1))[0])
        result_summary[data_file]["Laplacian"]+=(1-((y_test != y_MNB).sum()/(y_test.shape[0])))/(num_CV*number_of_repeats)
        result_summary[data_file]["Laplace_time"]=t.stop()/num_CV



    winner_optimal = [result_summary[data_file]["Gaussian"],
              result_summary[data_file]["Laplacian"],
              result_summary[data_file]["kNN"],
              result_summary[data_file]["MNB_optimal"]]
    
    winner_fixed = [result_summary[data_file]["Gaussian"],
          result_summary[data_file]["Laplacian"],
          result_summary[data_file]["kNN_20"],
          result_summary[data_file]["MNB_20"],
          result_summary[data_file]["MNB_60"]]
    
    result_summary[data_file]['Winner'] = winner_optimal.index(max(winner_optimal))
    result_summary[data_file]['Winner_fixed'] = winner_fixed.index(max(winner_fixed))
    print(result_summary[data_file])
    return result_summary
    


result_summary={}
    







LOCATION='FILE_LOCATION'
files = ['FILE_NAME']


for data_file in files:  
    print(data_file)
    
    naive_bayes_analysis('{}/{}'.format(LOCATION,data_file),result_summary,
                         final_features =  ['10', '1', '4', '6', '2', '9', '8', '0'],
     
                         kappa=42)

#     print(result_summary)
for result in result_summary:
    print(result)

keys=None
for result in result_summary:
    keys = result_summary[result].keys()


with open('MNB_bank', 'w') as f: 
    w = csv.DictWriter(f, keys)
    w.writeheader
    for result in result_summary:
        w.writerow(result_summary[result])


