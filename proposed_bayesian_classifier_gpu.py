import cupy as cp
import cudf
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import time
import pandas as pd




class MNBClassifier(ClassifierMixin, BaseEstimator):
    """ 
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
        """A reference implementation of a fitting function for a classifier.
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

        self.classes_ = cp.unique(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
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

        #X is test data
        #self.X_ is the training data
        
        #prior probabilities
        #access cth label using z[cth]
        #Z are class prior probabilities
        
        y_pred=cp.zeros((len(X),1))
        z = cp.zeros((cp.shape(self.classes_)[0],1 ))
        for i in range(len(self.X_)):
            z[int(self.y_[i])] += 1  

        z = z/cp.sum(z)
            

        
        R = cp.zeros((1,cp.shape(self.X_)[1]))
        for k in range(0, cp.shape(self.X_)[1]): 
            R[0][k] = cp.subtract(cp.amax(self.X_[:,k]) ,cp.amin(self.X_[:,k]))
        

        one = cp.ones((self.X_.shape[0],1))
        for i in range(len(X)):
            v = cp.zeros((cp.shape(self.classes_)[0],1))
            c = cp.zeros((cp.shape(self.classes_)[0],1))
            p = cp.zeros((cp.shape(self.classes_)[0],1))
            x_cp = cp.full((self.X_.shape[0],X[i].shape[0]),X[i])
            
            d_1 = cp.subtract(x_cp,self.X_)/R
            d_1 = cp.square(d_1)
            d_1 = cp.sqrt(cp.sum(d_1,axis = 1))
            d_1 = cp.add(1,d_1)
            d_1 = cp.power(d_1,-1*self.kappa)
            for k in range(0,cp.shape(self.classes_)[0]):
                loc_arr = cp.where(self.y_== self.classes_[k],True,False)
                v[k] = cp.sum(d_1[loc_arr])
                c[k] = cp.sum(loc_arr)
                
            p = z*v/c 
            y_pred[i] = cp.argmax(p)

      
        return y_pred

    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
