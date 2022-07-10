
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class ProposedClassifier(ClassifierMixin, BaseEstimator):
    """ 
    Attributes
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
#         X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = np.unique(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ 
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
#         # Check is fit had been called
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
