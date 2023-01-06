import numpy as np
import math
from numpy import linalg
from scipy import linalg
import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.gaussian_process.kernels import RBF
import scipy.optimize as sco
from sklearn.datasets import load_iris
from itertools import cycle, islice
import pandas as pd

# def read_file(file_name):
    
#     print("reading file",end=".....")
#     df=pd.read_csv(file_name)
#     x=[]
#     for i in range(1,len(df.columns)-1):
#         x.append(df["x"+str(i)])
#     seed = np.random.randint(0,100000)
#     print("done")
#     x=np.array(x)
#     np.random.seed(seed)
#     print("shuffling",end=".......")
#     for i in range(0,len(df.columns)-2):
#         np.random.shuffle(x[i])
#     np.random.seed(seed)
#     z=np.array(df["Target"])
#     np.random.shuffle(z)
#     print("done")
#     return x,z
# X,Y=read_file("csv_result-circle.csv")
# # print(np.shape(X),np.shape(Y))
# X_no_label=X[:,300:900]
# # print(Y)
# # print(np.shape(X_no_label))
# XTest=X[:,900:]
# Actual=Y[900:]
# X,Y=X[:,:300],Y[0:300]
# print(np.shape(X),np.shape(XTest),np.shape(XTest))
# print(XTest.T)
# # def multi_shuffle(x,y):
# #     temp = list(zip(x, y))
# #     random.shuffle(temp)
# #     res1, res2 = zip(*temp)
# #     return(x(res1),y(res2))

def read_file(file_name):
    print("reding file",end="...")
    df=pd.read_csv(file_name)
    x=[]
    for i in range(1,len(df.columns)-1):
        x.append(df["V"+str(i)])
    print("done")
    print("shuffling",end="...")
    x=np.array(x)
    seed = np.random.randint(0,100000)
    np.random.seed(seed)
    for i in range(0,len(df.columns)-2):
        np.random.shuffle(x[i])
    np.random.seed(seed)
    z=np.array(df["Target"])
    np.random.shuffle(z)
    print("done")
    return x,z
X,Y=read_file("creditcard.csv")
# print(np.shape(X),np.shape(Y))
X_no_label=X[:,800:2000]
# print(Y)
# print(np.shape(X_no_label))
XTest=X[:,2000:]
print(len(XTest.T))
Actual=Y[2000:]
print(len(Actual))
X,Y=X[:,:800],Y[0:800]
# print(Y[:492])
# print(np.shape(X),np.shape(XTest),np.shape(XTest))
# print(XTest.T)
#============================================================================
#================================ 1. ALGORITHM
def polynomial_kernel(x,y):
    return (np.dot(x, y.T)) ** 2

def rbf(x,y):
    sig=1
    return sklearn.metrics.pairwise.rbf_kernel(x, Y=y, gamma=None)

class LapSVM(object):

    def __init__(self, n_neighbors, kernel, lambda_k, lambda_u):
        """
        Laplacian Support Vector Machines
        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        """
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
    

    def fit(self, X, X_no_label, Y):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        # Storing parameters
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        Y = np.diag(Y)
        # Memory optimization
        del X_no_label
        
        # Building adjacency matrix from the knn graph
        print('Computing adjacent matrix', end='...')
        W = kneighbors_graph(self.X, self.n_neighbors, mode='connectivity')
        W = (((W + W.T) > 0) * 1)
        print('done')

        # Computing Graph Laplacian
        print('Computing laplacian graph', end='...')
        L = np.diag(W.sum(axis=0)) - W
        print('done')

        # Computing K with k(i,j) = kernel(i, j)
        print('Computing kernel matrix', end='...')
        K = self.kernel(self.X,self.X)
        print('done')
        print(K.shape)

        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        ###########################################################################
        
        # Computing "almost" alpha
        print('Inverting matrix', end='...')
        
        #error line #################################################################
        
        f=2 * self.lambda_k * np.identity(l + u) + ((2 * self.lambda_u) / (l + u) ** 2) * L.dot(K)
        almost_alpha = linalg.inv(f).dot(J.T).dot(Y)
        #error end #################################################################
        # Computing Q
        global Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        print('done')
        
        # Memory optimization
        del W, L, K, J
        
        # Solving beta using scypy optimize function
        
        print('Solving beta', end='...')
        
        e = np.ones(l)
        q = -e
        
        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)
        
        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))
        
        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]
        
        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))
        
        def constraint_grad(beta):
            return np.diag(Y)
        
        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}
        
        # ===== Solving =====
        x0 = np.zeros(l)
        
        beta_hat = sco.minimize(objective_func, x0, jac=objective_grad, \
                                constraints=cons, bounds=bounds, method='L-BFGS-B')['x']
        print('done')
        
        # Computing final alpha
        print('Computing alpha', end='...')
        self.alpha = almost_alpha.dot(beta_hat)
        print('done')
        
        del almost_alpha, Q
        
        ###########################################################################
        
        # Finding optimal decision boundary b using labeled data
        new_K = polynomial_kernel(self.X, X)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == np.diag(Y)) / len(predictions))
        
        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
    

    def predict(self, Xtest):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """

        # Computing K_new for X
        new_K = self.kernel(self.X, Xtest)
        print(new_K.shape)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        predictions = np.array((f > self.b) * 1)
        return predictions
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        print(len(predictions))
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))
        
    
c=LapSVM(3, rbf, 0.4, 1.2)
c.fit(X.T, X_no_label.T, Y)
c.accuracy(XTest.T, Actual)


