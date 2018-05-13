# -*- coding: utf-8 -*-
"""
Created on Thu May 10 18:55:48 2018

@author: brieu
"""
#%%
#les imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import linalg
#from scipy.sparse import issparse
from sklearn.utils.extmath import svd_flip
#%%

class PCA_homemade():
    
    def __init__(self,n_components=None,components_=None):
       self.n_components = n_components

    def fit(self,X):
        """
        On soustrait la moyenne a chaque observations
        """
        X_values=X.values
        self.mean = X_values.mean(axis=0)
        X_values-=self.mean

        """
        Calcule de la matrice de covariance
        """
        self.cov = np.cov(X_values.T)
        """
        calcule des vecteurs prores et valeurs propres
        """
        U, S, V = linalg.svd(self.cov, full_matrices=False)
        
        # flip eigenvectors' sign to enforce deterministic output
        U, V = svd_flip(U, V)
        self.eig_pairs_bis = [(np.abs(S[i]), V[0:self.n_components,i]) for i in range(self.n_components)]
        self.eig_pairs_bis.sort(key=lambda x: x[0], reverse=True)
        tot = sum(S)
        S_copy = S[0:self.n_components]
        var_exp = [(i / tot)*100 for i in sorted(S_copy, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)
        with plt.style.context('seaborn-whitegrid'):
           plt.figure(figsize=(6, 4))    
           plt.bar(range(self.n_components), var_exp, alpha=0.5, align='center',label='individual explained variance')
           plt.step(range(self.n_components), cum_var_exp, where='mid',label='cumulative explained variance')
           plt.ylabel('Explained variance ratio')
           plt.xlabel('Principal components')
           plt.legend(loc='best')
           plt.tight_layout()
        return U,S,V
      
    def transform(self,X):
        """
        Create the feature vector
        """
        list_of_eigvecs = np.array([i[1] for i in self.eig_pairs_bis])
        vector_of_eigvecs = list_of_eigvecs.T
        """
        Recontruire le nouveau jeu de donnees ( pca.transform)
        """
        rep=np.dot(vector_of_eigvecs,X[:,:self.n_components].T)
        return rep.T

#%%
"""
1 get the data in a dataframe
"""
dataset = pd.read_csv("data.csv", sep=";")
del dataset["Unnamed: 0"]

list(dataset.columns.values)

#clean the Sales column
dataset.Sales=dataset.Sales.apply(lambda x: x.replace(',','.'))
dataset = dataset.astype(float)

#separate Sales target 
dataset.Sales = dataset.Sales.astype(float)
target = dataset.Sales
del dataset["Sales"]
#%%

pca_homemade=PCA_homemade(n_components=5)
U,S,V = pca_homemade.fit(dataset)
transformed_data_handmade = pca_homemade.transform(dataset.values)

#%%
"""
comparaison avec la librairie
"""
pca_sklearn = PCA(n_components=5)
pca_sklearn.fit(dataset.values)
#print(pca.explained_variance_ratio_)  
transformed_data=pca_sklearn.transform(dataset.values)


