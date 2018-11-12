# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 11:38:47 2018

@author: brieu
"""
#%%
from sklearn import tree
from collections import namedtuple
import pandas as pd
from sklearn import preprocessing
#%%

Data = namedtuple('Data', ["outlook", "temperature", "humidity", "wind", "target"])

data = [Data("sunny", "hot", "high", "false", "no"), 
        Data("sunny", "hot", "high", "true", "no"),
        Data("overcast", "hot", "high", "false", "yes"),
        Data("rain", "mild", "high", "false", "yes"),
        Data("rain", "cool", "normal", "false", "yes"),
        Data("rain", "cool", "normal", "true", "no"),
        Data("overcast", "cool", "normal", "true", "yes"),
        Data("sunny", "mild", "high", "false", "no"),
        Data("sunny", "cool", "normal", "false", "yes"),
        Data("rain", "mild", "normal", "false", "yes"),
        Data("sunny", "mild", "normal", "true", "yes"),
        Data("overcast", "mild", "high", "true", "yes"),
        Data("overcast", "hot", "normal", "false", "yes"),
        Data("rain", "mild", "high", "true", "no")
]

data = pd.DataFrame(data)
data = data.apply(preprocessing.LabelEncoder().fit_transform)

#%%
X = data.iloc[:,:4]
Y =data.target
#%%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#%%
import graphviz 
dot_data = tree.export_graphviz(clf, out_file="tennis.dot") 
graph = graphviz.Source(dot_data) 
graph.render("data") 

#%%
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=list(X.columns.values),  
                         class_names="tennis",  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

