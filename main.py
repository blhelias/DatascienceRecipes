# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 16:19:19 2018

@author: brieu
"""
import logging 
from regression.linear_regression_brieuc import LinearRegressionBrieuc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def compute_results(x,y,b,m,title,color):
    """
    param:
        colonne X
        colonne Y
        coeff M (pente)
        B (intercept)
        titre du graph
        couleur de la droite de regression
    """


    print(" {} : b = {} ; m = {}".format(title,b,m))
    plt.figure(title)
    plt.scatter(x,y,c=color)
    ablines_values = [m * i + b for i in x]
    plt.plot(x,ablines_values)
    plt.title(title)

    plt.show()
    
"""
1 - chargement du dataset
"""
logging.warning('[INFO] chargement du dataset')
data = pd.read_table("data/data.txt",sep="\t",header=None)
x= np.array(data[0])
y= np.array(data[1])

"""
2 - regression descente de gradient
"""
logging.warning('[INFO] entrainement regression')
lin_reg_gradient_descent = LinearRegressionBrieuc(gradient_descent=True)

# train liner regression model gradient descent
b,m = lin_reg_gradient_descent.fit(x,y)
title_dg = "gradient descent"
compute_results(x,y,b,m,title_dg,"blue")

"""
3 - regression moindre carrés
"""

lin_reg_least_square = LinearRegressionBrieuc(least_square=True)
#train linear regression model least_square
b,m = lin_reg_least_square.fit(x,y)
title_ls = "least square"
compute_results(x,y,b,m,title_ls,"red")

"""
4 - regression Scikit-Learn
"""
regr = LinearRegression(fit_intercept=True)    
# Train the model using the training sets
x=x.reshape((100,1))
y=y.reshape((100,1))
 
regr.fit(x, y)

b = regr.intercept_[0]
m = regr.coef_[0][0]
title = "scikit learn"
compute_results(x,y,b,m,title,"green")
plt.show()
