# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 23:28:52 2018

@author: brieu
"""
import numpy as np

class LogisticRegression:
    """
    Let's build a logistic regression model
    
    1) plutot que de prédire une classe, on prédit la probabilité d'appartenir
        a la classe 1.
        
        p(y = 1|x) = hw(x) = logistic(w.x)

    2) choisir la classe la plus probable selon le model:
        Si hw(x) >= 0.5:
            choisir classe 1
        Sinon:
            choisir classe 0
            
    3) on définit une fonction de cout:
        L(y, hw(x)) = - y * log(hw(x)) - (1 - y) * log(1 - hw(x)))
    
        on veut donc minimiser (y - hw(x))
    """
    # TODO: build logReg class
        
  
