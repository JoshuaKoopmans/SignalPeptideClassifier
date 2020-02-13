# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:21:55 2020

@author: gebruiker
"""

import matplotlib.pyplot as plt
from sklearn import svm, metrics
from matplotlib.colors import ListedColormap
import numpy as np


def svc_classifier(train, train_labels, test, test_labels, show_plot=True):

    svc = svm.SVC(kernel='linear')

    svc.fit(train, train_labels)
    
    predicted = svc.predict(test)
    score = svc.score(test, test_labels)
    
    print('============================================')
    print('\nScore ', score)
    print('\nResult Overview\n',   metrics.classification_report(test_labels, predicted))
    print('\nConfusion matrix:\n', metrics.confusion_matrix(test_labels, predicted)      )          
                    
    ##########################################                
    cmap, cmapMax = plt.cm.RdYlBu, ListedColormap(['#FF0000', '#0000FF'])           
               
    # Plotting broken
     
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)     
           
           
    h = 0.3  
    x_min, x_max = train[:, 0].min()-.3, train[:, 0].max()+.3
    y_min, y_max = train[:, 1].min()-.3, train[:, 1].max()+.3                
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))    
       
    if hasattr(svc, "decision_function"):
            Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
            Z = svc.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.7)
    
    # Plot also the training points
    ax.scatter(train[:, 0], train[:, 1], c=train_labels, cmap=cmapMax)
    # and testing points
    ax.scatter(test[:, 0], test[:, 1], c=test_labels, cmap=cmapMax, alpha=0.5)
    
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    plt.title(str(score))
    
    if show_plot:
        plt.show()