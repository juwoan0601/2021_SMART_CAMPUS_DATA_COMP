import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn
from xgboost.sklearn import XGBRegressor; seaborn.set() 
from sklearn.preprocessing import LabelEncoder 

def learning_curve(X, y, pipe_lr): # TPOT 외에 다른 것 사용할 때는 pipe_lr 대신 model, 밑의 pipe_lr 주석 없애기
     
    from sklearn.preprocessing import StandardScaler 
    from sklearn.pipeline import Pipeline 
    from sklearn.model_selection import learning_curve 
     
    # pipe_lr = Pipeline([
    #         ('scl', StandardScaler()),
    #         ('fit', model)
    # ])
    
    train_sizes = np.linspace(0.1, 1.0, 6)

    train_sizes, train_scores, test_scores =\
        learning_curve(estimator = pipe_lr, X = X, y = y, train_sizes=train_sizes, cv=5, scoring = 'neg_mean_absolute_percentage_error') 
    print(train_sizes)
    print(train_scores)
    print(test_scores)

    train_mean = np.mean(train_scores,axis=1) 
    train_std = np.std(train_scores,axis=1) 
    test_mean = np.mean(test_scores,axis=1) 
    test_std = np.std(test_scores,axis=1) 
     
    plt.plot(train_sizes, train_mean, color='green', marker='o', markersize=5, label='training accuracy') 
    plt.plot(train_sizes, test_mean, color='red',linestyle='--', marker='s', markersize=5, label='validation accuracy') 
     
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='green') 
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='red') 
     
    plt.title('learning curve', fontsize=17) 
    plt.xlabel('Number of training samples', fontsize=17) 
    plt.ylabel('Negative Mean Squared Error', fontsize=17)
    plt.legend(loc='lower right', fontsize=17) 
     
    plt.show()
