# model validation script 
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsRegressors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_metrics import mean_squared_error as mse
from sklearn.model_selection import r2_score

X_test = np.load(open('X_test', 'rb'))
y_test = np.load(open('y_test', 'rb'))
X_train = np.load(open('X_train', 'rb'))
y_train = np.load(open('y_train', 'rb'))

y_predict = model.predict(X_test)

scoring = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           '-Max': 'max_error'
           }

scores = cross_validate(model, X_train, y_train, scoring = scoring,
