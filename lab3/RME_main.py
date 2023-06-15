 # Script to load dataset, for dataset normalization, for train_test model
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# Script to load dataset,
load_dataset = pd.read_csv('/home/prof/google-disk/kaggle_datasets/powerlifting/numeric_powerlifting_man.csv')
print('\nLoaded Dataset\n')
print(load_dataset)


# Script for dataset normalization
powerlifting_norm = load_dataset

data = powerlifting_norm
names = data.columns
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

powerlifting_norm = pd.DataFrame(data, columns=names)

print('\nDataset with normalization\n')
print(powerlifting_norm)


# Script for train_test model

X, y = powerlifting_norm.drop(columns = ['Wilks']).values, powerlifting_norm['Wilks'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# создаем объект класса с указанием гиперпараметров
k = 8
weights = 'distance'

kNN = KNeighborsRegressor(n_neighbors=k,
                          weights=weights )

# обучаем на тренировочных данных
kNN.fit(X_train, y_train)

# предсказываем на тестовых данных
y_predict=kNN.predict(X_test)

scoring = {'R2': 'r2',
           '-MSE': 'neg_mean_squared_error',
           '-MAE': 'neg_mean_absolute_error',
           '-Max': 'max_error'
           }

scores = cross_validate(kNN, X_train, y_train, scoring = scoring,
                       cv= ShuffleSplit(n_splits=5, random_state=42)
                       )
DF_cv_kNN = pd.DataFrame(scores)

print('\nОшибка на тестовых данных\n')

print('MSE: %.5f' % mse(y_test, y_predict))
print('RME: %.5f' % mse(y_test, y_predict, squared=False))
