# Script for dataset load, for dataset normalization, for train_test model
# for y prediction 
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score


# Load dataset
load_dataset = pd.read_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/numeric_powerlifting_man.csv')
print('\nLoaded Dataset\n')
print(load_dataset)

# Dataset normalization
powerlifting_norm = load_dataset

data = powerlifting_norm
names = data.columns
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

powerlifting_norm = pd.DataFrame(data, columns=names)

print('\nDataset with normalization\n')
print(powerlifting_norm)

# Split dataset

X, y = powerlifting_norm.drop(columns = ['Wilks']).values, powerlifting_norm['Wilks'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create model
k = 8
weights = 'distance'

kNN = KNeighborsRegressor(n_neighbors=k,
                          weights=weights )

# train on train data
kNN.fit(X_train, y_train)

# predict on test data
y_predict=kNN.predict(X_test)

scoring = {'R2': 'r2'}

scores = cross_validate(kNN, X_train, y_train, scoring = scoring,
                       cv= ShuffleSplit(n_splits=5, random_state=42)
                       )
DF_cv_kNN = pd.DataFrame(scores)

print('\nОшибка на тестовых данных\n')
print('R2: %.5f' % r2_score(y_test, y_predict))
