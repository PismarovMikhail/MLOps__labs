# Script for model training
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsRegressor

# load X_train, y_train
X_train = pd.read_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/train/X_train.csv')
y_train = pd.read_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/train/y_train.csv')


# создаем объект класса с указанием гиперпараметров
k = 8
weights = 'distance'

kNN = KNeighborsRegressor(n_neighbors=k, 
                          weights=weights )

# traning on train data
kNN.fit(X_train, y_train)

# save train model
kNN.save('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/kNN.h5')

