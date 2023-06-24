#  model validation script 
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score

# load X_test, y_test
X_test = pd.read_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/test/X_test.csv')
y_test = pd.read_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/test/y_test.csv')

# load model
kNN = load_model('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/kNN.h5')

# prediction on test data
y_predict=kNN.predict(X_test)

scoring = {'R2': 'r2'}

scores = cross_validate(kNN, X_train, y_train, scoring = scoring,
                       cv= ShuffleSplit(n_splits=5, random_state=42)
                       )
DF_cv_kNN = pd.DataFrame(scores)

print('\nОшибка на тестовых данных\n')
print('R2: %.5f' % r2_score(y_test, y_predict))
