# Script for dataset normalization 
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from data_creation import load_dataset 
from sklearn.model_selection import train_test_split

powerlifting_norm = load_dataset

# Dataset normalization
data = powerlifting_norm
names = data.columns
scaler = MinMaxScaler()
data = scaler.fit_transform(data)


X, y = powerlifting_norm.drop(columns = ['Wilks']).values, powerlifting_norm['Wilks'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42
                                                   )
# Write X_train, y_train in foulder train
# Write X_test, y_test in foulder test
X_train = pd.DataFrame(data, columns=names)
y_train = pd.DataFrame(data, columns=names)

X_train.to_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/train/X_train.csv')
y_train.to_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/train/y_train.csv')

X_test = pd.DataFrame(data, columns=names)
y_test = pd.DataFrame(data, columns=names)

X_test.to_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/test/X_test.csv')
y_test.to_csv('/home/mike/magistracy/google_drive/kaggle_datasets/powerlifting/powerlifting_man_old/test/y_test.csv')

print('\nDataset with normalization\n')
print(powerlifting_norm)
