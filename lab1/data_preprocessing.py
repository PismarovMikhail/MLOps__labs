# Script for dataset normalization 
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from data_creation import load_dataset 
# Открытие файла с загруженныи датасетом

#file = open('/home/prof/MLOps/MLOps_labs/MLOps_first_semester_labs/lab1/data_creation.py', 'r')

powerlifting_norm = load_dataset

data = powerlifting_norm
names = data.columns
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

powerlifting_norm = pd.DataFrame(data, columns=names)

print('\nDataset with normalization\n')
print(powerlifting_norm)


