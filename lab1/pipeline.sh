#!/bin/bash

# Script that runs all scripts
#import data_creation

python3 data_creation.py
python3 data_preprocessing.py
python3 model_testing.py
