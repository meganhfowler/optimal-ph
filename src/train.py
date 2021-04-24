#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from model import NeuralNet, BaselineModel
import config

# Load data set
with open('data/train_set.csv', 'rb') as train_data:
    # check how many training rows are used
    df = pd.read_csv(train_data, nrows = config.training_rows)

df_train, df_test = train_test_split(df, test_size=0.2)

if config.use_neural_net:
    NeuralNet(model_file_path='src/model.pickle').train(df_train)
else:
    BaselineModel(model_file_path='src/model.pickle').train(df_train)
