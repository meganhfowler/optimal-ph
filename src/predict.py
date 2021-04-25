#!/usr/bin/env python3
import argparse
import pandas as pd
from model import NeuralNet, BaselineModel
import torch
import config

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', default='submission/input.csv')
args = parser.parse_args()

# Config
output_file_path = 'test/predictions.csv'

# Load input.csv
with open(args.input_csv) as input_csv:
    df = pd.read_csv(input_csv)

def neural_predict():
    # Run predictions
    y_predictions = NeuralNet(model_file_path='src/model.pickle').predict(df)
    # Save predictions to file
    y_predictions = y_predictions.detach().numpy()
    df_predictions = pd.DataFrame(y_predictions)
    df_predictions.columns = ['prediction']
    df_predictions.to_csv(output_file_path, index=False)

    print(f'{len(y_predictions)} predictions saved to a csv file')


def encoded_predict():
    # old version:
    # Run predictions
    y_predictions = BaselineModel(model_file_path='src/model.pickle').predict(df)

    # Save predictions to file
    df_predictions = pd.DataFrame({'prediction': y_predictions})
    df_predictions.to_csv(output_file_path, index=False)

    print(f'{len(y_predictions)} predictions saved to a csv file')


if config.use_neural_net:
    neural_predict()
else:
    encoded_predict()
