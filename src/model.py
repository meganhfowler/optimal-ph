#!/usr/bin/env python3
from sklearn import tree
from sklearn.linear_model import SGDRegressor
import numpy as np
import pickle
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import scipy



class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def vectorize_sequences(self, sequence_array):
        vectorize_on_length = np.vectorize(len)
        return np.reshape(vectorize_on_length(sequence_array), (-1, 1))

    def train(self, df_train):
        X = self.vectorize_sequences(df_train['sequence'].to_numpy())
        y = df_train['mean_growth_PH'].to_numpy()

        #model = tree.DecisionTreeRegressor()
        model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        model.fit(X, y)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)

class Utils:
    max_length = 8192
    def encode_aa(aa: str) -> float:
        assert len(aa) == 1

        mapping = {
            'A': 0.04,
            'C': 0.08,
            'D': 0.12,
            'E': 0.16,
            'F': 0.20,
            'G': 0.24,
            'H': 0.28,
            'I': 0.32,
            'K': 0.36,
            'L': 0.40,
            'M': 0.44,
            'N': 0.48,
            'P': 0.52,
            'Q': 0.56,
            'R': 0.60,
            'S': 0.64,
            'T': 0.68,
            'V': 0.72,
            'W': 0.76,
            'Y': 0.80,
        }
        e = mapping.get(aa.upper())
        if e is None:
            e = 0.00

        return e

    def encode_protein(prot: str):
        encoding = [
            Utils.encode_aa(aa)
            for aa in prot
        ]
        return np.array(encoding)

    def pad(encoding, max_length):
        enc_length = len(encoding)
        pad_length = max_length - enc_length
        assert pad_length >= 0, "Encoding exceeded max length"
        pad = np.zeros((1, pad_length)).squeeze()
        padded_enc = np.concatenate((encoding, pad))
        return padded_enc

    def encode_and_pad(prot: str):
        return Utils.pad(Utils.encode_protein(prot), Utils.max_length)

    def score(self, df_test):
        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        y_true = df_test['mean_growth_PH'].to_numpy()
        y_pred = self.predict(df_test)
        return (self.mean_squared(y_true, y_pred), self.spearmanr(y_true, y_pred))

    def mean_squared(self, y_true, y_pred):
        return sklearn.metrics.mean_squared_error(y_true, y_pred)

    def spearmanr(self, y_true, y_pred):
        return scipy.stats.spearmanr(y_true, y_pred)



class BaselineEncodedModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def vectorize_sequences(self, sequence_array):
        encoded_sequences = [
            Utils.encode_and_pad(sequence)
            for sequence in sequence_array
        ]
        return np.array(encoded_sequences)

    def train(self, df_train):
        X = self.vectorize_sequences(df_train['sequence'].to_numpy())
        y = df_train['mean_growth_PH'].to_numpy()
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        model.fit(X, y)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)


    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)


