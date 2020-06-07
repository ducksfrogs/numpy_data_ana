import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
dataset = load_boston()
samples, label, feature_names = dataset.data, dataset.target, dataset.feature_names

bostondf = pd.DataFrame(dataset.data)
bostondf.columns = dataset.feature_names
bostondf['Target price'] = dataset.target
bostondf.head()

bostondf.plot(x='RM', y='Target price', style='o')

def prediction(X, coefficient, intercept):
    return X*coefficient + intercept

def cost_function(X, Y, coefficient, intercept):
    MSE = 0.0
    for i in range(len(X)):
        MSE += (Y[i] -(coefficient*X[i] + intercept))**2
    return MSE / len(X)

def update_weights(X, Y, coefficient, intercept, learning_rate):
    coefficient_derivative = 0
    intercept_derivative = 0
    for i in range(len(X)):
        coefficient_derivative += -2*X[i] *(Y[i] -(coefficient * X[i] + intercept))
        intercept_derivative += -2*(Y[i] - (coefficient* X[i] + intercept))
    coefficient -= (coefficient_derivative / len(X)) * learning_rate
    intercept -= (intercept_derivative / len(X)) * learning_rate
    return coefficient, intercept

def train(X, Y, coefficient, intercept, learning_rate, iteration):
    cost_hist = []
    for i in range(iteration):
        coefficient, intercept = update_weights(X, Y, coefficient, intercept, learning_rate)
        cost = cost_function(X, Y, coefficient, intercept)
        cost_hist.append(cost)
    return coefficient, intercept, cost_hist


learning_rate = 0.01
iteration = 10001
coefficient = 0.3
intercept = 2

X = bostondf.iloc[:, 5:6].values
Y = bostondf.iloc[:, 13:14].values

# coefficient, intercept, cost_history = train(X, Y, coefficient, intercept, learning_rate, iteration)
coefficient, intercept, cost_history = train(X, Y, coefficient, intercept=2, learning_rate=0.01, iteration=10001)
