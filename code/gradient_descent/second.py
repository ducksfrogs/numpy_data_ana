import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
dataset = load_boston()

samples, label, feature_names = dataset.data, dataset.target, dataset.feature_names

samples_trim = stats.trimboth(samples, 0.1)
label_trim = stats.trimboth(label, 0.1)

print(samples.shape, label.shape)
print(samples_trim.shape, label_trim.shape)

from sklearn.model_selection import train_test_split
samples_train, samples_test, label_train, label_test = train_test_split(samples_trim, label_trim,
 test_size=0.2, random_state=0)

print(samples_train.shape, label_train.shape)
print(samples_test.shape, label_test.shape)

regressor = LinearRegression()
regressor.fit(samples_train, label_train)

label_pred = regressor.predict(samples_test)

import matplotlib.pyplot as plt

plt.scatter(label_test, label_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted Prices")
plt.title("Prices vs Predicted Prices")
plt.axis("equal")

from sklearn., metrics import mean_squared_error
from sklearn.metrics import r2_score

mse = mean_squared_error(label_test, label_pred)
r2 = r2_score(label_test, label_pred)
print(mse)
print(r2)
