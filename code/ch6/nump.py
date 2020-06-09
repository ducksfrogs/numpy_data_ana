from sklearn import datasets
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X = boston.data
y = boston.target

lstat = x[0:, -1]
lstat.shape

from scipy import stats
slope, intercept, r_value, std_err = stats.linregress(lstat, y)
