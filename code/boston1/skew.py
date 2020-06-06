from sklearn.datasets import load_boston
from scipy import stats

datast = load_boston()

samples, label, feature_names = dataset.data, dataset.target, dataset.feature_names

for n in range(0,len(feature_names_new)):
    kurt = stats.describe(samples[n])[5]
    skew = stats.describe(samples[n])[4]
    print(feature_names_new[n] + "-Kurtosis: {} Skewness: {}".format(kurt, skew))
