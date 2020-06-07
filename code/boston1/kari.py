import numpy as np
from sklearn.datasets import load_boston
from scipy import stats
dataset = load_boston()

samples, target, featre_names = dataset.data, dataset.target, dataset.featre_names

CRIM = samples[:,0:1]
minimum = np.round(np.amin(CRIM), decimals=1)
maximum = np.round(np.amax(CRIM), decimals=1)
variance  = np.round(np.var(CRIM), decimals=1)
mean = np.round(np.mean(CRIM), decimals=1)
Befor_trim = np.vstack(minimum, maximum, variance, mean)
minimum_trim = stats.tmin(CRIM,1)
maximum_trim = stats.tmax(CRIM, 40)
variance_trim = stats.tmin(CRIM, (1,40))
mean_trim = stats.tmin(CRIM, (1,40))
After_trim = np.round(np.vstack(minimum_trim, maximum_trim, variance_trim, mean_trim), decimals=1)

stat_labels1 = ['minm','maxm','vari','mean']
Basic_stastice1 = np.hstack((Befor_trim, After_trimfrm))

print("        Before  After")
for stat_labels1, row1 in zip(stat_labels1, Basic_stastice1):
    print('%s [%s]' % (stat_labels1,''.join('%07s' % a for a in row1)))
