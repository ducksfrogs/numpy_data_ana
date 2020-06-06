import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

dataset = load_boston()
samples, labal, feature_names = dataset.data, dataset.target, dataset.feature_names
samples_new = np.delete(samples, 3, axis=1)

fig,((ax1, ax2, ax3),(ax4,ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)))
    = plt.subplots(4,3, figsize=(10,15))
axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
feature_names_new = np.delete(feature_names, 3)
for n in range(0, len(axs)):
    axs[n].hist(samples_new[:,n:n+1], bins='auto', normed=True)
    axs[n].set_title('{}'.format(feature_names[n]))
