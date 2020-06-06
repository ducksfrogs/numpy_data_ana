from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt


dataset = load_boston()
data, target, feature_names = dataset.data, dataset.target, dataset.feature_names

NOX = data[:,5:6]

fig, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2,3, sharex=True)
axs = [ax1, ax2, ax3, ax4, ax5, ax6]
list_methods = ['fd','doane',' scott','rice', 'sturges','sqrt']
plt.tight_layout(pad=1.1, w_pad=0.8, h_pad=1.0)
for n in range(0, len(axs)):
    axs[n].hist(NOX, bins= list_methods[n])
    axs[n].set_title('{}'.format(list_methods[n]))
    plt.show()
