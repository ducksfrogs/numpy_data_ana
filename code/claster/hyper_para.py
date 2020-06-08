import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=30, centers=3, n_features=3, random_state=42)

k_means = KMeans(n_clusters=3)
y_hat = k_means.fit_predict(X)

data = [1,2,3,2,1,3,9,8,11,12,10,14,25,26,24,30,22,24,27]

trace1 = go.Scatter(x=data, y=[0 for num in data],mode='markers', name='Data',
                    markers=dict(size=12)
                    )

layout = go.Layout(title='1D vector')

traces = [trace1]

fig = go.Figure(data=traces, layout=layout)
plot(fig)

n_clusters = 3
c_centers = np.random.choice(data, n_clusters)
print(c_centers)

deltas = np.array([np.abs(point - c_centers) for point in X])
