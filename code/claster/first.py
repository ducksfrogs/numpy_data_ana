import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objs as go

init_notebook_mode(connected=True)
sys.path.append("".join([os.environ["HOME"]]))

from sklearn.datasets import load_iris

iris_data = load_iris()
iris_data.feature_names

x = [v[0] for v in iris_data.data]
y = [v[1] for v in iris_data.data]

trace = go.Scatter(
    x =x,
    y =y,
    mode = 'makers'
)

layout = go.Layout(title="Iris Dataset",
                   hovermode='closest',
                   xaxis= dict(title='sepal length(cm)',
                               ticklen=5,
                               zeroline=False,
                               gridwidth=2,
                               ),
                   yaxis=dict(title='sepal width (cm)', ticklen=5,gridwidth=2,),
                   showlegend=False
                   )
data = [trace]

fig= go.Figure(data=data, layout=layout)
plot(fig)

df = pd.DataFrame(iris_data.data, columns=[iris_data.feature_names])
df['class'] = [iris_data.target_names[i] for i in iris_data.target]
