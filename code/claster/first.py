import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.graph_objs as go

init_notebook_mode(connected=true)
sys.path.append("".join([os.environ["home"]]))

from sklearn.datasets import load_iris

iris_data = load_iris()
iris_data.feature_names

x = [v[0] for v in iris_data.data]
y = [v[1] for v in iris_data.data]

trace = go.scatter(
    x =x,
    y =y,
    mode = 'makers'
)

layout = go.layout(title="iris dataset",
                   hovermode='closest',
                   xaxis= dict(title='sepal length(cm)',
                               ticklen=5,
                               zeroline=false,
                               gridwidth=2,
                               ),
                   yaxis=dict(title='sepal width (cm)', ticklen=5,gridwidth=2,),
                   showlegend=false
                   )
data = [trace]

fig= go.figure(data=data, layout=layout)
plot(fig)

df = pd.dataframe(iris_data.data, columns=[iris_data.feature_names])
df['class'] = [iris_data.target_names[i] for i in iris_data.target]

import plotly.figure_factory as ff

fig = ff.create_scatterplotmatrix(df, index='class', diag='histgram', size=10
                                  height=800, with=800)

plot(fig)
