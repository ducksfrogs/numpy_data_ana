import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

dataset = datasets.load_diabetes()

df = pd.DataFrame(dataset)
dataset.feature_names
