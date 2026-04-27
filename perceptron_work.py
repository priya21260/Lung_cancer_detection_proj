import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

dataset= pd.read_csv("Placement.csv")

dataset.head(3)

sns.scatterplot(x="cgpa",y="score",data=dataset,hue="placed")


