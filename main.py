import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# visualising the first data set (you have to put the same the data file in to the same file as your python project)
df = pd.read_csv('BTC-USD.csv')
print(df.head())


df.plot ( x = "Date", y = "Close", kind = "line", figsize =(15,8), title = "BTC Price", xlabel = "Date", ylabel = "Price")

plt.show()


