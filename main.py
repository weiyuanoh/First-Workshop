import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# visualising the first data set (you have to put the same the data file in to the same file as your python project)
df = pd.read_csv('BTC-USD1.csv')  # is to init excel into dataframe represented by df

# plotting time series data
# df.plot(x="Date", y="Close", kind="line", figsize=(15, 8), title="BTC Price", xlabel="Date", ylabel="Price")
# plt.show()


# SMA 20
df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
# SMA 50
df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
# creating list to close
close = df['Close'].values.tolist()
print(close)
print(sum(close))


# calculating variance
def variance(close):
    n = len(close)
    mean = sum(close) / n
    variance = 0
    for close in close:
        dev = close - mean
        variance += (dev ** 2)/n

    return variance
varclose = variance(close)
print(varclose)

# plot SMA
# df.plot(x="Date", y=["Close", "SMA_20", "SMA_50"], kind="line", figsize=(15, 8), title="BTC Price", xlabel="Date",
#        ylabel="Price")
# plt.show()
