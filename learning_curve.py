import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('a.csv', usecols=['avg', 'min', 'max'])
df.plot()
plt.legend(loc="upper left")
plt.show()
