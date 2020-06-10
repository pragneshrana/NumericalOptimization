import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Result.csv')
plt.plot(data['Regions'],data['Coverage'],c='red',label='Coverage')
plt.plot(data['Regions'],data['Budget'],c='green',label='Coverage')
plt.show()

plt.plot(data['Regions'],data['Execution Time'],c='orange',label='Coverage')
plt.show()

