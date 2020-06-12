import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Result.csv')
data  = data.sort_values(by=['Regions'])
# fit = np.polyfit(data['Regions'],data['Execution Time'], 3, rcond=None, full=False)
# print('fit: ', fit)
# plt.plot(data['Regions'],data['Coverage'],c='red',label='Coverage')
# plt.show()
# plt.plot(data['Regions'],data['Budget'],c='green',label='Coverage')
# plt.show()

plt.plot(data['Regions'],data['Execution Time'],c='orange',label='Computational time')
plt.xlabel('Number of Clusters (Input parameter)')
plt.ylabel('Time (sec)')
plt.title('Plot of computational time')
plt.legend()
plt.savefig('./result/time.jpg')
plt.show()

