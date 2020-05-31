#calling libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define area of 
def area(x,y):
	pass

#Voronoi Diagram
points = np.array([[5, 0], [2, 4], [3, 2], [1, 0], [1, 1]])
vor = Voronoi(points)
voronoi_plot_2d(vor)


from sklearn.datasets import make_blobs

#data , label = function()
X, y = make_blobs(n_samples=500, centers=3, n_features=2,
                  random_state=0)


# unique labels
plt.scatter(X[:,0],X[:,1], color="red", s=10, label="People")
plt.show()