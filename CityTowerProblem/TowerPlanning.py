#calling libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d
import random


def GeneratePopulation(dim,main_cities,total_population,min_c,max_c):
	'''
	This method will generate the random population for simulation 
	'''
	# Define area of 
	def random_population(dimension=2, min_c=0, max_c=10):
		return [random.randint(min_c, max_c) for _ in range(dimension)]

	#Generating clusters
	from sklearn.datasets import make_blobs
	'''
	90% of  the population is assumed to generate cluster of population
	10% of population is scattered for business or store or other purpose
	'''
	#90%
	main_population, y = make_blobs(n_samples=int(total_population * 0.9),cluster_std=1.5, centers=main_cities, center_box=(min_c, max_c) ,n_features=dim,random_state=41)
	#10%
	other_population = np.zeros((int(0.3*total_population),dim))

	for i in range(len(other_population)):
		other_population[i] = random_population(dim,min_c,max_c)

	# unique labels
	plt.scatter(main_population[:,0], main_population[:,1], marker = '.',color="red", s=10, label="City People")
	plt.scatter(other_population[:,0],other_population[:,1],  marker = '.' , color="green", s=10, label="Scattered/Temporary People")
	plt.show()

	population = np.concatenate((main_population, other_population))
	return population

if __name__ == "__main__":
	## Parameters
	dim = 2 #dimension
	main_cities = 5
	total_population = 1000
	#area
	min_c = 10
	max_c = 20
	population = GeneratePopulation(dim,main_cities,total_population,min_c,max_c)