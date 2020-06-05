#calling libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
import pandas as pd

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

	#Visualization of population generation
	plt.scatter(main_population[:,0], main_population[:,1], marker = '.',color="red", s=10, label="City People")
	plt.scatter(other_population[:,0],other_population[:,1],  marker = '.' , color="green", s=10, label="Scattered/Temporary People")
	plt.show()

	population = np.concatenate((main_population, other_population))
	return population

def generate_clusters(population_data, Optimal_NumberOf_Components=8):
	'''
	This method will generate clusters 
	'''
	from sklearn.cluster import KMeans
	import sklearn
	# silhouette_score_values= []
 
	# NumberOfClusters=range(2,30)
	
	# for i in NumberOfClusters:
	# 	classifier=KMeans(i,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
	# 	classifier.fit(population_data)
	# 	labels= classifier.predict(population_data)
	# 	print("Number Of Clusters:")
	# 	print(i)
	# 	print("Silhouette score value")
	# 	print(sklearn.metrics.silhouette_score(population_data,labels ,metric='euclidean', sample_size=None, random_state=None))
	# 	silhouette_score_values.append(sklearn.metrics.silhouette_score(population_data,labels ,metric='euclidean', sample_size=None, random_state=None))
	
	# plt.plot(NumberOfClusters, silhouette_score_values)
	# plt.title("Silhouette score values vs Numbers of Clusters ")
	# plt.show()
	
	# Optimal_NumberOf_Components=NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
	# print("Optimal number of components is:")
	# print(Optimal_NumberOf_Components)
	kmeans = KMeans(n_clusters=Optimal_NumberOf_Components, random_state=0).fit(population_data)
	cluster_label  = kmeans.labels_
	region_centers = kmeans.cluster_centers_
	return cluster_label, region_centers

def cell_tower_problem():
	'''
	This method will solve cell tower problem using gurobi library
	'''
	import gurobipy as gp
	from gurobipy import GRB

	# tested with Gurobi v9.0.0 and Python 3.7.0

	# Parameters
	budget = 200 #lakhs
	regions, population = gp.multidict({
		0: 523, 1: 690, 2: 420,
		3: 1010, 4: 1200, 5: 850,
		6: 400, 7: 1008, 8: 950
	})

	sites, coverage, cost = gp.multidict({
		0: [{0,1,5}, 42],
		1: [{0,7,8}, 61],
		2: [{2,3,4,6}, 52],
		3: [{2,5,6}, 55],
		4: [{0,2,6,7,8}, 48],
		5: [{3,4,8}, 92]
	})

	# MIP  model formulation
	# m = gp.Model("cell_tower")
	m = gp.Model()

	build = m.addVars(len(sites), vtype=GRB.BINARY, name="Build")
	is_covered = m.addVars(len(regions), vtype=GRB.BINARY, name="Is_covered")

	m.addConstrs((gp.quicksum(build[t] for t in sites if r in coverage[t]) >= is_covered[r]
							for r in regions), name="Build2cover")
	m.addConstr(build.prod(cost) <= budget, name="budget")

	m.setObjective(is_covered.prod(population), GRB.MAXIMIZE)

	m.optimize() 

	# display optimal values of decision variables

	for tower in build.keys():
		if (abs(build[tower].x) > 1e-6):
			print(f"\n Build a cell tower at location Tower {tower}.")
	
	# Percentage of the population covered by the cell towers built is computed as follows.

	total_population = 0

	for region in range(len(regions)):
		total_population += population[region]

	coverage = round(100*m.objVal/total_population, 2)

	print(f"\n The population coverage associated to the cell towers build plan is: {coverage} %")

	# Percentage of budget consumed to build cell towers
	total_cost = 0

	for tower in range(len(sites)):
		if (abs(build[tower].x) > 0.5):
			total_cost += cost[tower]*int(build[tower].x)

	budget_consumption = round(100*total_cost/budget, 2)

	print(f"\n The percentage of budget consumed associated to the cell towers build plan is: {budget_consumption} %")

def voronoi_diagram(centers):
	'''
	This method will generate voronoi diagram 
	'''
	from scipy.spatial import Voronoi, voronoi_plot_2d
	vor = Voronoi(centers)
	voronoi_plot_2d(vor)
	vertices = vor.vertices #coord of voronoi vertices
	print('vertices: ', vertices)
	# ind_reg = vor.regions #indices of voronoi vertices
	# print('ind_reg: ', ind_reg)
	# ind_redig_verti = vor.ridge_vertices #indices of voronoi vertices forming ridge
	# print('ind_redig_verti: ', ind_redig_verti)
	# ind_ver_poi = vor.ridge_points #indices of each voronoi between which each voronoi lies
	# print('ind_ver_poi: ', ind_ver_poi)
	return vertices

def vertex_to_centroid(vertices,region_centers,cover_nearest_locations):
	'''
	This method will find nearest three centroid from the vertex
	'''
	dataframe = pd.DataFrame([])
	for i in range(len(vertices)): #for all vertices
		data_array = [vertices[i]]
		measured_dist = []
		for j in range(len(region_centers)):
			measured_dist.append(calculate_euclidian_dist(vertices[i],region_centers[j]))
		print('measured_dist: ', measured_dist)
		data_array.append(np.argsort(measured_dist)[:cover_nearest_locations])
		print('data_array: ', data_array)
		dataframe.append(data_array)
	return dataframe

def calculate_euclidian_dist(array1,array2):
	'''
	This method will calculate euclidian distance 
	'''
	dist = np.linalg.norm((array1-array2))
	return dist

	


if __name__ == "__main__":
	## Parameters
	dim = 2 #dimension
	main_cities = 5  #to generate data points
	total_population = 500
	cover_nearest_locations = 3
	#area
	min_c = 10
	max_c = 20
	regions = 10 #to generate clusters
	population = GeneratePopulation(dim,main_cities,total_population,min_c,max_c)
	population_label , region_centers = generate_clusters(population,regions)
	
	#voronoi diagram 
	vertices = voronoi_diagram(region_centers)

	#finding nearest centroid from each vertex 
	allocated_facility = vertex_to_centroid(vertices,region_centers,cover_nearest_locations)
	print('allocated_facility: ', allocated_facility)
	exit()

	#Writing center on graph plot 
	for i in range(len(region_centers)):
		plt.text(region_centers[i][0],region_centers[i][1],str(i), fontsize=15)
	
	#writing vertex on plot
	for i in range(len(vertices)):
		plt.text(vertices[i][0],vertices[i][1],str(i), fontsize=15)

	#Visualization population Regions
	regional_data = [] #clusterwise node of person
	regional_population = [] #clusterwise population
	unique_label = np.unique(population_label)
	for i in range(len(unique_label)):
		temp_data = []
		for j in range(len(population)):
			if(population_label[j] == unique_label[i]):
				temp_data.append(list(population[j,:]))
		temp_data = np.array(temp_data)
		regional_population.append(len(temp_data))
		regional_data.append(temp_data)
		color = "#%06x" % random.randint(0, 0xFFFFFF)
		plt.scatter(temp_data[:,0],temp_data[:,1],c=color,marker='.',label='cluster'+str(i))
	plt.legend()
	plt.show()
	# cell_tower_problem()


	
		