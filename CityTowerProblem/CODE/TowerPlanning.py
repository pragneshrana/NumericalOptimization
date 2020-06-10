#calling libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d
import random
import pandas as pd
import sys
import os 
from datetime import date
import time 

class TowerPlanning():

	def __init__(self,dim,main_cities,total_population,min_c,max_c,budget,RequiredRegions,NeighborsToCover,CostPerEach,PopulationData=None):
		self.dim = dim
		self.main_cities = main_cities
		self.total_population = total_population
		self.min_c = min_c
		self.max_c = max_c
		self.PopulationData = None
		self.budget = budget
		self.RequiredRegions = RequiredRegions
		self.PopulationData = PopulationData
		self.Usedbudget = None
		self.CoveredRegion = None
		self.NeighborsToCover = NeighborsToCover
		self.CostPerEach = CostPerEach
		#creating Directory
		try:
			os.mkdir('./result/')  
		except FileExistsError:
			pass
			
		try:
			os.mkdir('./result/'+str(self.RequiredRegions)+'/')  
		except FileExistsError:
			pass
		
		try:
			os.mknod('./result/'+str(self.RequiredRegions)+'/ResultSummary.txt')
		except FileExistsError:
			pass
		
		self.f = open('./result/'+str(self.RequiredRegions)+'/ResultSummary.txt',"w")
		self.f.write('\n\n\n'+str(date.today()))

		#Creating Folder
		try:
			os.mknod('Result.csv')
		except FileExistsError:
			pass

	def GeneratePopulation(self,):
		'''
		This method will generate the random population for simulation 
		'''
		# Define area of 
		def random_population():
			random.seed(30)
			return [random.randint(self.min_c, self.max_c) for _ in range(self.dim)]

		#Generating clusters
		from sklearn.datasets import make_blobs
		'''
		90% of  the population is assumed to generate cluster of population
		10% of population is scattered for business or store or other purpose
		'''
		#90%
		main_population, y = make_blobs(n_samples=int(self.total_population),cluster_std= (self.max_c - self.min_c), centers=self.main_cities, center_box=(self.min_c, self.max_c) ,n_features=self.dim,random_state=41)
		#10%
		other_population = np.zeros((int(0.3*total_population),self.dim))

		for i in range(len(other_population)):
			other_population[i] = random_population()

		#Visualization of population generation
		plt.scatter(main_population[:,0], main_population[:,1], marker = '.',color="red", s=10, label="City People")
		plt.scatter(other_population[:,0],other_population[:,1],  marker = '.' , color="green", s=10, label="Scattered/Temporary People")
		# plt.show()

		self.PopulationData = np.concatenate((main_population, other_population))

	def GenerateClusters(self,):
		'''
		This method will generate clusters 
		'''
		from sklearn.cluster import KMeans
		from sklearn.cluster import DBSCAN
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
		
		# self.RequiredRegions=NumberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
		# print("Optimal number of components is:")
		# print(self.RequiredRegions)

		##Kmeans
		kmeans = KMeans(n_clusters=self.RequiredRegions, init='k-means++', max_iter=100, n_init=1, verbose=0, random_state=3425).fit(self.PopulationData)
		cluster_label  = kmeans.labels_
		region_centers = kmeans.cluster_centers_
		return cluster_label, region_centers


	def ResultPlot(self,FinalNodes,region_centers):
		print('region_centers: ', region_centers)
		print('FinalNodes: ', FinalNodes)
		plt.clf()
		self.VoronoiDiagram(region_centers)
		plt.scatter(self.PopulationData[:,0],self.PopulationData[:,1],  marker = '.' , color="green", s=10, label="Scattered/Temporary People")
		for i in range(len(FinalNodes)):
			plt.scatter(FinalNodes[i][0],FinalNodes[i][1],marker = 'x' , color="red",label="TowerLocation"+str(i))
			plt.text(FinalNodes[i][0]+0.25,FinalNodes[i][1]+0.25,str(i), fontsize=15)
		# plt.ylim(self.min_c, self.max_c)
		# plt.xlim(self.min_c,self.max_c)
		plt.savefig('./result/'+str(self.RequiredRegions)+'/FinalResult.jpg')


	def cell_tower_problem(self,AllocatedFacilityData,RegionWisePopulation):
		print('AllocatedFacilityData: ', AllocatedFacilityData)
		print('RegionWisePopulation: ', RegionWisePopulation)
		'''
		This method will solve cell tower problem using gurobi library
		'''
		import gurobipy as gp
		from gurobipy import GRB

		# tested with Gurobi v9.0.0 and Python 3.7.0
		Populationkey = [*range(0,len(self.PopulationData))]
		PopulationDict = dict(zip(Populationkey,RegionWisePopulation))
		print('PopulationDict: ', PopulationDict)
		regions, population = gp.multidict(PopulationDict)

		# # Parameters
		# regions, population = gp.multidict({
		# 	0: 523, 1: 690, 2: 420,
		# 	3: 1010, 4: 1200, 5: 850,
		# 	6: 400, 7: 1008, 8: 950
		# })

		#Calculating Cost of each tower
		'''
		Summation of total population covered in all region
		'''
		cost = []
		for i in range(len(AllocatedFacilityData)):
			TempCost = 0
			sum = 0
			RegionsOccupiedByVertex = AllocatedFacilityData.iloc[i,1:]
			for j in range(self.NeighborsToCover):
				sum += RegionWisePopulation[RegionsOccupiedByVertex[j]]
			cost.append(sum + self.CostPerEach)

		RegionKey = [*range(0,len(AllocatedFacilityData))]

		RegionValue = []
		coverageData = []
		for i in range(len(AllocatedFacilityData)):
			coverageData = [list(AllocatedFacilityData.iloc[i,1:])]
			coverageData.append(cost[i])
			RegionValue.append(coverageData)
		RegionDict = dict(zip(RegionKey,RegionValue))
		print('RegionDict: ', RegionDict)

		# print('RegionDict: ', RegionDict)
		# sites, coverage, cost = gp.multidict({
		# 	0: [[0,1,5], 42],
		# 	1: [[0,7,8], 61],
		# 	2: [[2,3,4,6], 52],
		# 	3: [[2,5,6], 55],
		# 	4: [[0,2,6,7,8], 48],
		# 	5: [[3,4,8], 92]
		# })

		sites, coverage, cost = gp.multidict(RegionDict)



		# MIP  model formulation
		m = gp.Model("cell_tower")
		# m = gp.Model()

		build = m.addVars(len(sites), vtype=GRB.BINARY, name="Build")
		is_covered = m.addVars(len(regions), vtype=GRB.BINARY, name="Is_covered")

		m.addConstrs((gp.quicksum(build[t] for t in sites if r in coverage[t]) >= is_covered[r]
								for r in regions), name="Build2cover")
		m.addConstr(build.prod(cost) <= self.budget, name="budget")

		m.setObjective(is_covered.prod(population), GRB.MAXIMIZE)

		m.optimize() 

		# display optimal values of decision variables
		
		LocationFound = []
		for tower in build.keys():
			if (abs(build[tower].x) > 1e-6):
				print(f"\n Build a cell tower at location Tower {tower}.")
				self.f.write("\n Build a cell tower at location Tower "+str(tower))
				LocationFound.append(tower)
		# Percentage of the population covered by the cell towers built is computed as follows.

		total_population = 0

		for region in range(len(regions)):
			total_population += population[region]

		self.CoveredRegion = round(100*m.objVal/total_population, 2)

		print(f"\n The population coverage associated to the cell towers build plan is: {self.CoveredRegion} %")
		self.f.write("\n The population coverage associated to the cell towers build plan is: "+str(self.CoveredRegion))
		# Percentage of budget consumed to build cell towers
		total_cost = 0

		for tower in range(len(sites)):
			if (abs(build[tower].x) > 0.5):
				total_cost += cost[tower]*int(build[tower].x)
		try:
			self.Usedbudget = round(100*total_cost/budget, 2)
		except:
			return 0,0

		print(f"\n The percentage of budget consumed associated to the cell towers build plan is: {self.Usedbudget} %")
		self.f.write("\n The percentage of budget consumed associated to the cell towers build plan is: "+str(self.Usedbudget))
		return LocationFound

	def VoronoiDiagram(self,centers):
		'''
		This method will generate voronoi diagram 
		'''
		from scipy.spatial import Voronoi, voronoi_plot_2d
		vor = Voronoi(centers)
		voronoi_plot_2d(vor)
		vertices = vor.vertices #coord of voronoi vertices
		# ind_reg = vor.regions #indices of voronoi vertices
		# print('ind_reg: ', ind_reg)
		# ind_redig_verti = vor.ridge_vertices #indices of voronoi vertices forming ridge
		# print('ind_redig_verti: ', ind_redig_verti)
		# ind_ver_poi = vor.ridge_points #indices of each voronoi between which each voronoi lies
		# print('ind_ver_poi: ', ind_ver_poi)
		# return vertices

	def DistBtnCentroidNNeighbor(self,region_centers):
		'''
		This method will find nearest three centroid from the vertex
		return : methos will return 
		'''
		headers = ['RegionCenter']
		for i in range(self.NeighborsToCover):
			headers.append('NeighborCenter'+str(i))

		dataframe = pd.DataFrame([],dtype=int)

		for i in range(len(region_centers)): #find nearest centroid from all
			data_array = np.array([i])
			measured_dist = []
			for j in range(len(region_centers)):
					measured_dist.append(self.CalculateEuclidianDist(region_centers[i],region_centers[j]))
			print('measured_dist: ', measured_dist)
			data_array = np.concatenate((data_array,np.argsort(measured_dist)[1:self.NeighborsToCover+1]))
			print('np.argsort(measured_dist)[:self.NeighborsToCover]): ', np.argsort(measured_dist)[1:self.NeighborsToCover+1])
			print('np.argsort(measured_dist): ', np.argsort(measured_dist))
			dataframe = dataframe.append(pd.Series(list(data_array)),ignore_index=True)
		dataframe = dataframe.astype('int64', copy=False)
		print('dataframe: ', dataframe)
		dataframe.columns = headers
		return dataframe

	def CalculateEuclidianDist(self,array1,array2):
		'''
		This method will calculate euclidian distance 
		'''
		dist = np.linalg.norm((array1-array2))
		return dist
	
	def FindFinalNode(self,region_centers,IndexOfNodesToBuild,AllocatedFacilityData):
		print('region_centers: ', region_centers)
		FinalNodes = []
		
		for i in range(len(IndexOfNodesToBuild)):
			temp_nodes = []
			for j in range(self.NeighborsToCover+1):
				temp_nodes.append(region_centers[AllocatedFacilityData.iloc[IndexOfNodesToBuild[i],:][j]])
			FinalNodes.append(sum(temp_nodes) / len(temp_nodes))
		return FinalNodes
	

	def Simulate(self,):
		population_label , region_centers = self.GenerateClusters()
		
		#voronoi diagram 
		self.VoronoiDiagram(region_centers)
		

		#finding nearest centroid from each vertex 
		AllocatedFacilityData = self.DistBtnCentroidNNeighbor(region_centers)
		print('AllocatedFacilityData: ', AllocatedFacilityData)


		#Writing center on graph plot 
		for i in range(len(region_centers)):
			plt.text(region_centers[i][0],region_centers[i][1],str(i), fontsize=15)
		
		# #writing vertex on plot
		# for i in range(len(vertices)):
		# 	plt.text(vertices[i][0],vertices[i][1],str(i), fontsize=15)

		#Visualization population Regions
		regional_data = [] #clusterwise data
		RegionWisePopulation = [] #clusterwise population count
		unique_label = np.unique(population_label)
		for i in range(len(unique_label)):
			temp_data = []
			for j in range(len(self.PopulationData)):
				if(population_label[j] == unique_label[i]):
					temp_data.append(list(self.PopulationData[j,:]))
			temp_data = np.array(temp_data)
			RegionWisePopulation.append(len(temp_data))
			regional_data.append(temp_data)
			color = "#%06x" % random.randint(0, 0xFFFFFF)
			plt.scatter(temp_data[:,0],temp_data[:,1],c=color,marker='.',label='cluster'+str(i))
		plt.savefig('./result/'+str(self.RequiredRegions)+'/Regions.jpg')

		#optimizing 
		start = time.time()
		IndexOfNodesToBuild = self.cell_tower_problem(AllocatedFacilityData,RegionWisePopulation)
		end = time.time()
		ElapsedTime = end - start	

		FinalNodes  = self.FindFinalNode(region_centers,IndexOfNodesToBuild,AllocatedFacilityData)
		self.ResultPlot(FinalNodes,region_centers)
		self.f.close()
		return self.Usedbudget, self.CoveredRegion ,ElapsedTime

if __name__ == "__main__":

	#For Population
	dim = 2 #dimension
	main_cities = 5  #to generate data points
	headers = ['Regions','Coverage','Budget','Execution Time']

	if(sys.argv[1].isnumeric()):
		RequiredRegions = int(sys.argv[1]) #to generate clusters
		print('RequiredRegions: ', RequiredRegions)
	else:
		print('Pass Integer')
		exit()

	#############################
	#######   Input   ###########
	#############################
	## Parameters
	total_population = 100000
	NeighborsToCover = 3
	CostPerEach = 3000000
	budget = 1000000 * RequiredRegions  #lakhs#Budget
	#area
	min_c = 100
	max_c = 200
	#############################
	#############################
	
	TP = TowerPlanning(dim,main_cities,total_population,min_c,max_c,budget,int(RequiredRegions),NeighborsToCover,CostPerEach)
	TP.GeneratePopulation()
	
	try:
		result = pd.read_csv('Result.csv')
	except pd.errors.EmptyDataError :
		result = pd.DataFrame([],columns=headers)
	
	# coverage , budget  = Simulate(population,self.NeighborsToCover,budget,8)
	coverage , budget , ElapsedTime  = TP.Simulate()
	append_data = [int(RequiredRegions),coverage,budget,ElapsedTime]
	resu = pd.DataFrame([append_data],columns=headers)
	result = pd.concat([result,resu])
	result.to_csv('Result.csv',index=False)
	# plt.show()
	plt.close('all')


