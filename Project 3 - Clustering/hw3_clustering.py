import numpy as np
import pandas as pd
import scipy as sp
import sys
import math
from operator import add

X = np.genfromtxt(sys.argv[1], delimiter = ",")
clusterNumber = 5
#python hw3_clustering.py X.csv
def KMeans(data):
	#perform the algorithm with 5 clusters and 10 iterations
	#initialize centroids as first 5 data points
	centerslist = [data[0], data[1], data[2], data[3], data[4]]
	numPoints = len(data)
	for iteration in range(10):
		centroidAssignmentHash = {}
		#for each data point, find the closest centroid and assign to that centroid
		for i in range(numPoints):
			dataPoint = data[i]
			closestCentroid = 0
			minDistance = float('inf')
			for j in range(clusterNumber):
				centroid = centerslist[j]
				distance = Distance(dataPoint, centroid)
				if (distance < minDistance):
					minDistance = distance
					closestCentroid = j		
			newAssignment = [dataPoint]
			if (str(closestCentroid) in centroidAssignmentHash.keys()):
				existingAssignment = centroidAssignmentHash[str(closestCentroid)]
				newAssignment = existingAssignment
				newAssignment.append(dataPoint)
			centroidAssignmentHash[str(closestCentroid)] = newAssignment
		#next, re-calculate centroids based on new data point assignments
		for centroid in range(clusterNumber):
			if str(centroid) in centroidAssignmentHash.keys():
				pointsInCentroid = centroidAssignmentHash[str(centroid)]
				numPointsInCentroid = len(pointsInCentroid)
				newCentroid = [0.0 for i in range(len(data[0]))]
				for point in pointsInCentroid:
					newCentroid = map(add, newCentroid, point)
				newCentroid = [i/float(numPointsInCentroid) for i in newCentroid]
				centerslist[centroid] = newCentroid

		filename = "centroids-" + str(iteration+1) + ".csv" #"i" would be each iteration
		np.savetxt(filename, centerslist, delimiter=",")
    
def EMGMM(data):
	pi=[1.0/float(clusterNumber),1.0/float(clusterNumber),1.0/float(clusterNumber),1.0/float(clusterNumber),1.0/float(clusterNumber)]
	mu=[data[0],data[1],data[2],data[3],data[4]]
	id = np.identity(len(data[0]))
	sigma = [id, id, id,id,id]
	phi = np.zeros((len(data), clusterNumber))
	phi_n = np.zeros((len(data), clusterNumber))

	for iteration in range(10):
		#Expectation step
		for cluster in range(clusterNumber):
			#first, calculate N(x|mu, sigma)
			sigma_cluster = sigma[cluster]
			sigma_det = np.linalg.det(sigma_cluster)
			sigma_inv = np.linalg.inv(sigma_cluster)
			mu_cluster = mu[cluster]
			pi_cluster = pi[cluster]
			for dataPointIndex in range(len(data)):
				e = np.matmul(np.matmul((data[dataPointIndex] - mu_cluster).transpose(),sigma_inv), (data[dataPointIndex] - mu_cluster))
				phi_dataPoint = pi_cluster * ((2*math.pi)**(-float(len(data[0]))/2.0)) * (sigma_det**(-.5)) * math.exp(-.5 * e)
				phi[dataPointIndex, cluster] = phi_dataPoint
			for dataPointIndex in range(len(data)):
				s = 0.0
				for x in phi[dataPointIndex]:
					s = s + x
				phi_n[dataPointIndex,:] = phi[dataPointIndex,:]/s

		#Maximixation step
		n_k = [0.0,0.0,0.0,0.0,0.0]
		for row in phi_n:
			n_k = n_k + row
		pi = n_k/float(len(data))
		for cluster in range(clusterNumber):
			s = np.matmul(phi_n[:, cluster].transpose(),data)
			mu[cluster] = s/n_k[cluster]
		for k in range(clusterNumber):
			sum_forSigma = np.zeros((len(data[0]),len(data[0])))
			for i in range(len(data)):
				xMinusMew = np.matrix(np.subtract(np.asarray(data[i]), np.asarray(mu[k])))
				xMinusMew_t = np.matrix(xMinusMew).T
				matmul = np.matmul(xMinusMew_t, xMinusMew)
				sum_forSigma = sum_forSigma + (phi_n[i, k] * matmul)
			sigma_k = sum_forSigma / n_k[k]
			sigma[k] = sigma_k

		for kluster in range(clusterNumber): #k is the number of clusters 
			f = "Sigma-" + str(kluster+1) + "-" + str(iteration+1) + ".csv"
			np.savetxt(f, sigma[kluster], delimiter=",")

		filename = "pi-" + str(iteration+1) + ".csv" 
		np.savetxt(filename, pi, delimiter=",") 
		filename = "mu-" + str(iteration+1) + ".csv"
		np.savetxt(filename, mu, delimiter=",")  #this must be done at every iteration

def Distance(point, centroid):
	squaredDistanceSum = 0
	for i in range(len(point)):
		squaredDistance = (point[i] - centroid[i])**2.0
		squaredDistanceSum = squaredDistanceSum + squaredDistance
	return squaredDistanceSum ** (.5)


KMeans(X);
EMGMM(X);