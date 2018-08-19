from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter = ",")

lam = 2
sigma2 = 0.1
d = 5

# Implement function here
#python hw4_PMF.py ratings.csv
def PMF(train_data):
	L = np.zeros((50, 1))
	numUsers = int(np.amax(train_data[:,0]))
	numObjects = int(np.amax(train_data[:,1]))
	U_matrices = np.zeros((50, numUsers, d))
	V_matrices = np.zeros((50, numObjects, d))
	#initialize V, where vj~N(0, lam^-1*I)
	V = np.random.normal(0, (1/lam)**(.5), (numObjects, d))
	U = np.zeros((numUsers, d))
	#create M based on train_data input
	M = np.zeros((numUsers, numObjects))
	for data in train_data:
		M[int(data[0])-1,int(data[1])-1]=data[2]

	# iterate 50 times
	for iteration in range(0,50):
		#update user location
		for i in range(0, numUsers):
			lamSigI = lam*sigma2*np.identity(d)
			#find indices of objects rated by this user
			objectsRatedByUser = []
			colIndex = 0
			for rating in M[i]:
				if rating != 0:
					objectsRatedByUser.append(int(colIndex))
				colIndex = colIndex + 1
			objects = V[objectsRatedByUser]
			sum1 = np.matmul(objects.transpose(), objects)
			firstTerm = np.linalg.inv(lamSigI + sum1)
			ratings = M[i, objectsRatedByUser]
			sum2 = np.matmul(ratings, objects)
			newU = np.matmul(firstTerm, sum2)
			U[i] = newU

		#update object location
		for j in range(0, numObjects):
			lamSigI = lam*sigma2*np.identity(d)
			#find indices of users who rated this object
			UsersWhoRatedObject = []
			rowIndex = 0
			for rating in M[:,j]:
				if rating != 0:
					UsersWhoRatedObject.append(int(rowIndex))
				rowIndex = rowIndex + 1
			users = U[UsersWhoRatedObject]
			sum1 = np.matmul(users.transpose(), users)
			firstTerm = np.linalg.inv(lamSigI + sum1)
			ratings = M[UsersWhoRatedObject, j]
			sum2 = np.matmul(ratings, users)
			newV = np.matmul(firstTerm, sum2)
			V[j] = newV
		U_matrices[iteration] = U
		V_matrices[iteration] = V

		#find value of L(objective function) for this iteration
		sum1 = 0.0
		rowIndex = 0
		for row in M:
			colIndex = 0
			for col in row:
				if col != 0:
					sum1 = sum1 + (col - np.matmul(U[rowIndex],V[colIndex]))**2.0
				colIndex = colIndex + 1
			rowIndex = rowIndex + 1
		sum1 = sum1/(2.0*sigma2)
		sum2 = 0.0
		for u in U:
			sum2 = sum2 + (np.linalg.norm(u))**2.0
		sum2 = sum2*lam/2.0
 		sum3 = 0.0
		for v in V:
			sum3 = sum3 + (np.linalg.norm(v))**2.0
		sum3 = sum3*lam/2.0
		L[iteration] = -sum1-sum2-sum3

	return L, U_matrices, V_matrices

# Assuming the PMF function returns Loss L, U_matrices and V_matrices (refer to lecture)
L, U_matrices, V_matrices = PMF(train_data)

np.savetxt("objective.csv", L, delimiter=",")

np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("V-50.csv", V_matrices[49], delimiter=",")