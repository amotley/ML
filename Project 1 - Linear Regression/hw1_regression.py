import numpy as np
import sys

#python hw1_regression.py 4 4.5  X_train.csv y_train.csv X_test.csv
lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    ## ridge regression solution: w_rr = (lambda*I + X^TX)^(-1)X^Ty
    ## note: ridge regression can always be solved analytically
    identityDimension = X_train.shape[1]
    identity = np.identity(identityDimension)
    lambdaIdentity = lambda_input*identity
    xtranspose = X_train.transpose()
    xtransposex = np.matmul(xtranspose,X_train)
    matrixSum = lambdaIdentity + xtransposex
    inverseSum = np.linalg.inv(matrixSum)
    xtransposey = np.matmul(xtranspose, y_train)
    w_rr = np.matmul(inverseSum, xtransposey)
    return w_rr

wRR = part1()  # Assuming wRR is returned from the function
np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    ## Prior is based on lambda, sigma, and covariance X
    ## Prior = (lambda*I + sigma^(-2)X^TX)^-1
    ## Posterior = (Prior^-1 + sigma^(-2)x_0x_0^t)^-1 
    ## sigma_0^2 = sigma^2 + x_0^t*Prior*x_0
    ## After we choose our next x_0, we update the Prior to contain the new x_0.
    ## This is the same as the posterior.
    toReturn = []
    ## calculate initial prior based on X_train
    identityDimension = X_train.shape[1]
    identity = np.identity(identityDimension)
    lambdaIdentity = lambda_input*identity
    sigmaSquared = sigma2_input
    sigma = sigma2_input ** .5
    xtranspose = X_train.transpose()
    xtransposex = np.matmul(xtranspose,X_train)
    priorSum = lambdaIdentity + (sigma**(-2.0))*xtransposex
    prior = np.linalg.inv(priorSum)
    ## run through the learning algorithm 10 times
    dictionaryOfAlreadyChosenXIndices = {}
    for i in range(0,10):
    	## find x in X_test with highest sigma^2 value
    	## this is the same as the x_0 w/ the highest x_0^t*Prior*x_0 value
    	x_next_value = 0
    	x_next_index = 0
    	x_next_sigmaUpdate = 0
    	for x_index in range(0, X_test.shape[0]):
    		x_value = X_test[x_index]
    		value = np.matmul(np.matmul(x_value.transpose(), prior),x_value)
    		if value > x_next_sigmaUpdate and (x_index not in dictionaryOfAlreadyChosenXIndices):
    			x_next_value = x_value
    			x_next_index = x_index
    			x_next_sigmaUpdate = value

    	## add next x index to dictionary to keep track of it and avoid repeats
    	dictionaryOfAlreadyChosenXIndices[x_next_index] = x_next_index
    	## add next x index to the return list
    	## Add plus 1 since we want to return 1-indexing
    	toReturn.append(x_next_index+1)
    	## update sigma squared
    	sigmaSquared = sigmaSquared + x_next_sigmaUpdate
    	sigma = sigmaSquared ** .5
    	## update prior with new x
    	prior = np.linalg.inv(np.linalg.inv(prior) + (sigma ** (-2))*np.matmul(x_next_value,x_next_value.transpose()))
    return [toReturn]

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active, delimiter=",") # write output to file