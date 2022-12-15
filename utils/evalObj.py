
# objective: evaluate KL divergence 
# 	on the microstructure statistical descriptor

import numpy as np
import os, sys, glob
from scipy.stats import entropy # KL divergence
from scipy.stats import gaussian_kde # as gaussian_kde # import kde 

currenDirectory = os.getcwd()
targetDirectory = currenDirectory + '/../targetSample/'

def computeKullbackLeiblerDivergence(trainList, testList):
	trainPdf = gaussian_kde(trainList)
	testPdf = gaussian_kde(testList)
	# build the same support
	q = np.linspace(np.min(np.hstack([trainList, testList])) - 0.5, 
		1.1 * np.max(np.hstack([trainList, testList])), num=int(1e4))

	## DEPRECATED due to numerically instability
	# # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.entropy.html
	# return entropy(trainPdf(q), testPdf(q))

	## compute using np.trapz()
	# define a tolerance
	tol = 1e-14 # tol = np.finfo(float).eps
	# convert to np.array
	trainpdf = trainPdf(q)
	testpdf = testPdf(q)
	# check if integral is normalized
	print('np.trapz(trainpdf,q) = %.6e' % np.trapz(trainpdf,q))
	print('np.trapz(testpdf,q) = %.6e' % np.trapz(testpdf,q))
	# normalize
	trainpdf = trainPdf(q) / np.trapz(trainpdf,q)
	testpdf = testPdf(q) / np.trapz(testpdf,q)
	# define an integral
	integralFunction = np.zeros(q.shape)
	for i in range(q.shape[0]):
		if trainpdf[i] > tol and testpdf[i] > tol:
			tmpVal = trainpdf[i] * np.log( trainpdf[i] / testpdf[i] )
			# check if tmpVal is not overflow
			if isinstance(tmpVal, float) and not np.isnan(tmpVal):
				integralFunction[i] = tmpVal
	KullbackLeiblerDivergence = np.trapz(integralFunction, q)
	print('KullbackLeiblerDivergence (scipy) = %.6e' % entropy(trainPdf(q), testPdf(q)))
	print('KullbackLeiblerDivergence (manual) = %.6e' % KullbackLeiblerDivergence)
	return KullbackLeiblerDivergence

### run KullbackLeiblerDivergence for all statistical descriptors
outFile = open('objVal.out', 'w+')
objVal = []

# for statsDescriptor in ['a','b','theta','grainArea']: # ,'chordLengthX','chordLengthY']: # fast debug
MsDescriptorList = 	[	# 'a',
						# 'b', 
						# 'theta',
						'grainArea',
						# 'chordLengthX','chordLengthY',
						# 'xc', 
						]
# MsDescriptorList += [x.replace('.stat.txt','') for x in  glob.glob('localChordLengthY_Band*')]

for statsDescriptor in MsDescriptorList:
	print('Processing statsDescriptor = %s ' % statsDescriptor)
	trainList = np.loadtxt(targetDirectory + '/' + statsDescriptor + '.stat.txt')
	testList = np.loadtxt(currenDirectory + '/' + statsDescriptor + '.stat.txt')
	KLDistance = computeKullbackLeiblerDivergence(trainList, testList)
	print('computeKullbackLeiblerDivergence(%s) = %.6f\n' % (statsDescriptor, KLDistance))
	objVal.append(KLDistance) # append the list of output to objVal list
	outFile.write('%.6e\n' % KLDistance)

outFile.close()

### adopt another output.dat for moBO

outFile = open('output.dat', 'w+')
combinedObj = - np.sum(objVal)  # combined objective
outFile.write('%.8f\n' % (combinedObj))
outFile.close()

feasibleFile = open('feasible.dat', 'w+')
feasibleFile.write('%d\n' % (not np.any(np.isinf(objVal)))) # check if any objVal is 'float(inf)'
feasibleFile.close()

completeFile = open('complete.dat', 'w+')
completeFile.write('%d\n' % 1)
completeFile.close()

rewardFile = open('rewards.dat', 'w+')
rewardFile.write('%.8f\n' % (- combinedObj)) # (* -1) for minimization; (* +1) for maximization
rewardFile.close()




