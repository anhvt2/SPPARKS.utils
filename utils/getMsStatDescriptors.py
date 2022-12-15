
# Author: Anh Tran (Sandia)
# 20Jun19
# get multiple mircostructure descriptors for a microstructure

# how to fit contours:
# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

import glob,sys
import numpy as np 
import skimage
import skimage.measure
from scipy.stats import gaussian_kde # as gaussian_kde # import kde 
import matplotlib
matplotlib.use('Agg') # HPC backend: 'Agg' or 'pdf'
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/12282232/how-do-i-count-unique-values-inside-a-list
# count the number of unique values
from collections import Counter 

# objective: get microstructure and its corresponding physical descriptors
#	(1) grain ellipse: 5 QoIs: location (2), principal axes (2), orientation
#	(2) chord-length distribution # adopt from getChordLengthAlong{X,Y}.py


# schematic explanation:
# -----------------------------------------------------------------
# |                                                               |
# |                                                               |
# |                                                               |
# |                                                               |
# |                                                               |
# |                                                               |
# |                                                               |
# -----------------------------------------------------------------
# origin here: x-axis: vertical; y-axis: horizontal

### documentation
# input:
# 	pixelsThreshold: only consider grain whose size is larger than pixelsThreshold
#	kwargs: func(grainID=grainID, grainIDArray=grainIDArray)

# xArray, yArray, zArray -- list of locations of pixels, corresponding to the same grainID

# functions:
# 	fitEllipse():
#		input: xy -- (x,y) coordinates of a discretized contour of a grain
#		output: 
# 			fitEllipseBool: boolean -- whether the underlying algorithm converges
# 			xc, yc -- location of a grain
#			a, b -- principal axis of the ellipse
# 			theta -- orientation (in radians)
#
#	getGrainBoundary():
#		input:
#			grainID -- ID of a grain
#			grainArray -- build from dump file of a SPPARKS simulation
#		output:
#			xyContourList -- list of discretized points on the contour of a grain
#
#	getChordLengthX():
#		input:
#			grainID -- ID of a grain
#			grainArray -- build from dump file of a SPPARKS simulation
#		output:
#			localChordLengthXList -- list of chord length in x-direction
#
#	getChordLengthY():
#		input:
#			grainID -- ID of a grain
#			grainArray -- build from dump file of a SPPARKS simulation
#		output:
#			localChordLengthYList -- list of chord length in y-direction
#
# 	## deprecated: getChordLengthY_ConstX(**localKwargs): 
#		input:
#			grainID
#			grainArray
#			xLocation: specify x-location on the specimen
# 			bandWidth: width of the band
#		output:
#			localChordLengthYList_ConstX: list of chord length, given constant x, and a bandWidth
#
#	getGrainArea():
#		input:
#			grainID -- ID of a grain
#			grainArray -- build from dump file of a SPPARKS simulation
#		output: number of pixels corresponding to grainID
#
#	filterGrain():
# 		objective:
#			filter out too small grains that are not a concern
#			usually grains that are involed in the initialization process
#		input: pixelsThreshold (global value)
#		output: 
#			boolean PassOrFail -- whether the grain passes the filter or not
#
#	plotMsStatDescriptor():
#		input:
#			aList -- a list of samples (population samples)
#		output:
#			a plot


### input params
global pixelsThreshold
pixelsThreshold = 150 # 100


### user-defined functions:


## fit ellipse


def fitEllipse(xy):
	# xy : (N,2) -  2d contour data
	## test
	# xy = skimage.measure.EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25), 
	# 			params=(10, 15, 4, 8, np.deg2rad(30)))
	ellipse = skimage.measure.EllipseModel()
	if ellipse.estimate(xy):
		xc, yc, a, b, theta = ellipse.params # note: theta in radians
		fitEllipseBool = True
		return fitEllipseBool, xc, yc, a, b, theta
	else:
		fitEllipseBool = False
		return fitEllipseBool, [], [], [], [], []


def getGrainBoundary(**kwargs):
	## objective: get a list of pixels (x,y) that contours a grainID
	# given a grainIDArray
	# NOTE: ONLY WORKS FOR 2D, DO NOT WORK FOR 3D
	## implement a boundary tracing or contour tracing algorithm	
	# look up a grainID in a grainIDArray and 
	# 	return the location of the pixels
	indexArray = (np.where(grainIDArray == grainID))
	if indexArray == []:
		print('grainID %d is not contained in grainIDArray' % (grainID))
		return []
	xArray, yArray, zArray = indexArray[0], indexArray[1], indexArray[2]
	# complementary comutation; do not use, can comment out
	numOfPixels = len(xArray)
	pixelLocationList = np.zeros([numOfPixels, 3])
	for i in range(numOfPixels):
		pixelLocationList[i] = [xArray[i], yArray[i], zArray[i]]
	# create msSegmented that is 0 everywhere
	# 	and 1 at pixelLocationList
	msSegmented = np.zeros(grainIDArray.shape)
	for i in range(numOfPixels):
		x = xArray[i]; y = yArray[i]; z = zArray[i]
		msSegmented[x, y, z] = 1
	xyContourList = skimage.measure.find_contours(msSegmented[:,:,0], 0.5)[0]
	return xyContourList


def getGrainArea(**kwargs):
	## objective: get the number of pixels/voxels that corresponds to a particular grainID
	# given a grainIDArray
	indexArray = (np.where(grainIDArray == grainID))
	if indexArray == []:
		print('grainID %d is not contained in grainIDArray' % (grainID))
		return []
	xArray, yArray, zArray = indexArray[0], indexArray[1], indexArray[2]
	# complementary comutation; do not use, can comment out
	numOfPixels = len(xArray)
	return numOfPixels


def getChordLengthX(**kwargs):
	indexArray = (np.where(grainIDArray == grainID))
	if indexArray == []:
		print('grainID %d is not contained in grainIDArray' % (grainID))
		return []
	xArray, yArray, zArray = indexArray[0], indexArray[1], indexArray[2]
	# init
	localChordLengthXList = list(Counter(xArray).values())
	return localChordLengthXList


def getChordLengthY(**kwargs):
	grainID = kwargs['grainID']
	grainIDArray = kwargs['grainIDArray']
	# debug
	# print('def func:')
	# for key, value in kwargs.items(): # debug
	# 	print("%s == %s" % (key, value)) 
	print('def func: grainID == %d' % grainID) # debug
	# 
	indexArray = (np.where(grainIDArray == grainID))
	if indexArray == []:
		print('grainID %d is not contained in grainIDArray' % (grainID))
		return []
	xArray, yArray, zArray = indexArray[0], indexArray[1], indexArray[2]
	# init
	localChordLengthYList = list(Counter(yArray).values())
	# print(grainIDArray) # debug
	# print(grainIDArray.shape) # debug
	# print(indexArray) # debug
	# print(yArray) # debug
	# print(localChordLengthYList) # debug
	return localChordLengthYList


def filterGrain(**kwargs):
	indexArray = (np.where(grainIDArray == grainID))
	xArray, yArray, zArray = indexArray[0], indexArray[1], indexArray[2]
	numOfPixels = len(xArray)
	# implement a filter to filter out too small grains
	if numOfPixels > pixelsThreshold:
		print('The area of grain %d is %d > %d: passed.' % (grainID, numOfPixels, pixelsThreshold))
		PassOrFail = True
	else:
		print('The area of grain %d is %d < %d: failed.' % (grainID, numOfPixels, pixelsThreshold))
		PassOrFail = False
	return PassOrFail


def plotMsStatDescriptor(aList,XLabel='QoI',YLabel='pdf', PlotTitle='pdf plot', FileName='kdeQoI.png', showBoolean=False): 
	aKDE = gaussian_kde(aList)
	aplot = np.linspace(-0.5, 1.1 * np.max(aList), num=1000)
	plt.clf()
	aPlot = plt.plot(aplot, aKDE(aplot), 'b-', linewidth=4, label='chordLength pdf')
	plt.plot(aList, np.zeros(aList.shape), 'ro', ms=5, mew=5)
	plt.ylabel(YLabel, fontsize=24)
	plt.axes().yaxis.get_major_formatter().set_powerlimits((0, 1))
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.tick_params(axis='both', which='minor', labelsize=24)
	plt.title(PlotTitle + ': %d samples' % len(aList), fontsize=36)
	plt.savefig(FileName)
	if showBoolean == True:
		plt.show()

### ------------------------------------------------ main function

## read dump data from SPPARKS
# format: id type(grainID) x y z energy
dumpFileName = 'dump.gg.23.out'
# dumpFileName = glob.glob('dump.gg.*.out')[0]

# get W L H variables from dump files
dumpFile = open(dumpFileName)
for i, line in enumerate(dumpFile):
	if i == 5: # 6-th line
		W = int(line.split(' ')[1])
	elif i == 6: # 7-th line
		L = int(line.split(' ')[1])
	elif i == 7: # 8-th line
		H = int(line.split(' ')[1])
	elif i > 7: 
		break

dumpFile.close()

# export to grainID array
lastDump = np.loadtxt(dumpFileName, skiprows=9)
grainIDArray = np.zeros([W, L, H]) # init

for lineNum in range(lastDump.shape[0]):
	line = lastDump[lineNum, :]
	grainID, x, y, z = line[1], line[2], line[3], line[4]
	x, y, z = int(x), int(y), int(z) # float to int 
	# write to grainIDArray
	# print(x,y,z) # debug
	grainIDArray[x,y,z] = grainID

del lastDump # free mem
del grainID
# init
xcList, ycList, aList, bList, thetaList, grainAreaList = [], [], [], [], [], []
chordLengthXList, chordLengthYList = [], []

# get a unique list of grainID
grainIDList = np.unique(grainIDArray).astype(int)

# loop over all the grains # process all grainID
for grainID in grainIDList:
	# parse inputs
	kwargs = {"grainID": grainID, "grainIDArray": grainIDArray}
	# collect the stats ONLY IF the grain passes through filter
	if filterGrain(**kwargs): 
		print('grain %d passed through the filter, processing...' % (grainID))
		# get grain area stats
		grainArea = getGrainArea(**kwargs)
		# get fittedEllipse stats
		xyContourList = getGrainBoundary(**kwargs)
		fitEllipseBool, xc, yc, a, b, theta = fitEllipse(xyContourList)
		print('fitEllipse() on grainID = %d' % (grainID))
		# if the fitEllipse() converge then add statistics
		if fitEllipseBool:
			xcList.append(xc)
			ycList.append(yc)
			aList.append(a)
			bList.append(b)
			thetaList.append(theta)
			grainAreaList.append(grainArea)
		# aggregate chord length stats
		chordLengthXList += getChordLengthX(**kwargs)
		chordLengthYList += getChordLengthY(**kwargs)


# if the fitEllipse fails
if np.sort(aList)[-1] / np.sort(aList)[-2] > 1e0:
	maxIndex = np.argmax(aList)
	del xcList[maxIndex]
	del ycList[maxIndex]
	del aList[maxIndex]
	del bList[maxIndex]
	del thetaList[maxIndex]
	del grainAreaList[maxIndex]

xcList, ycList, aList, \
bList, thetaList, grainAreaList = \
	np.array(xcList), np.array(ycList), np.array(aList), \
	np.array(bList), np.array(thetaList), np.array(grainAreaList)

## visualize pdf
# plotMsStatDescriptor(grainAreaList, 
# 	XLabel='Grain Area', 
# 	PlotTitle='Grain Area KDE Statistics', 
# 	FileName='grainArea.png',
# 	showBoolean=True)

## save statistics
np.savetxt('a.stat.txt', aList, fmt='%.2f', delimiter='\n')
np.savetxt('b.stat.txt', bList, fmt='%.2f', delimiter='\n')
np.savetxt('xc.stat.txt', xcList, fmt='%.2f', delimiter='\n')
np.savetxt('yc.stat.txt', ycList, fmt='%.2f', delimiter='\n')
np.savetxt('theta.stat.txt', xcList, fmt='%.2f', delimiter='\n')
np.savetxt('grainArea.stat.txt', grainAreaList, fmt='%.2f', delimiter='\n')
np.savetxt('chordLengthX.stat.txt', chordLengthXList, fmt='%.2f', delimiter='\n')
np.savetxt('chordLengthY.stat.txt', chordLengthYList, fmt='%.2f', delimiter='\n')



# ## ------------------------------- added on 24Jul19 for accounting for ms heterogeneity 

# # aggregate local chord length stats
# numOfBands = 5 # assuming symmetric, bands go both ways
# bandSpacing = 30 
# bandWidth = 20 # constraints: cannot be too wide, cannot be too thin # total band width 
# xCoordOfWeldAxis = np.floor(W/2.) # x-coordinate location of the weld axis

# ## safeguards: 2 constraints: (a) band not overlap (b) band not go beyond boundary
# if numOfBands * (bandSpacing + bandWidth) > W/2.:
# 	print('getMsStatDescriptor.py: constraints violated: bands should not go beyond boundary!')

# if bandSpacing < bandWidth:
# 	print('getMsStatDescriptor.py: constraints violated: bands should not overlap!')	

# ## either specify numOfBands or bandSpacing
# for i in range(numOfBands): # i=0 corresponds to weld axis
# 	# xLoc = f(W, numOfBands) # , bandWidth)
# 	localChordLengthXList, localChordLengthYList = [], [] # init
# 	for dumIdx in [-1, 1]: # go up or down from the weld axis
# 		xLoc = xCoordOfWeldAxis + dumIdx * i * bandSpacing # dumIdx = +/- 1
# 		localGrainIDArray = grainIDArray[np.arange(xLoc - bandWidth/2., xLoc + bandWidth/2. + 1, dtype=int), :, :]
# 		localGrainIDList = np.unique(localGrainIDArray).astype(int)	# shape: (bandWidth + 1, L, H)
# 		for localGrainID in localGrainIDList:
# 			localKwargs = {"grainID": localGrainID, "grainIDArray": localGrainIDArray}
# 	# kwargs2 = {"grainIDArray": grainIDArray,
# 	# 			"xLocation": xLoc,
# 	# 			"bandWidth": bandWidth,
# 	# 			"bandSpacing": bandSpacing}
# 			print('main func: grainID == %d' % localGrainID)
# 			localChordLengthYList += getChordLengthY(**localKwargs)
# 			print('done %d\n' % localGrainID) # debug
# 	print('getMsStatDescriptor.py: localChordLengthYList: done processing band %d' % i)
# 	np.savetxt('localChordLengthY_Band%d.stat.txt' % i, localChordLengthYList, fmt='%.2f', delimiter='\n')
	

