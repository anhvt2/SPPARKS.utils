#Python script to generate input microstructures for CP simulations
#This will take-in an existing microstructure, as well as a list of grain orientations
#The grain ID will be used to correlate each xyz coordinate with an orientation.
#We may want to do a microstructure from multiple orientations, in this case, we'll use two input
#Orientation distributions and select from each one depending on the grain centroid

import networkx as nx
import numpy as np
#import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import os
#from numba import jit
from datetime import datetime
from skimage.measure import label
from skimage import *
#from scipy.stats import mode
startTime = datetime.now()

################################################################################
#Specifiy the input file name. The output file will be titled "FileNameprocessed"
baseFileName = 'dump.potts.'
angleFileName = 'Uniform.csv'
firstFile = 1 # N of the first processed file
numberOfFiles = 20
lastFile = numberOfFiles + firstFile
nvx = 100
ny = 100
nz = 100
#What is our lattice spacing (in m)
dx = 10.0
#What is the maximum size to remove
smallCutoff = 10
#Should we remove small grains specified by "smallCutoff" (1 for yes, 0 for no)
removeSmallGrainsFlag = 1
#Should we try to reduce the number of unique Grain IDs as much as possible? (1 yes, 0 no)
#This returns sorted, contiguous grain IDs.
minimizeGrainIDs = 0
#At how many remaining small grains should we switch strategies (default ~ 1000)
removeMethodSwitch = 1000
#Should we get rid of quad-edges
quadEdgeFlag = 0
#Do we want to output a phase field SCULPT file?
phaseFlag = 0
#Should we sort the remaining grains in ascending order?
sortFlag = 1
#Should we calculate grain equivalent spherical diameter and add an output column?
volumeFlag = 1

#Define periodicity of boundaries 1 = periodic, 0 = non-periodic
xPeriod = 0
yPeriod = 0
zPeriod = 0

latticeValues = np.zeros((nvx, ny, nz))
totalSites = nvx * ny * nz #Determine the total number of sites.

################################################################################



#Put all grainIDs into a graph as nodes and represent bordering grains as edges
#Then assign "colors" to the graph using as few as possible with no-shared edges
#This is having a really hard time with odd-numbered dimensions and I don't quite know why...
#@jit
def grainOverlap(grainIDTable):

    #Determine the unique IDs and add them to the graph
    print("Creating graph")
    grainGraph = nx.Graph()
    print("Adding nodes")
    #grainGraph.add_nodes_from(np.unique(grainIDTable.iloc[:,0], return_counts = False))

    #Find the neighbors in the 6-cardinal directions for each lattice site. Make a new function
    print("Adding edges")
    grainGraph = findGraphEdges(grainGraph, grainIDTable)

    #Run greedy_color to assign colors
    print("Reducing unique IDs")
    colorPairs = nx.coloring.greedy_color(grainGraph, strategy=nx.coloring.strategy_largest_first, interchange =True)
    npTable = grainIDTable.as_matrix(columns=[1])
    npTable = npTable.astype(np.int)
    #print npTable

    print("Assigning new IDs")
#    #Now assign the new colors to the data. "colorPairs" is a dictionary, so we'll loop over the keys, mask our data, and assign the value
#    for key, value in list(colorPairs.items()):
#        #mask = key == grainIDTable.iloc[:,0]
#        print(key, value)
#        #I guess we need to have our vector match size
#        key_array = np.ones_like(grainIDTable.iloc[:,0]) * key
#
#        dfOut = grainIDTable.where(key_array[:] != grainIDTable.iloc[:,0], value)

    #This loop is pretty slow. We should be able to do it in one swoop
    mrange = np.arange(0,np.amax(npTable) + 1)
    mrange[colorPairs.keys()] = colorPairs.values()
    npTable = mrange[npTable]
    latticeValues = np.zeros((nvx, ny, nz))

    latticeValues[reverseIndexHelper(range(totalSites))] = npTable.flatten()

    return latticeValues, npTable

################################################################################

#Find the 6 nearest neighbors for all lattice sites and add them to the graph as edges
#@jit
def findGraphEdges(grainGraph,grainIDTable):

    #We only need to do this once for X, Y, and Z. Don't need + and -.
    #Shift X

    originalFlat = latticeValues[:-1,:,:].flatten()
    newFlat = latticeValues[1:,:,:].flatten()

    #It seems worthwile to get rid of identical pairs
    mask = originalFlat != newFlat
    originalList = originalFlat[mask].tolist()
    newList = newFlat[mask].tolist()

    #Zip the lists up and add to the graph
    print("Adding edges from X shift")
    grainGraph.add_edges_from(zip(originalList,newList))

    #Shift y
    originalFlat = latticeValues[:,:-1,:].flatten()
    newFlat = latticeValues[:,1:,:].flatten()

    mask = originalFlat != newFlat
    originalList = originalFlat[mask].tolist()
    newList = newFlat[mask].tolist()

    #Zip the lists up and add to the graph
    print("Adding edges from Y shift")
    grainGraph.add_edges_from(zip(originalList,newList))

    #Shift z
    originalFlat = latticeValues[:,:,:-1].flatten()
    newFlat = latticeValues[:,:,1:].flatten()

    mask = originalFlat != newFlat
    originalList = originalFlat[mask].tolist()
    newList = newFlat[mask].tolist()
    #Zip the lists up and add to the graph
    print("Adding edges from Z shift")
    grainGraph.add_edges_from(zip(originalList,newList))

    #output our lists
    #f2 = open('testfile','w+')
    #print(zip(originalList,newList), file = f2)

    return grainGraph

################################################################################

#Make the grain IDs contiguous and sort them
#Lets also assign an orientation within the loop
#@jit
def grainSort(grainIDTable):

    #Determine the number of unique values and how many times they occur
    valuesThere = np.sort(np.unique(grainIDTable.iloc[:,0], return_counts = False))

    #Assign the new IDs to the existing data set. I'm sure there's a better way to do it, but the loop isn't too slow
    j = 0
    newGrainID = grainIDTable.iloc[:,0]
    for i in valuesThere.flatten():
        #print(i)
        mask = grainIDTable.iloc[:,0] == i
        newGrainID[mask] = j
        eulerAngles[mask] = eulerIn.iloc[j,:]
        j = j + 1

    grainIDTable.iloc[:,0] = newGrainID

    return grainIDTable

################################################################################

#Remove grains with a total count equal to or less than a threshold
#@jit
def smallRemove(grainIDTable, latticeValues):

    #Determine the number of unique values and how many times they occur
    print("Finding small grains")
#    segmentedGrains, numLab = label(latticeValues, return_num= True,connectivity = 1)
#    valuesThere, frequencies = np.unique(segmentedGrains, return_counts = True)
#    freqMask = frequencies <= smallCutoff
#    freqValues = valuesThere[freqMask]
#
#    #Remove the original values from the array if they correspond to small grains.
#    #Make both a positive and negative stencil.
#    vectorMask = np.in1d(segmentedGrains, freqValues)
#    vectorMaskInverted = np.in1d(segmentedGrains, freqValues, invert=True)
#    smallMask = np.reshape(vectorMask,(nvx,ny,nz))
#    smallMaskInverted = np.reshape(vectorMaskInverted,(nvx,ny,nz))

    smallMask, smallMaskInverted, segmented = smallMasker(latticeValues)

    #Compare neighbors in the 6-cubic directions. If the neighbor isn't small, copy it
    #This should eliminate most small grains. We can re-run the script a few times to
    #get rid of grains fully surrounded by small grains.

    #Shift +/-X, +/-Y, +/-Zc
    rightShift = np.copy(segmented)
    rightShift[:-1,:,:] = smallMask[:-1,:,:] * segmented[1:,:,:]
    rightShift[99,:,:] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + rightShift
    smallMask, smallMaskInverted, segmented = smallMasker(segmentedMasked)
    del(rightShift)

    leftShift = np.copy(segmented)
    leftShift[1:,:,:] = smallMask[1:,:,:] * segmented[:-1,:,:]
    leftShift[0,:,:] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + leftShift
    smallMask, smallMaskInverted, segmented = smallMasker(segmentedMasked)
    del(leftShift)

    frontShift = np.copy(segmented)
    frontShift[:,:-1,:] = smallMask[:,:-1,:] * segmented[:,1:,:]
    frontShift[:,99,:] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + frontShift
    smallMask, smallMaskInverted, segmented = smallMasker(segmentedMasked)
    del(frontShift)

    backShift = np.copy(segmented)
    backShift[:,1:,:] = smallMask[:,1:,:] * segmented[:,:-1,:]
    backShift[:,0,:] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + backShift
    smallMask, smallMaskInverted, segmented = smallMasker(segmentedMasked)
    del(backShift)

    upShift = np.copy(segmented)
    upShift[:,:,:-1] = smallMask[:,:,:-1] * segmented[:,:,1:]
    upShift[:,:,99] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + upShift
    smallMask, smallMaskInverted, segmented = smallMasker(segmentedMasked)
    del(upShift)

    downShift = np.copy(segmented)
    downShift[:,:,1:]  = smallMask[:,:,1:] * segmented[:,:,:-1]
    downShift[:,:,0] = 0

    segmentedMasked = segmented * smallMaskInverted
    segmentedMasked = segmentedMasked + downShift

    segmentedGrains = label(segmented, connectivity = 1)
    valuesThere, frequencies = np.unique(segmentedGrains, return_counts = True)
    freqMask = frequencies <= smallCutoff
    freqValues = valuesThere[freqMask]

    del(downShift)

    #update our input values
    latticeValues = segmented
    grainIDTable.iloc[:,0] = latticeValues.flatten('F')



    #See if we have a lot small grains left, if so recursively run the function
    if np.size(freqValues) > removeMethodSwitch:
        print("Small grains still present, reruning")
        print(np.size(freqValues))
        return smallRemove(grainIDTable, latticeValues)
    else:
        print("Most small grains eliminated, switching strategies to get rid of the rest")
        print(np.size(freqValues))
        return grainIDTable.iloc[:,0], latticeValues

################################################################################
#Given a lattice, segment the grains, find the small ones, and return the mask, inverse mask, and segmented lattice
def smallMasker(lattice):

    segmentedGrains = label(lattice, connectivity = 1)
    valuesThere, frequencies = np.unique(segmentedGrains, return_counts = True)
    freqMask = frequencies < smallCutoff
    freqValues = valuesThere[freqMask]
    print ('Number of small grains ', np.size(freqValues))

    #Remove the original values from the array if they correspond to small grains.
    #Make both a positive and negative stencil.
    vectorMask = np.in1d(segmentedGrains, freqValues)
    vectorMaskInverted = np.in1d(segmentedGrains, freqValues, invert=True)
    smallMask = np.reshape(vectorMask,(nvx,ny,nz))
    smallMaskInverted = np.reshape(vectorMaskInverted,(nvx,ny,nz))

    return smallMask, smallMaskInverted, segmentedGrains

################################################################################

def grainEliminator(lattice):

    segmentedGrains = label(lattice, connectivity = 1)
    valuesThere, frequencies = np.unique(segmentedGrains, return_counts = True)
    freqMask = frequencies <= smallCutoff
    freqValues = valuesThere[freqMask]

    vectorMask = np.in1d(segmentedGrains, freqValues)
    vectorMaskInverted = np.in1d(segmentedGrains, freqValues, invert=True)
    smallMask = np.reshape(vectorMask,(nvx,ny,nz))
    smallMaskInverted = np.reshape(vectorMaskInverted,(nvx,ny,nz))
    print ('Starting graph-based reduction')

    #Make the new labels negative so they won't conflict with original range
    segmentedMask = label(smallMask.astype(np.int8), connectivity = 1, background = 0) *-1

    #We should mask the large values too, so we don't accidently assign a small value to another small one
    reLabel = label(lattice,connectivity = 1, background = 0)
    bigMask = reLabel * smallMaskInverted.astype(np.int8)

    #Lets  do all directions
    #X
    maskFlat = segmentedMask[:-1,:,:].flatten()
    newFlat = bigMask[1:,:,:].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)


    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph = nx.MultiGraph()
    grainGraph.add_edges_from(zip(originalList,newList))

    maskFlat = segmentedMask[1:,:,:].flatten()
    newFlat = bigMask[:-1,:,:].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)


    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph.add_edges_from(zip(originalList,newList))
    print('X')
    #Y
    maskFlat = segmentedMask[:,:-1,:].flatten()
    newFlat = bigMask[:,1:,:].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)

    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph.add_edges_from(zip(originalList,newList))

    maskFlat = segmentedMask[:,1:,:].flatten()
    newFlat = bigMask[:,:-1,:].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)

    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph.add_edges_from(zip(originalList,newList))
    print('Y')
    #Z
    maskFlat = segmentedMask[:,:,:-1].flatten()
    newFlat = bigMask[:,:,1:].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)

    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph.add_edges_from(zip(originalList,newList))

    maskFlat = segmentedMask[:,:,1:].flatten()
    newFlat = bigMask[:,:,:-1].flatten()

    maskFlat = np.ma.masked_where(maskFlat >= 0, maskFlat)
    maskFlat = np.ma.masked_where(newFlat == 0, maskFlat)

    newFlat = np.ma.masked_where(maskFlat >= 0, newFlat)
    newFlat = np.ma.masked_where(newFlat == 0, newFlat)

    originalList = maskFlat.tolist()
    newList = newFlat.tolist()

    grainGraph.add_edges_from(zip(originalList,newList))
    grainGraph.remove_node(None)
    print('Z')
    outPutArray = np.copy(segmentedMask)
    outPutArray = outPutArray + bigMask

    print ('Number of nodes ', grainGraph.number_of_nodes())
    #Go through all the large grain nodes, find the most common neighbor
    #and update the matrix
    for n in grainGraph:
        if n < 0:

            neigh = grainGraph.neighbors_iter(n)
            maxEdge = 0
            maxVal = 0
            for m in neigh:
                edge = grainGraph.number_of_edges(n,m)
                if edge > maxEdge:
                    maxEdge = edge
                    maxVal = m

                outPutArray = np.ma.where(outPutArray == n, maxVal, outPutArray)

    #See if we have any small grains left, if so recursively run the function

    smallMask, smallMaskInverted, segmented = smallMasker(outPutArray)
    if np.sum(smallMask) > 0:
        print("Small grains still present, reruning")
        print(np.sum(smallMask))
        return grainEliminator(outPutArray)
    else:
        return outPutArray.flatten('F'), outPutArray

################################################################################

#Initialize latticeValues and determine the unique grain IDs present
#@jit
def loadGrains():

    #Take the data from the dataframe into a 2D numpy array with x, y,z as the indicies
    #and grainID as the value

    latticeValues[reverseIndexHelper(range(nvx*ny*nz))] = dataIn.iloc[:,0]
    #First create a vector of unique grainIDs
    uniqueGrainIDs = np.unique(latticeValues)
    print(np.size(uniqueGrainIDs))
    return uniqueGrainIDs

################################################################################

#Helper function to translate from X,Y,Z to single indexing
#@jit
def indexHelper(x, y, z):

    index = x + nvx * (y + ny * z)
    return index

################################################################################

#Take a single index and return a tuple of X,Y,Z
#@jit
def reverseIndexHelper(i):
    x = np.mod(i ,nvx)
    i = np.divide(i, nvx)
    y = np.mod(i , ny)
    i = np.divide(i,ny)
    z = i
    return (x,y,z)



################################################################################

def grainVolumes(data,lattice,dx):
    ESDOut = np.zeros((nvx*ny*nz))
    segmentedGrains = label(lattice, connectivity = 1)
    valuesThere, frequencies = np.unique(segmentedGrains, return_counts = True)
    print (frequencies)
    volumes = np.zeros_like(frequencies,dtype=float)
    ESD = np.zeros_like(frequencies,dtype=float)
    volumes[:] = frequencies[:] * dx**3
    ESD[:] = 2*(0.75/np.pi*volumes[:])**(1/3.0)
    print (ESD)
    for i in range(np.size(valuesThere)):
        indices = np.argwhere(data.iloc[:,0] == valuesThere[i])
        ESDOut[indices] = ESD[i]

    print (ESDOut)
    return ESDOut


################################################################################
for i in range(firstFile, lastFile):
        cwd = os.path.dirname(__file__)
	inputFile = baseFileName + str(i)
	fileName = baseFileName + 'done' + str(i)
	print('Beginning file ' + str(i))

	#Here's the main part of the program
	#Load grains
	print("Loading Grains")
	dataIn = pd.read_csv(inputFile, sep=' ', skiprows = 9, header=None,index_col = 0)
	dataIn = dataIn.sort_index(axis=0)
	eulerAngles = np.zeros((nvx*ny*nz,3))
	angleName2 = os.path.join(cwd,angleFileName)
	eulerIn = pd.read_csv(angleName2, sep=',',header=None,index_col=None)
	print("Creating 3D lattice")
	uniqueIDs = loadGrains()


	#First get rid of any small grains
	if removeSmallGrainsFlag == 1:
		print("Beginning grain removal")
		#dataIn.iloc[:,0], latticeValues = smallRemove(dataIn, latticeValues)
		#Get rid of the others
		dataIn.iloc[:,0], latticeValues = grainEliminator(latticeValues)
		print("Finished grain removal", np.size(np.unique(dataIn.iloc[:,0])))

	#Now, lets reduce the number of unique grains as much as possible
	if minimizeGrainIDs == 1:
		print("Beginning grain number reduction")
		latticeValues, dataIn.iloc[:,0] = grainOverlap(dataIn)
		print("Finished grain number reduction")


	#Finally, lets make all of the grainIDs contiguous
	if sortFlag == 1:
	   print("Beginning data sorting")
	   dataIn = grainSort(dataIn)
	   print("Finished data sorting")

	#Calculate grain volumes
	if volumeFlag == 1:
	    print("Calculating grain volumes")
	    #Recalculate latticeValues
	    latticeValues = np.zeros((nvx, ny, nz))
	    uniqueIDs = loadGrains()
	    ESDs = grainVolumes(dataIn,latticeValues,dx)

	#Append the orientation array
	dfAngle = pd.DataFrame({'phi1':eulerAngles[:,0],'Phi':eulerAngles[:,1],'phi2':eulerAngles[:,2]})
	dfAngle.index = np.arange(1, len(dfAngle) + 1)
	dataIn.index = np.arange(1,len(dataIn) + 1)
	#I think this will work but dont know for sure...
	dfESD = pd.DataFrame({'ESD':ESDs})
	dfESD.index = dfAngle.index
	dataTemp = pd.concat([dataIn,dfAngle],axis=1)
	dataTemp2 = pd.concat([dataTemp,dfESD],axis=1)
	print (dataTemp2)
	print(datetime.now() - startTime)
	print(np.size(np.unique(dataIn.iloc[:,0])))
	#See how many we're left with
	print(np.unique(latticeValues))

	dataOut = dataTemp2.sort_index(axis=0)

	header = 'x y z phi1 phi2 phi3 grainID ESD\n'
	plt.hist(dataTemp2['ESD'])
    #Open the file and write the header
	with open(fileName, mode = 'w') as f:

		f.write(header)
		#Write the first column of data
		dataOut.to_csv(f,columns=(2,3,4,'phi1','phi2','Phi',1, 'ESD'),header=False,sep=' ', index=False)

		f.close()
	print('Finished file ' + str(i))
