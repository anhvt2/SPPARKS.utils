import numpy as np
import os

iterNum, exploitSize, exploreSize = np.loadtxt('log.iteration.txt', dtype=str)
iterNum, exploitSize, exploreSize = int(iterNum), int(exploitSize), int(exploreSize)

booleanIndex = True;

def checkComplete(fileName):
	# check only if log.dat exists after simulation
	if os.path.isfile(fileName) and 'done' in open(fileName).read(): 
		print('%s is done' % fileName)
		updateBoolean = True
	else:
		print('%s is not done' % fileName)
		updateBoolean = False
	return updateBoolean

for j in range(1, exploitSize + 1):
	updateBoolean = checkComplete('ForwardFokkerPlanckModel_Iter%d_Exploit_Batch%d/log.status.txt' % (iterNum, j))
	booleanIndex *= updateBoolean

for j in range(1, exploreSize + 1):
	updateBoolean = checkComplete('ForwardFokkerPlanckModel_Iter%d_Explore_Batch%d/log.status.txt' % (iterNum, j))
	booleanIndex *= updateBoolean

outFile = open('log.sim_status.txt','w')
if booleanIndex:
	outFile.write('1\n')
else:
	outFile.write('0\n')

outFile.close()

