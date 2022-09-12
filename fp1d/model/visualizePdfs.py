
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import norm, truncnorm, uniform
# The standard Normal distribution
from scipy.stats import gaussian_kde as gkde # A standard kernel density estimator

import time

startTime 	= 3.839e0;					# startTime -- [time unit]
endTime  	= 9.800e0;					# endTime -- [time unit]
dt = 5.0e-4;							# time-step
M = int((endTime - startTime) / dt)
iteration_number = 5122 # (5, 5122, 11762) # input parameter: change this
pCurrent = np.loadtxt('pCurrent.%d.dat' % iteration_number) 

x = np.loadtxt('x.dat')
var = 'logAreaOfGrain'

q_pInit = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.15.dat'))
q_pCalibrated = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.25.dat'))
q_pFinal = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.39.dat'))

pInit = q_pInit(x)
pCalibrated = q_pCalibrated(x)
pFinal = q_pFinal(x)

### plot

plt.figure()

initPlt, 		= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
currentPlt, 	= plt.plot(x, pCurrent, 'b-', linewidth=2, label='evolving Fokker-Planck pdf')

plt.legend(handles=[initPlt, calibratedPlt, finalPlt, currentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)

plt.xlim([0,15])
plt.ylim([0, 1.5 * np.max(pCurrent)])
plt.title('Fokker-Planck density at iteration %d/%d (time-step %.4f)' % (iteration_number, M, startTime + iteration_number * dt), fontsize=24) 
plt.xlabel('support', fontsize=18)
plt.ylabel('p.d.f.', fontsize=18)

		# graph(2) = plot(x,pInit, 'ro:', 'LineWidth',2);
		# graph(4) = plot(x,pCalibrated, 'mo-.', 'LineWidth',2);		
		# graph(3) = plot(x,pFinal, 'go--', 'LineWidth',2);
		# graph(1) = plot(x, pCurrent, 'b', 'LineWidth', 2);



def drawSamples(pCurrent, x):
	min_x = np.min(x)
	max_x = np.max(x)
	#
	listOfSamples = []
	numOfSamples = 1e5
	#
	M = np.max( pCurrent / (1 / (max_x - min_x) ) )
	#
	while len(listOfSamples) < numOfSamples:
		# x_sample = min_x + np.random.rand() * (max_x - min_x)
		x_sample = x[np.random.randint(0, len(x))]
		u = np.random.rand()
		index_sample =  np.where(x == x_sample)
		if u < pCurrent[index_sample] / ( M * 1/(max_x - min_x) ):
			listOfSamples += [x_sample]
			print('Accepted %d samples!' % len(listOfSamples))
	return listOfSamples

# listOfSamples = drawSamples(pCurrent, x) # debug
# plt.plot(x, pCurrent, 'b-') # debug
# q_pCurrent = gkde(np.array(listOfSamples)) # debug
# plt.plot(x, q_pCurrent(x), 'r-.') # debug


### compare with the original (before log transformation)
plt.figure()

listOfSamples = drawSamples(pCurrent, x)
listOfSamples = np.exp(np.array(listOfSamples))

var = 'areaOfGrain'
q_pInit = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.15.dat'))
q_pCalibrated = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.25.dat'))
q_pFinal = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.39.dat'))
q_pCurrent = gkde(listOfSamples)

x = np.linspace(0, 5e4, num=10000)
pInit = q_pInit(x)
pCalibrated = q_pCalibrated(x)
pFinal = q_pFinal(x)
pCurrent = q_pCurrent(x)

initPlt, 		= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
currentPlt, 	= plt.plot(x, pCurrent, 'b-', linewidth=2, label='evolving Fokker-Planck pdf')

plt.legend(handles=[initPlt, calibratedPlt, finalPlt, currentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)

plt.xlim([np.min(x), np.max(x)])
plt.ylim([0, 1.5 * np.max(pCurrent)])
plt.title('Fokker-Planck density at iteration %d/%d (time-step %.4f)' % (iteration_number, M, startTime + iteration_number * dt), fontsize=24) # change this
plt.xlabel('support', fontsize=18)
plt.ylabel('p.d.f.', fontsize=18)


plt.show()





