
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

from scipy.stats import norm, truncnorm, uniform
# The standard Normal distribution
from scipy.stats import gaussian_kde as gkde # A standard kernel density estimator

import time

startTime 	= 3.839e0;					# startTime -- [time unit]
endTime  	= 9.800e0;					# endTime -- [time unit]
dt = 5.0e-4;							# time-step
M = int((endTime - startTime) / dt)

logTime = np.loadtxt('../probabilityRepo/log.logtime.dat')[15:] # training dataset starts
# iterationList = np.array((logTime - startTime) / dt, dtype=int) # start index offset
# iterationList = np.loadtxt('iterationList.dat')
iterationList = np.array([1, 502, 1022, 1542, 2042, 2562, 3062, 3582, 4082, 4602, 5122, 5622, 6142, 6642, 7162, 7682, 8182, 8702, 9202, 9722, 10222, 10742, 11262, 11762])
# log.16:39 in SPPARKS?

print(iterationList)

# for ii in iterationList:
for i in range(len(iterationList)):
	ii = iterationList[i]
	startTimePython = datetime.datetime.now()

	iteration_number = ii # (5, 5122, 11762) # input parameter: change this
	spkIndex = i + 16 # reference i to log.spparks
	pCurrent = np.loadtxt('pCurrent.%d.dat' % iteration_number) 

	x = np.loadtxt('x.dat')
	var = 'logAreaOfGrain'

	q_pInit = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.15.dat'))
	q_pCalibrated = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.25.dat'))
	q_pFinal = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.39.dat'))
	q_pSpkCurrent = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.%d.dat' % spkIndex))

	pInit = q_pInit(x)
	pCalibrated = q_pCalibrated(x)
	pFinal = q_pFinal(x)
	pSpkCurrent = q_pSpkCurrent(x)

	### plot

	plt.figure()

	initPlt, 		= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
	calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
	finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
	currentPlt, 	= plt.plot(x, pCurrent, 'b-', linewidth=2, label='evolving Fokker-Planck pdf')
	spkCurrentPlt,  = plt.plot(x, pSpkCurrent, c='violet', linestyle='--', linewidth=2, label='SPPARKS pdf')


	plt.legend(handles=[initPlt, calibratedPlt, finalPlt, currentPlt, spkCurrentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
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
		numOfSamples = 1e4
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
				# print('Accepted %d samples!' % len(listOfSamples)) # debug
		return listOfSamples

	# listOfSamples = drawSamples(pCurrent, x) # debug
	# plt.plot(x, pCurrent, 'b-') # debug
	# q_pCurrent = gkde(np.array(listOfSamples)) # debug
	# plt.plot(x, q_pCurrent(x), 'r-.') # debug


	### compare with the original (before log transformation)
	# fig2 = plt.figure()
	fig2 = plt.figure(num=None, figsize=(20, 11.3), dpi=300, facecolor='w', edgecolor='k') # screen size

	listOfSamples = drawSamples(pCurrent, x)
	listOfSamples = np.exp(np.array(listOfSamples))

	var = 'areaOfGrain'
	q_pInit = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.15.dat'))
	q_pCalibrated = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.25.dat'))
	q_pFinal = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.39.dat'))
	q_pCurrent = gkde(listOfSamples)
	q_pSpkCurrent = gkde(np.loadtxt('../probabilityRepo/' + 'log.' + var + '.%d.dat' % spkIndex))

	x = np.linspace(0, 5e4, num=10000)
	pInit = q_pInit(x)
	pCalibrated = q_pCalibrated(x)
	pFinal = q_pFinal(x)
	pCurrent = q_pCurrent(x)
	pSpkCurrent = q_pSpkCurrent(x)

	initPlt, 		= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
	calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
	finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
	currentPlt, 	= plt.plot(x, pCurrent, 'b-', linewidth=2, label='evolving Fokker-Planck pdf')
	spkCurrentPlt,  = plt.plot(x, pSpkCurrent, c='violet', linestyle='--', linewidth=2, label='SPPARKS pdf')

	plt.legend(handles=[initPlt, calibratedPlt, finalPlt, currentPlt, spkCurrentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.tick_params(axis='both', which='minor', labelsize=24)

	# plt.xlim([np.min(x), np.max(x)])
	# plt.ylim([0, 1.5 * np.max(pCurrent)])
	plt.title('Fokker-Planck density at iteration %d/%d (time-step %.4f)' % (iteration_number, M, startTime + iteration_number * dt), fontsize=24) # change this
	plt.xlabel('support', fontsize=18)
	plt.ylabel('p.d.f.', fontsize=18)

	### save fig


	plt.savefig('gg_FokkerPlanck_iter%d.png' % ii, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

	elapsedTimeInSeconds = datetime.datetime.now() - startTimePython

	print('Done iteration %d. Elapsed time: %.2f seconds' % (ii, elapsedTimeInSeconds.total_seconds()))
	# plt.show()




