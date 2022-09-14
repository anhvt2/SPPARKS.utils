
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime

from scipy.stats import norm, truncnorm, uniform
# The standard Normal distribution
from scipy.stats import gaussian_kde as gkde # A standard kernel density estimator

import time


"""
The script compares the prediction between (1) SPPARKS ICME model and (2) calibrated FPE model
in extrapolation time. The FPE solver is only activated after the training stops.

As such, this comparison is done purely at the testing time.
"""

startTime 	= 3.839e0;					# startTime -- [time unit]
endTime  	= 9.800e0;					# endTime -- [time unit]
dt = 5.0e-4;							# time-step
M = int((endTime - startTime) / dt)

logTime = np.loadtxt('../probabilityRepo/log.logtime.dat')[15:] # training dataset starts
# iterationList = np.array((logTime - startTime) / dt, dtype=int) # start index offset
# iterationList = np.loadtxt('iterationList.dat')
iterationList = np.array([1, 502, 1022, 1542, 2042, 2562, 3062, 3582, 4082, 4602, 5122, 5622, 6142, 6642, 7162, 7682, 8182, 8702, 9202, 9722, 10222, 10742, 11262, 11762]) # this list length is only 24
# log.16:39 in SPPARKS?
spkTimeList = [0, 1, 1.375, 1.75, 2.25, 2.875, 3.625, 4.75, 6, 7.75, 10, 13, 16.75, 21.625, 27.875, 36, 46.5, 60, 77.5, 100, 129.25, 166.875, 215.5, 278.375, 359.5, 464.25, 599.5, 774.375, 1000, 1291.62, 1668.12, 2154.5, 2782.62, 3593.88, 4641.62, 5994.88, 7742.75, 10000, 12915.5, 16681.1, 20000]

print(iterationList)

# for ii in iterationList:
for i in range(len(iterationList)):
# for i in range(len(iterationList)-1, -1, -1): # flip from the end to the beginning to get more images in shorter time (due to KDE)
	ii = iterationList[i]
	startTimePython = datetime.datetime.now()

	iteration_number = ii # (5, 5122, 11762) # input parameter: change this
	spkIndex = i + 16 # reference i to log.spparks -- 16 SPPARKS iterations has been used for training
	# note: training starts at 46.5 mcs to 599.5 mcs
	# testing starts at 599.5 mcs to 16681.1 mcs
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

	# initPlt, 			= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
	# calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
	# finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
	currentPlt, 	= plt.plot(x, pCurrent, c='red', linestyle='--', linewidth=4, label='Fokker-Planck pdf')
	spkCurrentPlt,  = plt.plot(x, pSpkCurrent, c='blue', linestyle='-', linewidth=4, label='SPPARKS pdf')


	plt.legend(handles=[currentPlt, spkCurrentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.tick_params(axis='both', which='minor', labelsize=24)

	plt.xlim([0,15])
	plt.ylim([0, 1.5 * np.max(pCurrent)])
	plt.title('Fokker-Planck density at iteration %d/%d (time-step %.4f)' % (iteration_number, M, startTime + iteration_number * dt), fontsize=24) 
	plt.xlabel(r'grain area (pixel$^2$)', fontsize=18)
	plt.ylabel('probability density function', fontsize=18)




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
	# plt.plot(x, pCurrent, c='blue', linestyle='-') # debug
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
	pInit = q_pInit(x); pInit /= np.trapz(pInit, x);
	pCalibrated = q_pCalibrated(x); pCalibrated /= np.trapz(pCalibrated, x);
	pFinal = q_pFinal(x); pFinal /= np.trapz(pFinal, x);
	pCurrent = q_pCurrent(x); pCurrent /= np.trapz(pCurrent, x);
	pSpkCurrent = q_pSpkCurrent(x); pSpkCurrent /= np.trapz(pSpkCurrent, x);

	# initPlt, 			= plt.plot(x, pInit, 'ro:', linewidth=2, label='first training pdf (kMC: 46.5 mcs)')
	# calibratedPlt, 	= plt.plot(x, pCalibrated, 'mo-.', linewidth=2, label='last training pdf (kMC: 599.5 mcs)')
	# finalPlt, 		= plt.plot(x, pFinal, 'go--', linewidth=2, label='last testing pdf (kMC: 16,681.1 mcs)')
	currentPlt, 	= plt.plot(x, pCurrent, c='red', linestyle='--', linewidth=2, label='evolving Fokker-Planck pdf')
	spkCurrentPlt,  = plt.plot(x, pSpkCurrent, c='blue', linestyle='-', linewidth=2, label='SPPARKS pdf')

	plt.legend(handles=[currentPlt, spkCurrentPlt], fontsize=24, markerscale=2, loc='best', frameon=False) # bbox
	plt.tick_params(axis='both', which='major', labelsize=24)
	plt.tick_params(axis='both', which='minor', labelsize=24)

	# plt.xlim([np.min(x), np.max(x)])
	# plt.ylim([0, 1.5 * np.max(pCurrent)])
	# plt.title("Extrapolation in time comparison between Fokker-Planck and SPPARKS" + "\n" + r"1d Fokker-Planck iter: %d/%d (SPPARKS $t$=%.4f)" % (iteration_number, M, startTime + iteration_number * dt), fontsize=24) # change this
	# plt.title("Extrapolation in time comparison between Fokker-Planck and SPPARKS: $t=%.4f$" % (np.exp(startTime + iteration_number * dt)), fontsize=24) # exponential in time for kinetic Monte Carlo -- time calculated using FPE
	plt.title("Extrapolation in time comparison between Fokker-Planck and SPPARKS: $t=%.4f$" % (spkTimeList[spkIndex]), fontsize=24) # time referenced using SPPARKS
	plt.xlabel(r'grain area (pixel$^2$)', fontsize=18)
	plt.ylabel('probability density function', fontsize=18)

	### add sample images from SPPARKS simulations
	# https://towardsdatascience.com/how-to-add-an-image-to-a-matplotlib-plot-in-python-76098becaf53
	# https://matplotlib.org/stable/gallery/text_labels_and_annotations/demo_annotation_box.html#sphx-glr-gallery-text-labels-and-annotations-demo-annotation-box-py

	ax = plt.gca()

	import matplotlib.image as image
	img1_file   = "./potts_pfm_large2d/1/grain.%d.jpg" % spkIndex
	img2_file   = "./potts_pfm_large2d/2/grain.%d.jpg" % spkIndex
	img3_file   = "./potts_pfm_large2d/3/grain.%d.jpg" % spkIndex
	img4_file   = "./potts_pfm_large2d/4/grain.%d.jpg" % spkIndex
	img5_file   = "./potts_pfm_large2d/5/grain.%d.jpg" % spkIndex
	img6_file   = "./potts_pfm_large2d/6/grain.%d.jpg" % spkIndex
	img7_file   = "./potts_pfm_large2d/7/grain.%d.jpg" % spkIndex
	img8_file   = "./potts_pfm_large2d/8/grain.%d.jpg" % spkIndex
	img9_file   = "./potts_pfm_large2d/9/grain.%d.jpg" % spkIndex
	img10_file  = "./potts_pfm_large2d/10/grain.%d.jpg" % spkIndex
	img11_file  = "./potts_pfm_large2d/11/grain.%d.jpg" % spkIndex
	img12_file  = "./potts_pfm_large2d/12/grain.%d.jpg" % spkIndex
	img13_file  = "./potts_pfm_large2d/13/grain.%d.jpg" % spkIndex
	img14_file  = "./potts_pfm_large2d/14/grain.%d.jpg" % spkIndex
	img15_file  = "./potts_pfm_large2d/15/grain.%d.jpg" % spkIndex
	img16_file  = "./potts_pfm_large2d/16/grain.%d.jpg" % spkIndex
	img17_file  = "./potts_pfm_large2d/17/grain.%d.jpg" % spkIndex
	img18_file  = "./potts_pfm_large2d/18/grain.%d.jpg" % spkIndex
	img19_file  = "./potts_pfm_large2d/19/grain.%d.jpg" % spkIndex
	img20_file  = "./potts_pfm_large2d/20/grain.%d.jpg" % spkIndex
	img1    = image.imread(img1_file)
	img2    = image.imread(img2_file)
	img3    = image.imread(img3_file)
	img4    = image.imread(img4_file)
	img5    = image.imread(img5_file)
	img6    = image.imread(img6_file)
	img7    = image.imread(img7_file)
	img8    = image.imread(img8_file)
	img9    = image.imread(img9_file)
	img10   = image.imread(img10_file)
	img11   = image.imread(img11_file)
	img12   = image.imread(img12_file)
	img13   = image.imread(img13_file)
	img14   = image.imread(img14_file)
	img15   = image.imread(img15_file)
	img16   = image.imread(img16_file)
	img17   = image.imread(img17_file)
	img18   = image.imread(img18_file)
	img19   = image.imread(img19_file)
	img20   = image.imread(img20_file)
	# crop black padding
	img1 	= img1[0+3:512-3,0+3:512-3]
	img2	= img2[0+3:512-3,0+3:512-3]
	img3	= img3[0+3:512-3,0+3:512-3]
	img4	= img4[0+3:512-3,0+3:512-3]
	img5	= img5[0+3:512-3,0+3:512-3]
	img6	= img6[0+3:512-3,0+3:512-3]
	img7	= img7[0+3:512-3,0+3:512-3]
	img8	= img8[0+3:512-3,0+3:512-3]
	img9	= img9[0+3:512-3,0+3:512-3]
	img10	= img10[0+3:512-3,0+3:512-3]
	img11 	= img11[0+3:512-3,0+3:512-3]
	img12	= img12[0+3:512-3,0+3:512-3]
	img13	= img13[0+3:512-3,0+3:512-3]
	img14	= img14[0+3:512-3,0+3:512-3]
	img15	= img15[0+3:512-3,0+3:512-3]
	img16	= img16[0+3:512-3,0+3:512-3]
	img17	= img17[0+3:512-3,0+3:512-3]
	img18	= img18[0+3:512-3,0+3:512-3]
	img19	= img19[0+3:512-3,0+3:512-3]
	img20	= img20[0+3:512-3,0+3:512-3]

	from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

	ymin, ymax = plt.gca().get_ylim()

	imgbox1 = OffsetImage(img1, zoom=0.18)
	ab = AnnotationBbox(imgbox1, (50000, 0.775 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox2 = OffsetImage(img2, zoom=0.18)
	ab = AnnotationBbox(imgbox2, (45000, 0.775 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox3 = OffsetImage(img3, zoom=0.18)
	ab = AnnotationBbox(imgbox3, (40000, 0.775 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox4 = OffsetImage(img4, zoom=0.18)
	ab = AnnotationBbox(imgbox4, (35000, 0.775 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox5 = OffsetImage(img5, zoom=0.18)
	ab = AnnotationBbox(imgbox5, (50000, 0.605 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox6 = OffsetImage(img6, zoom=0.18)
	ab = AnnotationBbox(imgbox6, (45000, 0.605 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox7 = OffsetImage(img7, zoom=0.18)
	ab = AnnotationBbox(imgbox7, (40000, 0.605 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox8 = OffsetImage(img8, zoom=0.18)
	ab = AnnotationBbox(imgbox8, (35000, 0.605 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox9 = OffsetImage(img9, zoom=0.18)
	ab = AnnotationBbox(imgbox9, (50000, 0.435 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox10 = OffsetImage(img10, zoom=0.18)
	ab = AnnotationBbox(imgbox10, (45000, 0.435 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox11 = OffsetImage(img11, zoom=0.18)
	ab = AnnotationBbox(imgbox11, (40000, 0.435 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox12 = OffsetImage(img12, zoom=0.18)
	ab = AnnotationBbox(imgbox12, (35000, 0.435 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox13 = OffsetImage(img13, zoom=0.18)
	ab = AnnotationBbox(imgbox13, (50000, 0.265 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox14 = OffsetImage(img14, zoom=0.18)
	ab = AnnotationBbox(imgbox14, (45000, 0.265 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox15 = OffsetImage(img15, zoom=0.18)
	ab = AnnotationBbox(imgbox15, (40000, 0.265 * ymax), frameon=False)
	ax.add_artist(ab)

	imgbox16 = OffsetImage(img16, zoom=0.18)
	ab = AnnotationBbox(imgbox16, (35000, 0.265 * ymax), frameon=False)
	ax.add_artist(ab)


	### save fig


	plt.savefig('ggComparison_FokkerPlanck_iter%d.png' % ii, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

	elapsedTimeInSeconds = datetime.datetime.now() - startTimePython
	plt.close()

	print('Done iteration %d. Elapsed time: %.2f seconds' % (ii, elapsedTimeInSeconds.total_seconds()))
	# plt.show()




