# python postproc script for log.postproc.txt
# clone from $doc/giw/convPlot.py or test.py (deprecated)
# adopt from /home/anhvt89/Documents/cellular/numExample/results/visualResContinuous_5Oct17.py

# use 6Jan18 for paper results

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24 

# -------- define input -------- #

# Y = - np.loadtxt('Y.dat')
Y = - np.loadtxt('log.output.dat')

# Y /= 1e15
trncLen = len(Y)
# trncLen = 170

# with initial-sampling
# feasible = feasible[:(trncLen + 1) ]
# Y = np.array(Y[:(trncLen + 1)] ) 

Y = Y[:trncLen]

minList = [0]
minY = Y[0]

for i in range(len(Y)):
	if Y[i] < minY:
		minList.append(i)
		minY = Y[i]

minListPlot = [0]
for i in range(1,len(minList) - 1):
	minListPlot.append(minList[i])

minListPlot.append(minList[-1])
minListPlot.append(len(Y))
Yplot = Y[minListPlot[:len(minListPlot) - 1]]
Yplot = list(Yplot)
Yplot.append(Y[minList[-1]])


# plot
iterList = np.array(range(len(Y)))
feasPlot, = plt.plot(iterList, Y, 'bo', ms=8, mew=5)
# plt.legend(handles=[feasPlot], fontsize=26, markerscale=2, loc=1)# bbox_to_anchor=(0.35, 0.20))
# plt.legend()
plt.step(minListPlot, Yplot, where='post', linewidth=3, color='r', linestyle='-')
# plt.errorbar(range(len(Y)), Y, yerr=mse, fmt='o',ecolor='g')
plt.xlabel('Iterations',fontsize=26)
# plt.ylabel(r'Average wear rate ($\mu m/hr$)',fontsize=26)
plt.ylabel(r'$L_2$-norm',fontsize=26)
plt.title('Fokker-Planck model calibration',fontsize=26)
# plt.ylim(-0.05, 1.00)
# plt.ylim(0.60, 0.80)
plt.show()
