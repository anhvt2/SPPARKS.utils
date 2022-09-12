
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, glob
from scipy import stats

mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

x = np.loadtxt('x.dat')
pCurrentList = natsorted(glob.glob('pCurrent.*.dat'), alg=ns.IGNORECASE)
timeFPE_list = []

pCurrent_array = np.empty([len(x), len(pCurrentList)])
for i in range(len(pCurrentList)):
	pCurrent_array[:,i] = np.loadtxt(pCurrentList[i])
	timeFPE_list += [float(pCurrentList[i].split('.')[1])]
	print('done %d' % i)

timeFPE_list = np.array(timeFPE_list)
timeFPE_list *= dt
gaussian_kde_list = []
for fileName in glob.glob('../probabilityRepo/log.areaOfGrain.*.dat'):
	gaussian_kde_list += [stats.gaussian_kde(np.loadtxt(fileName))]

# migrate settings from "ForwardFokkerPlanckModel.m"
dt = 1.0e-2
xp = np.linspace(0, 20000, 500)
log_spparks = np.loadtxt('../log.spparks.partial')

### plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

## plot reference pdfs
t = log_spparks[1:, 1]
# t = np.loadtxt('../probabilityRepo/log.time.dat')
n = np.min([len(gaussian_kde_list), len(t)])
for i in range(n):
# for i in range(1,10):
	ax.plot(t[i] * np.ones(xp.shape), xp, gaussian_kde_list[i](xp) / np.trapz(gaussian_kde_list[i](xp), xp), c='b') # normalized

Neverystep = 100
# for i in range(0, len(timeFPE_list)):
for i in range(0, len(timeFPE_list), Neverystep):
	ax.plot(timeFPE_list[i] * np.ones(x.shape), x, pCurrent_array[:,i] / np.trapz(pCurrent_array[:,i], x), c='r') # normalized

ax.set_xlabel('time (mcs)', fontsize=15)
ax.set_ylabel(r'grain area ($pixel^2$)', fontsize=15)
ax.set_zlabel('Density function', fontsize=15)
ax.set_title('Microstructure evolution: kMC - grain area', fontsize=15)
# ax.zaxis.set_scale('log')
ax.axes.set_ylim3d(bottom=0, top=20000) 
plt.show()
