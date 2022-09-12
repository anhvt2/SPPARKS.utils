
import numpy as np
from scipy.stats import moment

muAr  = []
vrnAr = []
m3Ar  = []
m4Ar  = []

for i in range(40):
	d = np.loadtxt('log.logAreaOfGrain.%d.dat' % i)
	muAr  += [np.mean(d)]
	vrnAr += [moment(d, moment=2)]
	m3Ar  += [moment(d, moment=3)]
	m4Ar  += [moment(d, moment=4)]


np.savetxt('log.logmuAr.dat', muAr, fmt='%.18e')
np.savetxt('log.logvrnAr.dat', vrnAr, fmt='%.18e')
np.savetxt('log.logm3Ar.dat', m3Ar, fmt='%.18e')
np.savetxt('log.logm4Ar.dat', m4Ar, fmt='%.18e')


