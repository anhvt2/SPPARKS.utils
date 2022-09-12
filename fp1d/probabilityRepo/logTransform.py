
import numpy as np
for i in range(40):
	d = np.loadtxt('log.areaOfGrain.%d.dat' % i)
	log_d = np.log(d)
	np.savetxt('log.logAreaOfGrain.%d.dat' % i, log_d, delimiter='\n', fmt='%.18e')

