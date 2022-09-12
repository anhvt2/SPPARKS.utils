import numpy as np 
d = 3 # dimensionality of the problem

a = np.loadtxt('log.input.dat')
b = np.loadtxt('log.output.dat')
np.savetxt('log.input.dat', a.reshape([len(b), d]), delimiter=',', fmt='%.16e', newline='\n')



