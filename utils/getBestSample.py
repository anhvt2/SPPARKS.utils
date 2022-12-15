
import numpy as np
i = np.loadtxt('postproc.input.dat')
o = np.loadtxt('postproc.output.dat')

iMax = np.argmax(o)

print('Best output (max) = %.8f' % o[iMax])
print('index iMax = %d' % iMax)
print('Corresponding input = %.8f' % i[iMax])


