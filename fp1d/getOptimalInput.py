import numpy as np
S = np.loadtxt('log.input.dat', delimiter=',')
Y = np.loadtxt('log.output.dat')

# S = np.loadtxt('S.dat', delimiter=',')
# Y = np.loadtxt('Y.dat')

print 'best input (for minimal L2-norm) is'
print(S[np.argmax(Y)])
print 'objective (L2-norm)'
print(- np.max(Y))
