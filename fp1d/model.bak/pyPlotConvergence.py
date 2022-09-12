import numpy as np
import matplotlib.pyplot as plt

# plot the convergence plot of KL distance and L2-norm between the testDensity and the propagated density
dat = np.loadtxt('log.KLdistance_numerics.txt',delimiter=',')
endStep = 439945
step = np.array(dat[:,0], dtype=int)
l2norm = np.array(dat[:,1], dtype=float)
kldistance = np.array(dat[:,2], dtype=float)

fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=30)
ax.tick_params(axis='both', which='minor', labelsize=30)

ax.set_yscale('log')
ax.plot(step/1e3, l2norm, 		'b--', linewidth = 4, label=r'$L_2$-norm')
ax.plot(step/1e3, kldistance, 	'r-.', linewidth = 4, label='KL-distance')
# ax.plot(a, c + d, 'k', label='Total message length')
legend = ax.legend(shadow=True, fontsize=30)
ax.set_xlabel(r'Monte-Carlo step ($10^3$ mcs)', fontsize=28)
ax.set_ylabel('Measure value', fontsize=28)
ax.set_title('Difference between propagated density and test density', fontsize=28)

plt.show()



