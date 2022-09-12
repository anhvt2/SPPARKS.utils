import numpy as np
import matplotlib.pyplot as plt

t = np.loadtxt('log.time.dat')

mu = np.loadtxt('log.logmuAr.dat')[:len(t)]
sigma = np.loadtxt('log.logvrnAr.dat')[:len(t)]

t = np.log(t)

idx_train = range(15,26)
idx_test = range(26,len(t))

plt.figure()
# plt.plot(t, mu, 'ko', markersize=8)
plt.plot(t[idx_train], mu[idx_train], 'bo-', markersize=8, label='train')
# plt.errorbar(t[idx_train], mu[idx_train], np.sqrt(sigma[idx_train]), marker='o', mfc='red', mec='green', ms=10, mew=4)
plt.plot(t[idx_test] , mu[idx_test],  'rs:', markersize=8, label='test')
plt.legend(fontsize=18)
plt.xlabel('log of time', fontsize=18)
plt.ylabel('first moment of log of grain area', fontsize=18)
# plt.errorbar(t[idx_test], mu[idx_test], np.sqrt(sigma[idx_test]), marker='s', mfc='red', mec='green', ms=10, mew=4)

import scipy.stats as stats
slope, intercept, r, p, se = stats.linregress(t[idx_train], mu[idx_train])
x = np.linspace(np.min(t), np.max(t))
plt.plot(x, slope * x + intercept, c='magenta', linewidth=2, linestyle=':', label='linear regression')

print(t[idx_train])
print(t[idx_test])
print(slope)

####
plt.figure()
# plt.plot(t, sigma, 'ko', markersize=8)
plt.plot(t[idx_train], sigma[idx_train], 'bo-', markersize=8, label='train')
# plt.errorbar(t[idx_train], sigma[idx_train], np.sqrt(sigma[idx_train]), marker='o', mfc='red', mec='green', ms=10, mew=4)
plt.plot(t[idx_test] , sigma[idx_test],  'rs:', markersize=8, label='test')
# plt.errorbar(t[idx_test], sigma[idx_test], np.sqrt(sigma[idx_test]), marker='s', mfc='red', mec='green', ms=10, mew=4)

import scipy.stats as stats
slope, intercept, r, p, se = stats.linregress(t[idx_train], sigma[idx_train])
x = np.linspace(np.min(t), np.max(t))
plt.plot(x, slope * x + intercept, c='magenta', linewidth=2, linestyle=':', label='linear regression')
plt.legend(fontsize=18)
plt.xlabel('log of time', fontsize=18)
plt.ylabel('second moment of log of grain area', fontsize=18)

print(t[idx_train])
print(t[idx_test])
print(slope)

plt.show()


