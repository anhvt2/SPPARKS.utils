# objectives
# (1) quantify the chord-length distribution from a ms image

import numpy as np
# import skimage.io as io
import skimage
from scipy.stats import gaussian_kde # as gaussian_kde # import kde 

# 1 = white, 0 = black

msGray = skimage.io.imread('topBnw.100.jpg', as_gray=True)
msBnw = (msGray == 1.00).astype(float) # either white, or not-white
H, L = msBnw.shape # (3220, 6300)

## filter single black pixels that stand alone
for i in  range(1,H-1):
	for j in range(1,L-1):
		# if all the surrounding is white and there is a single black pixel, then color it white
		if (msBnw[i,j] == 0) and (msBnw[i+1,j] == msBnw[i-1,j] == msBnw[i,j+1] == msBnw[i,j-1] == 1):
			msBnw[i,j] == 1
	# print('done horizontal slice %d' % i)

skimage.io.imsave('topStrictBnw.100.jpg', msBnw)

### user-defined functions
# msSlice = a slice of ms in y-direction
def getChordLengthList4msSlice(msSlice):
	# output: chordLengthList4msSlice = list of chord-lengths for a slice of ms
	# objective: given a slice, then get a list of chord-lengths
	pixelList = np.where(msSlice == 1)[0] # index of pixels inside a grain
	chordLength = 0 # init
	chordLengthList4msSlice = []
	# counting chord-length
	for iMs in range(len(pixelList) - 1):
		if pixelList[iMs + 1] == pixelList[iMs] + 1: # if the index increases only 1
			chordLength += 1 # incre
		elif chordLength > 0: # only append if the chord-length > 0 
			chordLengthList4msSlice.append(chordLength)
			chordLength = 0 # reset
		else:
			chordLength = 0
	return chordLengthList4msSlice

## get chordLengthList
chordLengthList = [] # init

# # # range(L): slice along y-direction
# for j in range(L):
# 	msSlice = msBnw[:, j]
# 	chordLengthList4msSlice = getChordLengthList4msSlice(msSlice)
# 	chordLengthList += chordLengthList4msSlice

# # range(H): slice along x-direction
stripWidth = 40 # define from the center H/2
for i in range(int(H/2 - stripWidth), int(H/2 + stripWidth + 1)):
	msSlice = msBnw[i, :]
	chordLengthList4msSlice = getChordLengthList4msSlice(msSlice)
	chordLengthList += chordLengthList4msSlice

## form pdf using gaussian_kde
chordLengthKde = gaussian_kde(chordLengthList)

# plot 
import matplotlib.pyplot as plt
qplot = np.linspace(-1, 1.25 * np.max(chordLengthList), num=1000)
plt.figure()
chordLengthPlot = plt.plot(qplot, chordLengthKde(qplot), 'b-', linewidth=4, label='chordLength pdf')
plt.title('chord-length pdf along x-direction', fontsize=36)
# plt.xlim([-1, ])
plt.show()
