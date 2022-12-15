# objectives
# (1) quantify the chord-length distribution from a ms image

import numpy as np
import skimage
from scipy.stats import gaussian_kde # as gaussian_kde # import kde 
import matplotlib.pyplot as plt

# 1 = white, 0 = black

msGray = skimage.io.imread('topBnw.100.jpg', as_gray=True)
# msBnw = (msGray == 1.00).astype(float) # either white, or not-white
# msBnw = (msGray > 0.5).astype(float)
thresh = skimage.filters.threshold_otsu(msGray)
msBnw = msGray > thresh

H, L = msBnw.shape # (3220, 6300)

skimage.io.imsave('topStrictBnw.100.jpg', msBnw.astype(float))
# plt.imshow(msBnw, cmap='gray')
# plt.show()

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
		elif chordLength > 1: # only append if the chord-length > 1 
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
qplot = np.linspace(-1, 1.25 * np.max(chordLengthList), num=1000)

# plot 
plt.figure()
chordLengthPlot = plt.plot(qplot, chordLengthKde(qplot), 'b-', linewidth=4, label='chordLength pdf')
plt.xlabel(r'chord-length (pixels)', fontsize=24)
plt.ylabel(r'pdf', fontsize=24)
plt.axes().yaxis.get_major_formatter().set_powerlimits((0, 1))
plt.tick_params(axis='both', which='major', labelsize=24)
plt.tick_params(axis='both', which='minor', labelsize=24)
plt.title('chord-length pdf along x-direction', fontsize=36)
plt.show()
