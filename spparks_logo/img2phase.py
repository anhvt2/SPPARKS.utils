
import skimage.io as io
import numpy as np
d = io.imread('cropped.spparks.png', as_gray=True) # d.shape = (192, 1039)

for i in range(d.shape[0]):
	for j in range(d.shape[1]):
		if d[i,j] < 1:
			d[i,j] = 0

io.imshow(d)
plt.show()
