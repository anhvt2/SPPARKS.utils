
"""
This file 
	(1) generates 
		(a) seed and 
		(b) input parameters
	(2) and write a "includable" spparks input script
"""

import numpy as np

seed = np.random.randint(low=1, high=100000, dtype=int)
params = np.random.uniform(low=0.0, high=1.0, size=None)

seedFile = open("in.seed.spk", "w")
seedFile.write("seed %d\n" % seed)
seedFile.close()

# inputFile = open("in.params.spk", "w")
# inputFile.write("temperature  %.2f\n" % params)
# inputFile.close()
