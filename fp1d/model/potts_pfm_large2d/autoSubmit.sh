#!/bin/bash

for i in $(seq 50); do
	cd $i
	ln -sf ../template/sbatch.spparks.solo
	# ln -sf ../template/in.grain_growth.spk
	ln -sf ../template/in.potts_pfm.spk
	sdel
	ssubmit
	cd ..
done
