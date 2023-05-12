#!/bin/bash

nRVE=10
for seed in $(seq ${nRVE}); do
	for T in $(seq 0.0 0.1 0.5); do
		for pin in $(seq 0.0 0.03 0.15); do

			folderName="potts-pin-T-${T}-pin-${pin}-seed-${seed}"
			rm -rfv $folderName
			cp -rfv template/ $folderName
			cd $folderName
			echo "seed         ${seed}" > in.seed
			echo "pin             ${pin} 0 0   " > in.pin
			echo "temperature  ${T}" > in.temperature
			cd ..
		done
	done
done
