#!/bin/bash

for i in $(seq 0 100000); do
	ln -sf $(pwd)/template/sbatch.spparks.solo  $i
	echo "done $i"
done
