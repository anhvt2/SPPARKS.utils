#!/bin/bash

for i in $(seq 0 50); do
	cp -rfv template $i
	ln -sf template/sbatch.spparks.solo  $i
done
