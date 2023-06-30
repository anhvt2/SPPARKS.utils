#!/bin/bash

for i in $(seq 1000); do
	strI=$(printf %03d $i)
	folderName="seed-${strI}"
	cp -rfv template/ ${folderName}
	cd ${folderName}/
    sed -i "3s|.*|seed         ${strI}|" in.potts_3d
    ssubmit
    cd ..
done 
