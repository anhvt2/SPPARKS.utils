#!/bin/bash

for View in top front left perspective; do
	cd ovito_${View}
	convert -resize 100% -delay 100 -loop 0 ${View}_small3d_00{01..20}.png ${View}_small3d.gif
	mv ${View}_small3d.gif ..
	cd ..
	echo "done ovito_${View}"
done

