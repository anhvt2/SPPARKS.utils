for folderName in $(ls -1dv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/*/); do
	folderName=$(echo $folderName | rev | cut -d/ -f2 | rev )
	

	mkdir $folderName
	cd $folderName

	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/80000/  . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/90000/  . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/100000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/110000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/120000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/130000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/140000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/150000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/160000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/170000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/180000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/190000/ . 
	cp -rfv /media/anhvt89/ExtHD3TB/dataAcquisitionEpistemic/decomposeData/outputs/$folderName/200000/ . 


	cd ..
	echo "done $folderName"

done

