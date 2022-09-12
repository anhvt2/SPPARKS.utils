#!/bin/bash

versionDate='28Mar18'
# rm -v *.tar.gz

echo "versionDate: $versionDate"
echo "Commands activate:"

# run through sub-folderName

for folderName in bayesSrc dace model; do
	cd $folderName

	for fileName in $(ls *_${versionDate}*); do
		filePrefix=$(echo $fileName | cut -d_ -f1)
		fileAppendix=$(echo $fileName | cut -d. -f2)
		echo "cp -v  ${folderName}/${fileName}  ${folderName}/${filePrefix}.${fileAppendix}"
		cp -v ${fileName} ${filePrefix}.${fileAppendix}
	done

	cd ..
done

for fileName in $(ls *_${versionDate}*); do
	filePrefix=$(echo $fileName | cut -d_ -f1)
	fileAppendix=$(echo $fileName | cut -d. -f2)
	echo "cp -v ${fileName} ${filePrefix}.${fileAppendix}"
	cp -v ${fileName} ${filePrefix}.${fileAppendix}
done

echo; echo;

tar -cvzf src_${versionDate}.tar.gz *m *py *txt *dat *sh bayesSrc/ dace/ model/
rm src.tar

echo; echo;

echo "Files updated:"
ls *_${versionDate}*

echo; echo;
echo "note: tar into src_${versionDate}.tar.gz"
echo; echo;
