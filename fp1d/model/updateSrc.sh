#!/bin/bash

versionDate='12Apr18'
# rm -v *.tar.gz

echo "Commands activate:"

# run through sub-folderName

for fileName in $(ls *_${versionDate}*); do
	filePrefix=$(echo $fileName | cut -d_ -f1)
	fileAppendix=$(echo $fileName | cut -d. -f2)
	echo "cp -v ${fileName} ${filePrefix}.${fileAppendix}"
	cp -v ${fileName} ${filePrefix}.${fileAppendix}
done

echo; echo;

tar -cvzf src_${versionDate}.tar.gz *m *sh
rm src.tar

echo; echo;

echo "Files updated:"
ls *_${versionDate}*

rm -v src.tar
echo; echo;
echo "note: tar into src_${versionDate}.tar.gz"
echo; echo;
