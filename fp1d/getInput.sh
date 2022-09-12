outFile=log.input.dat
rm -v $outFile

# for i in $(seq 3 1648); do
# 	cat ForwardFokkerPlanckModel_Iter${i}/${outFile} >> $outFile
# 	echo "done $i"
# done

for folderName in $(ls -1dv ForwardFokkerPlanckModel_Iter*/); do
	cat $folderName/${outFile} >> ${outFile}
	echo "done $folderName"
done


