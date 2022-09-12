#!/bin/bash

pawd='Kenshin@1986'
sshpass -p ${pawd} scp -r -v  "src_8Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "compressPapersList.txt" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "DEs in MATLAB.pdf" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "src_9Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "checkComplete.py" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "dace" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "bayesSrc" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog_9Mar18.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog_11Mar18.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "src_11Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog_12Mar18.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "S.dat" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "Y.dat" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "src_12Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "README" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog_13Mar18.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "visualResStepPlot.py" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "getOptimalInput.py" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "transposeInput.py" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "transfer.sh" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "src_13Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "updateSrc.sh" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog_16Mar18.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "mainprog.m" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "src_16Mar18.tar.gz" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "log.input.dat" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "log.output.dat" $pacedata/MATLAB/fp1d
sshpass -p ${pawd} scp -r -v  "model" $pacedata/MATLAB/fp1d

