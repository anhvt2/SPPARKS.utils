close all;
clear all;
home;
format longg;

% ------------------------------------------------------------ input parameters ------------------------------------------------------------
% path settings
% parentPath = 'C:\Users\Justin\Desktop\Research\Optimization\src_16Jan18_parallel';
parentPath = pwd;
cd(parentPath); % change to parent path

addpath(parentPath); 						% add current path
addpath(strcat(parentPath,'/dace')); 		% add GPR toolbox
addpath(strcat(parentPath,'/bayesSrc')); 	% add BO toolbox
addpath(strcat(parentPath,'/model')); 		% add model program

% variable lower/upper bounds 
% inputFokkerPlanck = [5.13500783e-3 -1.27475697e-3 1.22675403e-06];
% inputFokkerPlanck = [5.13500783e-3 -1.27475697e-3 1.22675403e-06 2.38070822e-08];
% inputFokkerPlanck = [5.00000000e-3, -7.98911869e-04,  1.50000000e-06,  3.50000000e-08];

% 4-param
% top -> bottom: lowest -> highest bench of bounds
% xLB = [+0.0e-5, -1.0e-1, -0.5e-4, -0.5e-1];
% xUB = [+1.0e-5, +1.0e-1, +0.5e-4, +0.5e-1];

% xLB = [+0.0e-3, -1.0e+1, -8.0e-3, -0.0e-1];
% xUB = [+1.0e-3, +1.0e+1, +8.0e-3, +5.0e-1];

% xLB = [+0.0e-3, -1.0e+1, -8.0e-3, -5.0e-1];
% xUB = [+1.0e-3, +1.0e+1, +8.0e-3, +5.0e-1];

xLB = [  1.0e-7   1e-1   1.0e+0];
xUB = [  5.0e-5   1e+1   5.0e+3];

% 5-param (17Mar18 -- not tested yet)
% xLB = [+0e-12, +0.0e-4, -1.0e-1, -0.5e-3, -0.5e+2];
% xUB = [+1e-12, +1.0e-4, +1.0e-1, +0.5e-3, +0.5e+2];

% 6-param
% xLB = [0.0e-2, 2.0e-3, -1.5e-3, -1.5e-3, 0.0e-5];
% xUB = [1.0e-2, 8.0e-3, -0.5e-3, +0.5e-3, 1.0e-5];

% 2-param
% xLB = [0.0e-2, 0.0e-5];
% xUB = [1.0e-2, 1.0e-5];

% input GP parameters settings
d = length(xLB);							% dimensionality of the problem
numInitPoint = 4;							% number of initial sampling points

% hyper-parameter settings (do not change)
theta = 1e-1 * ones(1,d); 					% init guess 
lob = 1e-2 * ones(1,d); 
upb = 1e+0 * ones(1,d); 

% optimization settings
maxIter = 1e3; 								% maximum number of iteration
checkTime = 2; 								% check time in minutes
waitTime = 1; 								% waitTime in hours, taskkill /im comsolmphserver.exe /f if exceeded 

% batch settings
exploitSize = 7;
exploreSize = 3;

% crash and patch settings
numSeqPoint = 2;							% (numSeqPoint + 1) is the start of next iteration
skipInitialSampling = 0;					% yes: skipInitialSampling process

% ------------------------------------------------------------ read available pdf ------------------------------------------------------------

% copy from ForwardFokkerPlanckModel(); could be inconsistent
var = 'areaOfGrain';					% name of variable
N = 6.0e3; 								% number of segments
startSupp = -3.0e4; endSupp = 9.0e4;	% support
sourcePath = pwd;
h = (endSupp - startSupp)/N;
x = [startSupp:h:endSupp]; 				% discretized segment 

% import time-series pdf from post-processing SPPARKS
fprintf('Importing time-series pdf... \n');
pHistorySpparks = zeros(40, length(x));
for i = 0:39
	pHistorySpparks(i+1, :) = ksdensity(csvread(strcat(sourcePath,'/probabilityRepo/','log.',var,'.', num2str(i), '.dat')), x);
	fprintf('Imported + Smoothed time-series pdf: %d/40... \n', i);
end
fprintf('Finished Importing time-series pdf... \n');


% ------------------------------------------------------------ sequential initial sampling ------------------------------------------------------------

if ~skipInitialSampling
	system('rm -rfv *dat initSampl');
	S = []; Y = [];
	currentFolder = sprintf('initSampl');
	system(sprintf('mkdir %s', currentFolder));
	cd(currentFolder);

	for i = 1:numInitPoint
		x = xLB + rand(1,d) .* (xUB - xLB);
		y = - ForwardFokkerPlanckModel(x, pHistorySpparks); S = [S; x]; Y = [Y; y]; % run simulation
	end

	cd(parentPath);

	dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16e'); 
	dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('Sinit.dat', S, 'delimiter', ',', 'precision', '%0.16e'); 
	dlmwrite('Yinit.dat', Y, 'delimiter', ',', 'precision', '%0.16e');
end

% fit model
S = dlmread('S.dat');
Y = dlmread('Y.dat');
dmodel = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb); 
fprintf('initial fit done!\n\n');

% % ------------------------------------------------------------ parallel optimization loop ------------------------------------------------------------

% 14Mar18: update two loops for convenience in handling skipInitialSampling
if ~skipInitialSampling
	startIndex = numInitPoint + 1;
else
	startIndex = numSeqPoint + 1;
end

for i = (startIndex):int32(maxIter)

	S = dlmread('S.dat'); Y = dlmread('Y.dat');
	startTime = clock();
	clear halluDmodel; halluDmodel = dmodel; 

	% create input
	for j = 1:exploitSize
		currentFolder = sprintf('ForwardFokkerPlanckModel_Iter%d_Exploit_Batch%d', i, j);
		system(sprintf('mkdir %s', currentFolder));
		cd(currentFolder);
		x = getNextSamplingPoint(halluDmodel, xLB, xUB); % get new sampling points via acquisition function -- write outputs to file
		dlmwrite('log.input.dat', x, 'delimiter' , ',' , 'precision' , '%0.16e');
		[mu, ~, sigma, ~] = predictor(x, halluDmodel);
		clear S Y;
		S = halluDmodel.origS; Y = halluDmodel.origY;
		S = [S; reshape(x,1,length(x))]; Y = [Y; mu]; 
		halluDmodel = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb); % update halluDmodel
		system('cp -v ../model/*m .'); system('cp -v ../model/qsub.runOneSim.pace . ; $submit');
		cd(parentPath);
	end

	for j = 1:exploreSize
		currentFolder = sprintf('ForwardFokkerPlanckModel_Iter%d_Explore_Batch%d', i, j);
		system(sprintf('mkdir %s', currentFolder));
		cd(currentFolder);
		x = getNextSamplingPoint(halluDmodel, xLB, xUB); % get new sampling points via acquisition function -- write outputs to file
		dlmwrite('log.input.dat', x, 'delimiter' , ',' , 'precision' , '%0.16e');
		[mu, ~, sigma, ~] = predictor(x, halluDmodel);
		clear S Y;
		S = halluDmodel.origS; Y = halluDmodel.origY;
		S = [S; reshape(x,1,length(x))]; Y = [Y; mu]; 
		halluDmodel = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb); % update halluDmodel
		system('cp -v ../model/*m .'); system('cp -v ../model/qsub.runOneSim.pace . ; $submit');
		cd(parentPath);
	end

	cd(parentPath); 
	iterFile = fopen('log.iteration.txt','w+'); fprintf(iterFile, '%d\n%d\n%d\n', i, exploitSize, exploreSize); fclose(iterFile);

	% wait and check results periodically
	system('python checkComplete.py'); clear sim_status; sim_status = dlmread('log.sim_status.txt');
	while ~sim_status; pause(waitTime * 60); system('python checkComplete.py');	sim_status = dlmread('log.sim_status.txt'); fprintf('Waiting...\n\n\n\n\n'); end;
	system('rm -v log.iteration.txt'); 


	% collect results
	clear S Y halluDmodel; 
	cd(parentPath);
	S = dlmread('S.dat'); Y = dlmread('Y.dat'); 

	for j = 1:exploitSize
		currentFolder = sprintf('ForwardFokkerPlanckModel_Iter%d_Exploit_Batch%d', i, j);
		cd(currentFolder);
		x = dlmread('log.input.dat'); y = - dlmread('log.output.dat'); system('rm -v *.m *.mat *cmaes*');
		S = [S; reshape(x,1,length(x))];
		Y = [Y; y];
		system('rm -v *cmaes*');
		cd(parentPath); 
	end

	for j = 1:exploreSize 
		currentFolder = sprintf('ForwardFokkerPlanckModel_Iter%d_Explore_Batch%d', i, j);
		cd(currentFolder);
		x = dlmread('log.input.dat'); y = - dlmread('log.output.dat'); system('rm -v *.m *.mat *cmaes*');
		S = [S; reshape(x,1,length(x))];
		Y = [Y; y];
		system('rm -v *cmaes*');
		cd(parentPath); 
	end

	% write results to file; reconstruct dmodel
	system('rm -v *outcmaes*');
	dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16e');
	fprintf('done iteration %d\n', i);

	[dmodel, perf] = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb); % update mode
	cd(parentPath);
end
