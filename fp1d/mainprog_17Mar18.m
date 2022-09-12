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
xLB = [+0.0e-5, -1.0e-1, -0.5e-4, -0.5e-1];
xUB = [+1.0e-5, +1.0e-1, +0.5e-4, +0.5e-1];

% xLB = [+0.0e-3, -1.0e+1, -8.0e-3, -0.0e-1];
% xUB = [+1.0e-3, +1.0e+1, +8.0e-3, +5.0e-1];

% xLB = [+0.0e-3, -1.0e+1, -8.0e-3, -5.0e-1];
% xUB = [+1.0e-3, +1.0e+1, +8.0e-3, +5.0e-1];

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
numInitPoint = 2;							% number of initial sampling points

% hyper-parameter settings (do not change)
theta = 1e-1 * ones(1,d); 					% init guess 
lob = 1e-2 * ones(1,d); 
upb = 1e+0 * ones(1,d); 

% optimization settings
maxIter = 10000; 							% maximum number of iteration
checkTime = 2; 								% check time in minutes
waitTime = 1; 								% waitTime in hours, taskkill /im comsolmphserver.exe /f if exceeded 

% batch settings
exploitSize = 7;
exploreSize = 3;

% crash and patch settings
numSeqPoint = 842;							% (numSeqPoint + 1) is the start of next iteration
skipInitialSampling = 0;					% yes: skipInitialSampling process

% ------------------------------------------------------------ read available pdf ------------------------------------------------------------

% copy from ForwardFokkerPlanckModel(); could be inconsistent
var = 'areaOfGrain';					% name of variable
N = 5.0e3; 								% number of segments
startSupp = 2.0e3; endSupp = 9.0e4;	 	% support
sourcePath = pwd;
h = (endSupp - startSupp)/N;
x = [startSupp:h:endSupp]; 				% discretized segment 

% import time-series pdf from post-processing SPPARKS
fprintf('Importing time-series pdf... \n');
pCheckAndMatch = zeros(41, length(x));
for i = 0:39
	pCheckAndMatch(i+1, :) = ksdensity(csvread(strcat(sourcePath,'/probabilityRepo/','log.',var,'.', num2str(i), '.dat')), x);
	fprintf('Imported + Smoothed time-series pdf: %d/40... \n', i);
end
fprintf('Finished Importing time-series pdf... \n');


% ------------------------------------------------------------ sequential initial sampling ------------------------------------------------------------

if ~skipInitialSampling
	S = []; Y = [];
	currentFolder = sprintf('initSampl');
	system(sprintf('mkdir %s', currentFolder));
	cd(currentFolder);

	for i = 1:numInitPoint
		x = xLB + rand(1,d) .* (xUB - xLB);
		y = - ForwardFokkerPlanckModel(x, pCheckAndMatch); S = [S; x]; Y = [Y; y]; % run simulation
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

for i = (startIndex):maxIter

	currentFolder = sprintf('ForwardFokkerPlanckModel_Iter%d', i)
	system(sprintf('mkdir %s', currentFolder));
	cd(currentFolder);

	x = getNextSamplingPoint(dmodel, xLB, xUB); % get new sampling points via acquisition function -- write outputs to file
	y = - ForwardFokkerPlanckModel(x, pCheckAndMatch);

	% log
	dlmwrite(sprintf('log.iter.%d.dat', i), [reshape(x,length(x),1);y], 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('log.input.dat', [x], 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('log.output.dat', [y], 'delimiter', ',', 'precision', '%0.16e');
	
	
	S = [S; reshape(x,1,length(x))]; Y = [Y; y];
	system('rm -v outcmaes*');
	dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16e');
	fprintf('done iteration %d\n', i);

	[dmodel, perf] = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb); % update mode
	cd(parentPath);
	dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16e');
	dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16e');
end
