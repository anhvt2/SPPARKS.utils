clear all;
close all;
clc;
format longg

system('rm -v *png');

% inputFokkerPlanck = [3.89360873e-04  6.56689836e-03 -1.49999994e-03  1.35913222e-10 1.00062386e-06  2.10000000e-08]; % from getOptimalInput.py
% inputFokkerPlanck = [5.02085133e-03  4.49051540e-03 -6.48611998e-04  3.92629546e-04 3.02935332e-06];
% inputFokkerPlanck = [6.13452265e-03 -1.40450972e-03  4.90665253e-07  1.00000000e-09];
% inputFokkerPlanck = [5.02085133e-03  4.49051540e-03 -6.48611998e-04 1e-9];
% inputFokkerPlanck = [6.10100941e-03 -1.38820969e-03  5.14461403e-07  1.00000000e-09];
% inputFokkerPlanck = [  1.00000000e-04  -6.49796060e-01   0   1.0e+2];
% inputFokkerPlanck = [  1.00000000e-06  6.49796060e-01   1e-3   1.0e+3];
% inputFokkerPlanck = [  1.00000000e-06  6.49796060e-01   0   5.0e+3];
% inputFokkerPlanck = [  1.00000000e-06  6.49796060e-01   5.0e+3];
% inputFokkerPlanck = [5.00000000e-06 2.71407475e+00 4.85323894e+03];
% inputFokkerPlanck = [ 1.25351135e-05   9.05348321e+00   6.35821500e+02]; 
% inputFokkerPlanck = [5.00000000e-05   1.58439936e+00   4.85299329e+03]; % 2Apr18
% inputFokkerPlanck = [0.635406754759, 7674.39585341];
% inputFokkerPlanck = [5.13500783e-3 -1.27475697e-3 1.22675403e-06];
inputFokkerPlanck = [0.7319977216145724, -0.05862016551717416 / 2.]; % 22Feb22 
% inputFokkerPlanck = [0.7319977216145724, 0.05862016551717416];
% copy from ForwardFokkerPlanck.m 
% ----------------------------------- PARAMETERS SETTINGS ---------------------------- %
var = 'logAreaOfGrain';					% name of variable
N = 7.5e2; 								% number of segments
startSupp = 0.0e0; endSupp = 1.5e1;	% support
% safeWidth = 0.1e4;						% width of zero density
% dt = 1.0e+0;							% time-step
% Nfrequency = 10/100 ;					% percentage - show progress at every Nfrequency
% magFactor = 1e2; 						% magnifying factor to penalize the objective function
% startTime/endTime of ForwardFokkerPlanckModel
% startTime 	= 1.375e0;					% startTime -- [time unit]
% endTime  	= 1.1e4;					% endTime -- [time unit]
checkTime   = [3.839, 4.094, 4.350, 4.605, 4.861, 5.117, 5.372, 5.628, 5.884, 6.140, 6.396, 6.652, 6.907, 7.163, 7.419, 7.675, 7.931, 8.186, 8.442, 8.698, 8.954, 9.210, 9.466, 9.722]; 	% ad-hoc implementaton -- see log.spparks.partial

% inputFokkerPlanck = [-9.38918107390727e-06      0.000488148520734925      2.24355902008616e-06      -0.00200778588539395]; % good guess
% inputFokkerPlanck = [6.88060991180513e-06       -0.0360800529639008      3.08642806941265e-05        0.0154445707757066]; % good guess

% 12Mar18: impose constant diffusion term
lenInput = length(inputFokkerPlanck);
% polyDrift = inputFokkerPlanck( 1 : (lenInput - 2) ) ;
% polyDiffusion = inputFokkerPlanck(lenInput - 1 : end); 
polyDrift = inputFokkerPlanck(1);
polyDiffusion = inputFokkerPlanck(2);


% shiftRightToZero = 0.25;				% estimated shifted gamma distribution fit
% O = 3;									% accuracy order of numerical derivatives
sourcePath = pwd;
% Neverystep = 2.5e4;
h = (endSupp - startSupp)/N;
x = [startSupp:h:endSupp]; 				% discretized segment 
% ----------------------------------- OBJECTIVE FUNCTIONS ------------------------------- %
% open and save pdf at t=? where the pdf is matched at t=?
% see log.spparks.partial for more infor about time-series pdf

fprintf('Importing time-series pdf... \n');
pHistorySpparks = zeros(41, length(x));
for i = 0:39
	pHistorySpparks(i+1, :) = ksdensity(csvread(strcat(sourcePath,'/../probabilityRepo/','log.',var,'.', num2str(i), '.dat')), x);
	fprintf('Imported + Smoothed time-series pdf: %d/40... \n', i);
end
fprintf('Finished Importing time-series pdf... \n');





% ForwardFokkerPlanckModelOnlinePlot(inputFokkerPlanck)
ForwardFokkerPlanckModelOnlinePlotAfterCalib(inputFokkerPlanck, pHistorySpparks)

% ForwardFokkerPlanckModel(inputFokkerPlanck)

