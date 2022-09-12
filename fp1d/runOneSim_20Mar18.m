clear all;
close all;
clc;
format longg

system('rm -v *png');

inputFokkerPlanck = dlmread('log.input.dat');


% copy from ForwardFokkerPlanck.m 
% ----------------------------------- PARAMETERS SETTINGS ---------------------------- %
var = 'areaOfGrain';					% name of variable
N = 7.0e3; 								% number of segments
startSupp = -5.0e4; endSupp = 9.0e4;	% support
% safeWidth = 0.1e4;						% width of zero density
% dt = 3.5e-2;							% time-step
% Nfrequency = 10/100 ;					% percentage - show progress at every Nfrequency
% magFactor = 1e2; 						% magnifying factor to penalize the objective function
% startTime/endTime of ForwardFokkerPlanckModel
% startTime 	= 1.375e0;					% startTime -- [time unit]
% endTime  	= 1.1e4;					% endTime -- [time unit]
% checkTime   = [10, 100, 1000, 10000]; 	% ad-hoc implementaton -- see log.spparks.partial

% inputFokkerPlanck = [-9.38918107390727e-06      0.000488148520734925      2.24355902008616e-06      -0.00200778588539395]; % good guess
% inputFokkerPlanck = [6.88060991180513e-06       -0.0360800529639008      3.08642806941265e-05        0.0154445707757066]; % good guess

% 12Mar18: impose constant diffusion term
% lenInput = length(inputFokkerPlanck);
% polyDrift = inputFokkerPlanck( 1 : (lenInput - 2) ) ;
% polyDiffusion = inputFokkerPlanck(lenInput - 1 : end); 


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
system('cp -v ../model/*.m .');
system('cp -v ../model/qsub* .');
% y = ForwardFokkerPlanckModelOnlinePlotAfterCalib(inputFokkerPlanck, pHistorySpparks);
y = ForwardFokkerPlanckModel(inputFokkerPlanck, pHistorySpparks);
dlmwrite('log.output.dat', y, 'delimiter', '\t', 'precision', '%8e');


