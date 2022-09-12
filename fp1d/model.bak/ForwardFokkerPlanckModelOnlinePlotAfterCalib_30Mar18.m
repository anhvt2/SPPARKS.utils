function statisticalDistance = ForwardFokkerPlanckModelOnlinePlotAfterCalib(inputFokkerPlanck, pHistorySpparks)
% convert from fp1dfdMsdtotal_9Mar18.m only input -- output

% compute 1D Fokker-Planck equation based on finite-difference method and ODE45 sovler
% ref: http://faculty.washington.edu/finlayso/ebook/pde/FD/FD.Matlab.htm
% assume form of equation to be numerically solved: du/dt = [a*u + b*du/dx + c*d^2u/dx^2]
inputFokkerPlanck 
% ----------------------------------- PARAMETERS SETTINGS ---------------------------- %
var = 'areaOfGrain';					% name of variable
N = 6.0e3; 								% number of segments
startSupp = -3.0e4; endSupp = 9.0e4;	% support
safeWidth = 0.1e4;						% width of zero density
dt = 3.5e-2;							% time-step
Nfrequency = 10/100 ;					% percentage - show progress at every Nfrequency
magFactor = 1e2; 						% magnifying factor to penalize the objective function
% startTime/endTime of ForwardFokkerPlanckModel
startTime 	= 1.375e0;					% startTime -- [time unit]
endTime  	= 1.1e4;					% endTime -- [time unit]
checkTime   = [10, 100, 1000, 10000]; 	% ad-hoc implementaton -- see log.spparks.partial

% inputFokkerPlanck = [-9.38918107390727e-06      0.000488148520734925      2.24355902008616e-06      -0.00200778588539395]; % good guess
% inputFokkerPlanck = [6.88060991180513e-06       -0.0360800529639008      3.08642806941265e-05        0.0154445707757066]; % good guess

% inputFokkerPlanck = [5.13500783e-03 -1.27475697e-03 1.22675403e-06 2.38070822e-08];
% 12Mar18: impose constant diffusion term
lenInput = length(inputFokkerPlanck);
polyDrift = inputFokkerPlanck( 1 : (lenInput - 1) ) ;
polyDiffusion = inputFokkerPlanck(lenInput : end); 


% shiftRightToZero = 0.25;				% estimated shifted gamma distribution fit
O = 3;									% accuracy order of numerical derivatives
sourcePath = pwd;
Neverystep = 2.5e3;						% number of iteration to print images
h = (endSupp - startSupp)/N;
x = [startSupp:h:endSupp]; 				% discretized segment 


% ----------------------------------- INITIAL CONDITION ------------------------------- %

% comment: read initial density from synthesized logFiles

p0 = ksdensity(csvread(strcat(sourcePath,'/../probabilityRepo/','log.',var,'.2.dat')), x); % 17Mar18: modify to fit this particular example
pInit = p0;

% ------------------------- SMOOTHING BY DATA REGULARIZATION -------------------------- %
% compute derivative matrix with arbitrary accuracy order

% theory + numerical recommendations provided by
% Stickel, Jonathan J. "Data smoothing and numerical differentiation by a regularization method." 
% Computers & chemical engineering 34.4 (2010): 467-475.

D1 = derivativeMatrix(1,O,h,length(x)-1); D1 = sparse(D1); 
D2 = derivativeMatrix(2,O,h,length(x)-1); D2 = sparse(D2); 
D3 = derivativeMatrix(3,O,h,length(x)-1); D3 = sparse(D3); 
D4 = derivativeMatrix(4,O,h,length(x)-1); D4 = sparse(D4); 
D5 = derivativeMatrix(5,O,h,length(x)-1); D5 = sparse(D5); 
D6 = derivativeMatrix(6,O,h,length(x)-1); D6 = sparse(D6); 

close all;
lambda = 1.500e+02; p0 = p0(:);
DMatrix = D1;
A = eye(size(DMatrix)) + lambda*transpose(DMatrix)*DMatrix; A = sparse(A);

p0Smooth = mldivide( A , p0); p0Smooth = p0Smooth/trapz(x,p0Smooth);


% figure(1); hold on; box on; grid on;
% plot(x,p0Smooth,'b-.','LineWidth',2);
% plot(x,p0,'r-.','LineWidth',2);
% legend('regularized','histogram');
% title(sprintf('L2norm = %.8f', L2norm(p0,p0Smooth,x)));
% pause('on'); pause;
% close all;

p0 = p0Smooth; % comment = do not regularize; not comment = regularize

% --------------------------------- PRELIMINARY PROCESS ------------------------------- %
% zero-flux BC

leftSafeEnd = min(x) + safeWidth; 
rightSafeEnd = max(x) - safeWidth; 

% ----------------------------------- FINAL CONDITION ------------------------------- %
pFinal = ksdensity(csvread(strcat(sourcePath,'/../probabilityRepo/','log.',var,'.37.dat')), x); % 17Mar18; modify to fit this particular example


% ----------------------------------- OBJECTIVE FUNCTIONS ------------------------------- %
% open and save pdf at t=? where the pdf is matched at t=?
% see log.spparks.partial for more infor about time-series pdf

% fprintf('Importing time-series pdf... \n');
% pHistorySpparks = zeros(41, length(x));
% for i = 0:39
% 	pHistorySpparks(i+1, :) = ksdensity(csvread(strcat(sourcePath,'/../probabilityRepo/','log.',var,'.', num2str(i), '.dat')), x);
% 	fprintf('Imported + Smoothed time-series pdf: %d/40... \n', i);
% end
% fprintf('Finished Importing time-series pdf... \n');

% ----------------------------------- SOLVERS ------------------------------- %
startClockTime = cputime();
x = x(:); 		% convert to column vector
pLast = p0(:); 	% convert to column vector
if exist('M') ~= 1
	M = int32((endTime - startTime)/dt);
end

% create an array to store pdf at certain index in iCheck
% iCheck -- array of index which pdf should be checked
iCheck = int32(checkTime - startTime)/dt; 		% compute distance at these iCheck
pSimCheck = zeros( length(iCheck), length(x) );	% store pdf -- iCheck rows, length(x) cols

% online plot -- turn on for debug
figure(1); 
xlabel('support' , 'HandleVisibility' , 'off' ) ;
% xlim([startSupp endSupp]);
xlim([0 endSupp]);
ylim([1.5*min(p0) 1.5*max(p0)]);
ylabel('p.d.f' , 'HandleVisibility' , 'off' ) ;
set( gca , 'NextPlot' , 'replacechildren' ) ;
hold on;

currentTimeArray = []; meanArray = []; stdArray = [];
fprintf('Start iterating...\n');

for i = 1:M
	currentTime = startTime + i * dt;

	% comment: use RK4 integrator. caution: can be numerically unstable. 
	% 			see [Numerical solution of two dimensional FP equations. Applied Mathematics and Computations. 1999. Zorzano, Mais, Vaquez. ]
	
	% dpOverdt = RK4(pLast, beta1, alpha1, beta2, alpha2, h);

	% implement Crank-Nicolson method -- semi-implicit

	dpOverdtExplicit = EulerIntegrator(pLast, x, polyDrift, polyDiffusion, h, D1, D2, currentTime); % explicit
	dpOverdtImplicit = EulerIntegrator(pLast + dpOverdtExplicit * dt, x, polyDrift, polyDiffusion, h, D1, D2, currentTime);
	dpOverdt = 0.5 * (dpOverdtImplicit + dpOverdtExplicit); 
	% dpOverdt = dpOverdtImplicit;

	pCurrent = pLast + dpOverdt * dt; clear dpOverdtExplicit dpOverdtImplicit;

	% ------------------------------------------------ ENHANCE CONDITION ------------------------------------------------ %
	% pCurrent = abs(pCurrent);					% absolute value if negative
	% pCurrent(find(pCurrent<0)) =  0;			% zero value if negative
	% pCurrent = pCurrent/sum(pCurrent);		% normalize by sum
	pCurrent = pCurrent/trapz(x,pCurrent);		% normalize by area
	% pCurrent(find(x<leftSafeEnd))  = zeros(length(find(x<leftSafeEnd)),1);
	% pCurrent(find(x>rightSafeEnd)) = zeros(length(find(x>rightSafeEnd)),1);
	% pCurrent(find(x<leftSafeEnd))  = spline([x(1) x(2) x(max(find(x<leftSafeEnd))+1)],[0 0 pCurrent(max(find(x<leftSafeEnd))+1)] , x(find(x<leftSafeEnd)));
	% pCurrent(find(x>rightSafeEnd)) = spline([x(min(find(x>rightSafeEnd))-1) x(length(x)-1) x(length(x))],[pCurrent(min(find(x>rightSafeEnd))-1) 0 0] , x(find(x>rightSafeEnd)));
	% pCurrent(1) = 0;							% startSupp BC
	% pCurrent(length(pCurrent)) = 0;				% endSupp BC
	% pCurrent(2) = 0;
	% pCurrent(length(pCurrent)-1) = 0;
	% pCurrent = mldivide( A , pCurrent); pCurrent = pCurrent/trapz(x,pCurrent); % regularization -- note: matrix A pre-computed. caution: reg. param is nut

	% -------------------------------------------------- UPDATE CURRENT DENSITY ----------------------------------------- %
	currentTimeArray = [currentTimeArray; i * dt + startTime]; meanArray = [meanArray; trapz(x,x.*pCurrent)]; stdArray = [stdArray; sqrt(trapz(x,(x-trapz(x,x.*pCurrent)).^2.*pCurrent)) ];
	pLast = pCurrent; % update the last prob vector

	% -------------------------------------------------- ONLINE PLOT/LOG ------------------------------------------------ % 
	% compute checkStatisticalDistance 
	checkStatisticalDistance = L2norm(pFinal,pCurrent,x);

	for j = 1:length(iCheck)
		if i == iCheck(j)
			pSimCheck(j, :) = pCurrent;
			fprintf('Save pdf to compare at iteration %d\n', i);
		end
	end

	% progress log

	% if (mod(i,Neverystep) == 0)
	% 	currentClockTime = cputime();
	% 	fprintf('Time-step %d/%d reached. Elasped time: %10.4f minutes; statisticalDistance = %10.4f.\n', i , M , (currentClockTime - startClockTime)/60, checkStatisticalDistance);
	% end

	% online plot -- turn on for debug
	if (mod(i,Neverystep) == 0)
		currentClockTime = cputime();
		fprintf('Time-step %d/%d reached. Elasped time: %10.4f seconds; L2norm = %.4e; KLdistance = %.4e.\n', i , M , (currentClockTime - startClockTime), L2norm(pFinal,pCurrent,x), KLdistance(pCurrent, pFinal, x) );

		% % title(sprintf('Fokker-Planck density at iteration %d (time-step %6.2f). mean = %6.4f; deviation = %6.4f ' , ...
		% % 				 i , i * dt + startTime, trapiterationz(x,x.*pCurrent) , sqrt(trapz(x,x.^2.*pCurrent)) ) , 'HandleVisibility' , 'off' ) ;
		title(sprintf('Fokker-Planck density at iteration %d/%d (time-step %6.2f); L2norm = %6.4e; KLdistance = %6.4e' , ...
						 i, M, i * dt + startTime, L2norm(pCurrent, pFinal, x), KLdistance(pCurrent, pFinal, x) ) , 'HandleVisibility' , 'off' ) ;

		graph(2) = plot(x,pInit,'ro:','LineWidth',2); 
		graph(3) = plot(x,pFinal,'go--', 'LineWidth',2);
		% graph(1) = plot(x,pCurrent,'b','LineWidth',2);
		graph(1) = plot(x(find(x>0)), pCurrent(find(x>0)) / trapz( x(find(x>0)), pCurrent(find(x>0)) ), 'b', 'LineWidth', 2);
		legend('init density' ,'final density','current density');
		figName = sprintf('densityAtStep%d.png',i); saveas(gcf,figName); % save figure
		ylim([0, 1.5 * max(pCurrent)]); set(graph(1),'Visible','off'); % delete graph(1);
	end

	% -------------------------------------------------- SAFE GUARD / QUIT FAST ------------------------------------------ %
	% enhance for divergence cases
	if checkStatisticalDistance > 10;
		statisticalDistance = 100;
		fprintf('ForwardFokkerPlanckModel diverges at iteration %d.\n', i); % notice
		return
	end
end

% ----------------------------------- POST PROCESS ------------------------------- %


close all;
pSimFinal = pCurrent;
fprintf('Kullback-Leibler distance between pFinal and pSimFinal: %10.8f\n', KLdistance(pFinal,pSimFinal,x) );

fprintf('L2norm distance between pFinal and pSimFinal: %10.8f\n', L2norm(pFinal,pSimFinal,x) );

% ---------------------------------- DIAGNOSTICS --------------------------------- %
% figure(1); subplot(1,2,1); hold on; box on;
% polyMean		= [1.18007764e-03 3.25738705e-01]; 		% polynomial fit for mean
% plot(currentTimeArray , meanArray, 'b:.');
% plot(currentTimeArray , polyval(polyMean , currentTimeArray), 'r-.');
% title('mean. D1(t); D2(x)'); ylabel(var); xlabel('time (ps)');

% figure(1); subplot(1,2,2);  hold on; box on;
% polyStd		= [5.09351806e-04 2.46014099e-02]; 		% polynomial fit for std
% plot(currentTimeArray , stdArray, 'b:.');
% plot(currentTimeArray , polyval(polyStd , currentTimeArray), 'r-.');
% title('standard deviation. D1(t); D2(x)'); ylabel(var); xlabel('time (ps)');

dlmwrite('pSimFinal.dat', [pSimFinal], 'delimiter', '\n', 'precision', '%0.16f');
dlmwrite('x.dat', [x], 'delimiter', '\n', 'precision', '%0.16f');

% ---------------------------------- COMPUTE FINAL DISTANCE --------------------------------- %
% see log.spparks.partial for more infor about what constitutes a final distance
% check at t = 10, 100, 1000, and 10000 (20,000 does not exist)
% index 10, 19, 28, 37
% line  11, 20, 29, 38,

% pSimCheck -- calculate from the FFP
% pHistorySpparks -- from SPPARKS model

statisticalDistance = 	(	L2norm(pSimCheck(1,:), pHistorySpparks(11,:), x) + ... 
							L2norm(pSimCheck(2,:), pHistorySpparks(20,:), x) + ... 
							L2norm(pSimCheck(3,:), pHistorySpparks(29,:), x) + ... 
							L2norm(pSimCheck(4,:), pHistorySpparks(38,:), x)	) * ... 
						(	KLdistance(pSimCheck(1,:) , pHistorySpparks(11,:), x) + KLdistance(pHistorySpparks(11,:) , pSimCheck(1,:) , x) + ... 
						 	KLdistance(pSimCheck(2,:) , pHistorySpparks(20,:), x) + KLdistance(pHistorySpparks(20,:) , pSimCheck(2,:) , x) + ... 
							KLdistance(pSimCheck(3,:) , pHistorySpparks(29,:), x) + KLdistance(pHistorySpparks(29,:) , pSimCheck(3,:) , x) + ... 
							KLdistance(pSimCheck(4,:) , pHistorySpparks(38,:), x) + KLdistance(pHistorySpparks(38,:) , pSimCheck(4,:) , x)  )  ;

statisticalDistance = statisticalDistance * magFactor;
fprintf('statisticalDistance = %10.8f\n', statisticalDistance );

end 
