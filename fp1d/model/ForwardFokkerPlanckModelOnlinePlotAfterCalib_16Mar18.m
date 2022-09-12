function statisticalDistance = ForwardFokkerPlanckModelOnlinePlotAfterCalib(inputFokkerPlanck)
% convert from fp1dfdMsdtotal_9Mar18.m only input -- output

% compute 1D Fokker-Planck equation based on finite-difference method and ODE45 sovler
% ref: http://faculty.washington.edu/finlayso/ebook/pde/FD/FD.Matlab.htm
% assume form of equation to be numerically solved: du/dt = [a*u + b*du/dx + c*d^2u/dx^2]

% ----------------------------------- PARAMETERS SETTINGS ---------------------------- %
var = 'msdtotal';						% name of variable
N = 5.0e3; 								% number of segments
startSupp = -0.1; endSupp = 1.5;	 	% support
safeWidth = 0.1e-0;						% width of zero density
dt = 4.0e-3;							% time-step - fs
Nfrequency = 10/100 ;					% percentage - show progress at every Nfrequency

% computing time
startTime    = int32(80e3)/1e3;			% unit fs -> ps
endTime  = int32(160e3)/1e3;			% unit fs -> ps
checkTime  = int32(200e3)/1e3;			% unit fs -> ps

% polyDrift 		= [5.13500783e-03 -1.27475697e-03];		% polynomial fit for drift
% polyDiffusion 	= [1.22675403e-06 2.38070822e-08];		% polynomial fit for diffusion

% inputFokkerPlanck = [5.13500783e-03 -1.27475697e-03 1.22675403e-06 2.38070822e-08];
% 12Mar18: impose constant diffusion term
lenInput = length(inputFokkerPlanck);
polyDrift = inputFokkerPlanck( 1 : (lenInput - 2) ) ;
polyDiffusion = inputFokkerPlanck(lenInput - 1 : end); 


shiftRightToZero = 0.25;				% estimated shifted gamma distribution fit
O = 3;									% accuracy order of numerical derivatives
sourcePath = pwd;
Neverystep = 1.0e3;
h = (endSupp - startSupp)/N;
x = [startSupp:h:endSupp]; 				% discretized segment 

% ----------------------------------- INITIAL CONDITION ------------------------------- %

% comment: read initial density from synthesized logFiles
y = csvread(strcat(sourcePath,'/../probabilityRepo/',var,'/',num2str(startTime * 1e3),'/','log.',var,'.txt'));

% KDE

p0 = ksdensity(y,x);
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
lambda = 1.500e-05; p0 = p0(:);
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
p0 = p0Smooth;

% --------------------------------- PRELIMINARY PROCESS ------------------------------- %
% zero-flux BC

leftSafeEnd = min(x) + safeWidth; 
rightSafeEnd = max(x) - safeWidth; 

% ----------------------------------- FINAL CONDITION ------------------------------- %
y = csvread(strcat(sourcePath,'/../probabilityRepo/',var,'/',num2str(endTime * 1e3),'/','log.',var,'.txt'));
pFinal = ksdensity(y,x);

y = csvread(strcat(sourcePath,'/../probabilityRepo/',var,'/',num2str(checkTime * 1e3),'/','log.',var,'.txt'));
pCheckFinal = ksdensity(y,x);


% ----------------------------------- SOLVERS ------------------------------- %
startClockTime = cputime();
x = x(:); 		% convert to column vector
pLast = p0(:); 	% convert to column vector
if exist('M') ~= 1
	% M = (endTime - startTime)/dt;
	M = (checkTime - startTime)/dt;
end

figure(1); 
xlabel('support' , 'HandleVisibility' , 'off' ) ;
xlim([startSupp endSupp]);
ylim([1.5*min(p0) 1.5*max(p0)]);
ylabel('p.d.f' , 'HandleVisibility' , 'off' ) ;
set( gca , 'NextPlot' , 'replacechildren' ) ;
hold on;

currentTimeArray = []; meanArray = []; stdArray = [];
for i = 1:M
	currentTime = startTime + i * dt;

	% comment: use RK4 integrator. caution: can be numerically unstable. 
	% 			see [Numerical solution of two dimensional FP equations. Applied Mathematics and Computations. 1999. Zorzano, Mais, Vaquez. ]
	
	% dpOverdt = RK4(pLast, beta1, alpha1, beta2, alpha2, h);

	% implement Crank-Nicolson method -- semi-implicit

	dpOverdtExplicit = EulerIntegrator(pLast, x, polyDrift, polyDiffusion, h, D1, D2, currentTime); % explicit
	dpOverdtImplicit = EulerIntegrator(pLast + dpOverdtExplicit * dt, x, polyDrift, polyDiffusion, h, D1, D2, currentTime);
	dpOverdt = 0.5 * (dpOverdtImplicit + dpOverdtExplicit); 

	pCurrent = pLast + dpOverdt * dt; clear dpOverdtExplicit dpOverdtImplicit;

	% ------------------------------------------------ ENHANCE CONDITION ------------------------------------------------ %
	% pCurrent = abs(pCurrent);					% absolute value if negative
	pCurrent(find(pCurrent<0)) =  0;			% zero value if negative
	% pCurrent = pCurrent/sum(pCurrent);		% normalize by sum
	pCurrent = pCurrent/trapz(x,pCurrent);		% normalize by area
	% pCurrent(find(x<leftSafeEnd))  = zeros(length(find(x<leftSafeEnd)),1);
	% pCurrent(find(x>rightSafeEnd)) = zeros(length(find(x>rightSafeEnd)),1);
	% pCurrent(find(x<leftSafeEnd))  = spline([x(1) x(2) x(max(find(x<leftSafeEnd))+1)],[0 0 pCurrent(max(find(x<leftSafeEnd))+1)] , x(find(x<leftSafeEnd)));
	% pCurrent(find(x>rightSafeEnd)) = spline([x(min(find(x>rightSafeEnd))-1) x(length(x)-1) x(length(x))],[pCurrent(min(find(x>rightSafeEnd))-1) 0 0] , x(find(x>rightSafeEnd)));
	pCurrent(1) = 0;							% startSupp BC
	pCurrent(length(pCurrent)) = 0;				% endSupp BC
	% pCurrent(2) = 0;
	% pCurrent(length(pCurrent)-1) = 0;
	% pCurrent = mldivide( A , pCurrent); pCurrent = pCurrent/trapz(x,pCurrent); % regularization -- note: matrix A pre-computed. caution: reg. param is nut

	% -------------------------------------------------- UPDATE CURRENT DENSITY ----------------------------------------- %
	currentTimeArray = [currentTimeArray; i * dt + startTime]; meanArray = [meanArray; trapz(x,x.*pCurrent)]; stdArray = [stdArray; sqrt(trapz(x,(x-trapz(x,x.*pCurrent)).^2.*pCurrent)) ];
	pLast = pCurrent; % update the last prob vector

	% -------------------------------------------------- ONLINE PLOT/LOG ------------------------------------------------ % 
	% compute statisticalDistance 
	statisticalDistance = L2norm(pFinal,pCurrent,x);

	% progress log

	% if (mod(i,Neverystep) == 0)
	% 	currentClockTime = cputime();
	% 	fprintf('Time-step %d/%d reached. Elasped time: %10.4f minutes; statisticalDistance = %10.4f.\n', i , M , (currentClockTime - startClockTime)/60, statisticalDistance);
	% end

	% online plot 
	if (mod(i,Neverystep) == 0)
		currentClockTime = cputime();
		if (mod(i,Neverystep) == 0)
			currentClockTime = cputime();
			fprintf('Time-step %d/%d reached. Elasped time: %10.4f minutes; statisticalDistance = %10.4f.\n', i , M , (currentClockTime - startClockTime)/60, statisticalDistance);
		end

		% title(sprintf('Fokker-Planck density at step %d (time-step %6.4f ps). mean = %6.4f; deviation = %6.4f ' , ...
		% 				 i , i * dt + startTime, trapz(x,x.*pCurrent) , sqrt(trapz(x,x.^2.*pCurrent)) ) , 'HandleVisibility' , 'off' ) ;
		% title(sprintf('Fokker-Planck density at step %d (time-step %6.4f ps)' , ...
		% 				 i , i * dt + startTime) , 'HandleVisibility' , 'off' ) ;
		title(sprintf('Fokker-Planck density at step %d (time-step %6.4f ps). L2-norm distance b/w pFinal and pCurrent = %6.4f' , ...
						 i , i * dt + startTime, statisticalDistance ) , 'HandleVisibility' , 'off' ) ;
		

		graph(2) = plot(x,pInit,'ro:','LineWidth',2); 
		graph(3) = plot(x,pFinal,'go--', 'LineWidth',2);
		graph(4) = plot(x,pCheckFinal,'mo--', 'LineWidth',2);

		graph(1) = plot(x,pCurrent,'b','LineWidth',3);
		legend('init density' ,'calibrated density','test density','current density');
		figName = sprintf('densityAtStep%d.png',i); saveas(gcf,figName);
		set(graph(1),'Visible','off'); % delete graph(1);
	end
end

% ----------------------------------- POST PROCESS ------------------------------- %


close all;
pSimFinal = pCurrent;
fprintf('Kullback-Leibler distance between pFinal and pSimFinal: %10.8f\n', KLdistance(pFinal,pSimFinal,x) );

fprintf('L2-norm distance between pFinal and pSimFinal: %10.8f\n', L2norm(pFinal,pSimFinal,x) );

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

statisticalDistance = L2norm(pFinal,pSimFinal,x);

end 
