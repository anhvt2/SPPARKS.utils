function xNext = getNextSamplingPointReduceMSE(dmodel, xLB, xUB)
	% define the global bounds for x at here
	% functions called: 
		% calcNegAcquis -> calcAcquis
		% cmaes

	[fBest,iBest] = min(dmodel.origY);
	xBest = dmodel.origS(iBest,:);
	l = length(xBest);

	OPTS = cmaes; % get default options
	OPTS.LBounds = reshape(xLB, length(xLB), 1); % set lowerbounds
	OPTS.UBounds = reshape(xUB, length(xUB), 1); % set upperbounds
	OPTS.Restarts = 3;
	% run mode
	OPTS.MaxIter = 500; % set MaxIter
	OPTS.MaxFunEvals = 1000; % set MaxFunEvals
	% % debug mode
	% OPTS.MaxIter = 12; % set MaxIter
	% OPTS.MaxFunEvals = 20; % set MaxFunEvals
	% OPTS.DispModulo = 1; % debug: aggressive settings

	% compute sigma based on the thresholds
	sigma = 1; 

	% NOTE: argmax instead of argmin in CMAES (need a -f)
	for i = 1:l
		xInitCMAES(i) = xLB(i) + rand() * (xUB(i) - xLB(i));
	end
	xNext = cmaes('calcNegMSE', xInitCMAES, sigma, OPTS, dmodel); 
	
end

