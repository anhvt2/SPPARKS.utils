function xNext = getNextSamplingPoint(dmodel, xLB, xUB)
	% define the global bounds for x at here
	% functions called: 
		% calcNegAcquis -> calcAcquis
		% cmaes

	[fBest,iBest] = max(dmodel.origY);
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
	sigma = 0.25 * min(xUB - xLB); 

	% NOTE: argmax instead of argmin in CMAES (need a -f)
	% xInitCMAES = getGoodInitSamplingPoint(parentPath);
	% xNext = cmaes('calcNegAcquis', xInitCMAES, sigma, OPTS, dmodel); 
	% xNext = cmaes('calcNegAcquis', xBest, sigma, OPTS, dmodel); 
	% xNext = cmaes('calcNegAcquis', 0.5 * (xBest + xInitCMAES), sigma, OPTS, dmodel); 
	for i = 1:l
		xInitCMAES(i) = xLB(i) + rand() * (xUB(i) - xLB(i));
	end
	% w = rand(); 
	w = 1;
    xNext = cmaes('calcNegAcquis', w * xBest + (1 - w) * xInitCMAES, sigma, OPTS, dmodel); 
    
end

