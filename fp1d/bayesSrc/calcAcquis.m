function a = calcAcquis(x,dmodel)

	[y, ~, mse, ~] = predictor(x,dmodel);

	if mse < 0;
		mse = 0;
	end

	% % GP-EI acquisition function
	[fBest,iBest] = min(dmodel.origY);
	gammaX = ( y - fBest )/sqrt(mse);
	a = sqrt(mse) * (gammaX * normcdf(gammaX) + normpdf(gammaX) );

	% % GP-UCB acquisition function
	% kappa = 1e2;
	% a = y + kappa * sqrt(mse);
	fprintf('Acquisition: mu = %0.8f\n', y);
	fprintf('Acquisition: mse = %0.8f\n', mse);
	return;


end

