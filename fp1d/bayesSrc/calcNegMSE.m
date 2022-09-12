function negMSE = calcNegMSE(x,dmodel)
	% function called: calcAcquis
	% flip sign of acquisition function 
	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	[y, ~, mse, ~] = predictor(x,dmodel); 
	negMSE = - sqrt(mse);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('Negative MSE Function Message: negMSE = %.8f\n', negMSE);
	fprintf('\n');
	return 
end

