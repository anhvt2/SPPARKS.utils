function a = calcNegAcquis(x,dmodel)
	% function called: calcAcquis
	% flip sign of acquisition function 
	a = - calcAcquis(x,dmodel);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('Negative Acquisition Function Message: a = %.8f\n', a);
	fprintf('\n');
end

