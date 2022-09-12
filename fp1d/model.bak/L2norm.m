function integralValue = L2norm(p,q,x)
	% Kullback-Leibler distance w.r.t p; p = reference density

	% assuming p and q die out at the boundary of x
	if length(p) ~= length(q) 
		error('Discretized length of 2 distributions are not the same');
	end
	if length(p) ~= length(x) 
		error('Discretized length of distribution p and support are not the same');
	end
	if length(q) ~= length(x) 
		error('Discretized length of distribution q and support are not the same');
	end

	% p = experimental
	% q = simulated
	p = p(:);
	q = q(:);
	integralValue = trapz(x , abs(p-q).^2 );
	integralValue = sqrt(integralValue);
end

