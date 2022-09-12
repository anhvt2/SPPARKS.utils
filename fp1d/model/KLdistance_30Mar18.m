function dKL = KLdistance(p,q,x)
	penal_dKL = 1e5; % penalized KL
	% Kullback-Leibler distance w.r.t p; p = reference density
	% try to match p (calibrated) from q (simulated)

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
	tol = 1e-8; 
	% tol = eps; % deprecated - 30Mar18
	% (1) prevent NaN numerical issue - too small -> +NaN
	% (2) p(i) = 0 -> log 0 = -NaN
	% (3) 30Mar18: add the case which p>0 and q<0 -> quick penalty
	for i = 1:length(x)
		if q(i) > tol && p(i) > tol
			integralFunction(i) = p(i) * log( p(i) / q(i) );
		% add a penalty case and quick return
		elseif p(i) > tol && q(i) < eps
			dKL = penal_dKL; return;
		else
			integralFunction(i) = 0;
		end
	end
	% https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
	dKL = trapz( x , real(integralFunction) );
end

