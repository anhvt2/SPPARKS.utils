function C = Coeff(m,O,h)
	% weighting coefficieints calculation

	n = m+O;
	k = [1:n];
	k = k(:);
	for i = 1:n
		v(k) = k-i;
		sigma = Sig(v,n);
		for l = 1:n
			C(i,l) = ((-1)^(l-m-1)*factorial(m)/(factorial(l-1)*factorial(n-l)* h^m ) ) * sigma(n-m,l);
		end
	end

end