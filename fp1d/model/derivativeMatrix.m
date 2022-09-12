function coef = derivativeMatrix(m,O,h,N)

	n = m + O;
	C = Coeff(m,O,h);
	Bickley_scale = h^m * factorial(n-1)/factorial(m);
	Bickley_coeff = C * Bickley_scale;
	Bickley_coeff = int64(Bickley_coeff);

	% modifying the numerical stencil for a unified accuracy
	if (mod(m,2) == 0 && mod(O,2)~=0)
		np = n+1;
		j = np/2;
		for i = 1:j-1
			coef_u(i,:) = [C(i,:) 0];
		end

		coef_m(1,:) = [0 C(j-1,:)];
		coef_m(2,:) = [C(j+1,:) 0];
		for i = j+2:np
			coef_l(i-j-1,:) = [0 C(i-1,:)];
		end

		C = [ coef_u;coef_m;coef_l ];

	else
		np = n;
	end

	% generating the derivative over the real domain of N+1 nodes
	% [N+1 b] = size(f);

	if N+1<np
		fprintf('N+1 = %d, np = %d\n',N+1,np) % debug
		error('The size of the given data should be greater than or equal to %d. \n\t Try using larger set of data or decrease the accuracy order.', np);
	end

	if mod(np,2)==0
		j = np/2;
	else
		j = (np+1)/2;
	end

	U = [C(1:j-1,:) zeros(j-1,N+1-np)];
	L = [zeros(np-j,N+1-np) C(j+1:np,:) ];
	k = C(j,:);
	M = zeros(N+1-np+1,N+1);

	for i = 1:N+1-np+1
		for j = 1:N+1
			if i==j
				M(i,j:j+np-1) = k;
			end
		end
	end

	coef = [U;M;L];

end