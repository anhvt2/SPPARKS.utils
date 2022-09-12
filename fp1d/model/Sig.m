function S = Sig(v,n)
	S = ones(n,n);
	s = S;
	for k = 1:n
		vv = v;
		vv(k) = v(1); 
		for i = 2:n
			s(i,i) = s(i-1,i-1)*vv(i);
			if i>2
				for j = i-1:-1:2
					s(j,i) = s(j-1,i-1) * vv(i) + s(j,i-1);
				end
			end
		end
	S(:,k) = s(:,n);
	end
end