function udot = firstDerivative(u,x,D1)
	% compute the first derivative by five-point stencil on 1D problem up to h^4
	% assign 0 at the end point; central difference for the next to end point
	% n = length(u); udot = zeros(1,n);
	% h = x(2) - x(1); % assuming uniform mesh

	% % http://www.shodor.org/cserd/Resources/Algorithms/NumericalDifferentiation/ -- end point derivative
	% udot(1) = (-3*u(1) + 4*u(2) - u(3))/(2*h);
	% udot(n) = (-3*u(n) + 4*u(n-1) - u(n-2))/(2*h);

	% for i = 3:n-2
	% 	udot(i) = 1/(12*h) * (-u(i+2) + 8*u(i+1) - 8*u(i-1) + u(i-2));
	% end
	% udot(2) = 1/(2*h) * (u(3) - u(1));
	% udot(n-1) = 1/(2*h) * (u(n) - u(n-2));

	% % due to numerical instability, change five-point stencil to first derivative
	% % udot(1) = 0;
	% % udot(n) = 0;
	% % for i = 2:n-1
	% % 	udot(i) = (u(i+1) - u(i)) / (x(i+1) - x(i));
	% % end

	% compute by f_derivative - finite difference method to arbitrary accuracy
	% h = x(2) - x(1);
	% O = 5;
	% u = u(:);
	% udot = f_derivative(1,O,u,h);

	% compute by derivativeMatrix - fast implementation for FP
	udot = D1 * u; % assuming uniform grids
end
