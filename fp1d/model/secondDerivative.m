function udotdot = secondDerivative(u,x,D2)
	% compute the second derivative by five-point stencil on 1D problem up to h^4
	% assign 0 at the end point; central difference for the next to end point
	% n = length(u); udotdot = zeros(1,n);
	% h = x(2) - x(1); % assuming uniform mesh

	% % https://en.wikipedia.org/wiki/Finite_difference#Higher-order_differences -- forward/backward finite difference for end points
	% udotdot(1) = (u(3) - 2*u(2) + u(1)) / h^2;
	% udotdot(n) = (u(n-2) - 2*u(n-1) + u(n)) / h^2;

	% for i = 3:n-2
	% 	udotdot(i) = 1/(12*h^2) * (-u(i+2) + 16*u(i+1)  - 30*u(i) + 16*u(i-1) - u(i-2));
	% end
	% udotdot(2) = 1/(h^2) * (u(3) - 2*u(2) + u(1));
	% udotdot(n-1) = 1/(h^2) * (u(n) - 2*u(n-1) + u(n-2));

	% due to numerical instability, change five-point stencil to firstDerivative
	% udotdot = firstDerivative(firstDerivative(u,x),x);

	% h = x(2) - x(1);
	% O = 5;
	% u = u(:);
	% udotdot = f_derivative(2,O,u,h);

	udotdot = D2 * u;
end
