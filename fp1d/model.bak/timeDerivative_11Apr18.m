% function K = timeDerivative(u,x,polyDrift,polyDiffusion,polyMean,polyStd,polytmt,polyfmt,h,D1,D2,D3,D4,currentTime)

function K = timeDerivative(u, x, polyDrift, polyDiffusion, h, D1, D2, currentTime)

	K = zeros(length(u),1);
	u = u(:);
	x = x(:);

	K = - firstDerivative( polyval(polyDrift,x) .* u , x , D1) + secondDerivative( polyval(polyDiffusion,x) .* u , x , D2);

	% K = - polyval(polyMean , currentTime) * firstDerivative( u , x , D1 ) + secondDerivative( polyval(polyDiffusion,x) .* u , x , D2);

	% K = - polyval(polyMean , currentTime) * firstDerivative( u , x , D1 ) + polyval(polyStd , currentTime) * secondDerivative( u , x , D2 );

	% K = - polyval(polyMean , currentTime) * firstDerivative( u , x , D1 ) + ...
	% secondDerivative( polyval(polyDiffusion,x) .* u , x , D2)  + ...
	% polyval(polyStd,currentTime) * secondDerivative( u , x , D2 );



end