% function K = timeDerivative(u,x,polyDrift,polyDiffusion,polyMean,polyStd,polytmt,polyfmt,h,D1,D2,D3,D4,currentTime)

function K = timeDerivative(u, x, polyDrift, polyDiffusion, h, D1, D2, currentTime)

	K = zeros(length(u),1);
	u = u(:);
	x = x(:);

	K = - polyval(polyDrift,currentTime) * firstDerivative( u , x , D1) + polyval(polyDiffusion,currentTime) * secondDerivative( u , x , D2);


end

