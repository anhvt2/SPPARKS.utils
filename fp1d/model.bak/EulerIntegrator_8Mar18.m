function duOverdt = EulerIntegrator(u, x, polyDrift, polyDiffusion, h, D1, D2, currentTime)

	K = timeDerivative(u, x, polyDrift, polyDiffusion, h, D1, D2, currentTime); 

	duOverdt = K;

end
