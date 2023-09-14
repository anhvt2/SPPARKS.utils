nspins = 100000;
%Specify Crystal symmetry
cs = crystalSymmetry('cubic');
ss = specimenSymmetry('orthorhombic');
%Specify the distribution function to use
odfUni = uniformODF(cs,ss);
fiber_odf = 0.1*uniformODF(cs,ss) + 0.6*fibreODF(Miller(0,0,0,1,cs),zvector,ss);
%Calculate sampled orientations
oriUni = calcOrientations(odfUni,nspins);
oriFiber = calcOrientations(fiber_odf,nspins);

%plotPDF(fiber_odf,[Miller(1,0,0,cs),Miller(1,1,0,cs),Miller(1,1,1,cs)]);
plotIPDF(oriFiber,[vector3d.X,vector3d.Y,vector3d.Z]);
figure(2)
%plotPDF(odfUni,[Miller(1,0,0,cs),Miller(1,1,0,cs),Miller(1,1,1,cs)]);
plotIPDF(odfUni,[vector3d.X,vector3d.Y,vector3d.Z]);

uniM(:,1) = oriUni.phi1;
uniM(:,2) = oriUni.Phi;
uniM(:,3) = oriUni.phi2;
csvwrite("Uniform.csv", uniM);

fiberM(:,1) = oriFiber.phi1;
fiberM(:,2) = oriFiber.Phi;
fiberM(:,3) = oriFiber.phi2;
csvwrite("Fiber.csv", fiberM);