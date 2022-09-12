function [] = Krig_model(climb)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
global p_krig T_krig metamodel n 

% ----------------------------------------------- DECLARE PARAMETERS -----------------------------------------------
initKrigSwit = 1;
deftheta = 2e-1 * ones(1,n); 	% default hyper params
deflob = 1e-0*ones(1,n);	 	% default lowerbound hyper-params
defupb = 2e3*ones(1,n);		 	% default upperbound hyper-params
Neverycluster = 721;		 	% number of data points in each cluster
Nfit = 4; 	 	 				% number of steps before refining the most recent cluster
fidKrig = fopen('tmpKrig.out','a+');		% debug kriging output file
% ----------------------------------------------- INIT METAMODEL -----------------------------------------------
initKrigSwitch = 1;
if ~isfield(metamodel,'cluster') % if there is metamodel
	fprintf(fidKrig,'debug: start building metamodel\n');
	[norows,nocols] = size(p_krig); fprintf(fidKrig,'size(p_krig) = [%d,%d]\n',norows,nocols);
	[norows,nocols] = size(T_krig); fprintf(fidKrig,'size(T_krig) = [%d,%d]\n',norows,nocols);
	p_T_site = unique([roundn(p_krig',-8),roundn(T_krig',-4)],'rows','stable');
	fprintf(fidKrig,'n (PES dimension) = %d\n',n);
	[norows,nocols] = size(p_T_site); fprintf(fidKrig,'size(p_T_site) = [%d,%d]\n',norows,nocols);
	p_T_site = clstr(p_T_site);
	fprintf(fidKrig, 'clustering according to x coords...\n');
	p_site = p_T_site(:,1:n);
	T_site = p_T_site(:,n+1);
	p_site = customSortX(p_site); % AT
	[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
	[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
	
	% build dmodel
	tic; 
	if length(p_site) > Neverycluster
		% build complete "floor(length(p_site) / Neverycluster)" clusters
		for j = 1:floor(length(p_site) / Neverycluster)
			fprintf('debug flag: building %d cluster sequentially.\n',j);
			tic;
			selIndex = [ (j - 1) * Neverycluster + 1 : j * Neverycluster ];
			[dmodel, ~] = dacefit( p_site(selIndex ,:) , T_site(selIndex,:) , Neverycluster,@regpoly0,@corrgauss,deftheta,deflob,defupb); 
			metamodel.cluster(j) = dmodel;
			toc;
		end
		% build partial last cluster
		if mod(length(p_site),Neverycluster) ~= 0
			lastClusterKrig = floor(length(p_site)/Neverycluster) + 1;
			Ndatapts = mod(length(p_site),Neverycluster);
			selIndex = [(lastClusterKrig - 1) * Neverycluster + 1:length(p_site)];
			% copy the data
			origS = metamodel.cluster(lastClusterKrig - 1).origS;
			origY = metamodel.cluster(lastClusterKrig - 1).origY;
			% replace the data
			origS(1:length(selIndex),:) = p_site(selIndex,:);
			origY(1:length(selIndex),:) = T_site(selIndex,:);
			[dmodel, ~] = dacefit( origS , origY , length(selIndex),@regpoly0,@corrgauss,deftheta,deflob,defupb); 
			metamodel.cluster(lastClusterKrig) = dmodel;
		else
			fprintf(fidKrig, 'mod(length(p_site),Neverycluster) = 0 exactly. \n');
			return
		end
	else
		[dmodel, ~] = dacefit(p_site,T_site,length(p_site),@regpoly0,@corrgauss,deftheta,deflob,defupb); 
		metamodel.cluster(1) = dmodel;
	end
	toc;

	return
else
	initKrigSwitch = 0; 		% metamodel exist
end

% ----------------------------------------------- UPDATE METAMODEL -----------------------------------------------
lastClusterKrig = length(metamodel.cluster);						% last cluster index
Ndatapts = metamodel.cluster(lastClusterKrig).Ndatapts;				% last index of kriging data points
[norows , nocols] = size(p_krig);
% retrieve new data
new_p_site = roundn(p_krig(:,nocols)',-8);
new_T_site = roundn(T_krig(nocols),-4);

fprintf(fidKrig, 'updating cluster. lastClusterKrig = %d\n', lastClusterKrig );
fprintf(fidKrig, 'Ndatapts = %d\n', Ndatapts);
% main scheme:
% is it the first cluster?
%	-> Y: is Ndatapts > Neverycluster
%		-> N: keep adding + don't substract + (Ndatapts += 1) + fit every Nfit
%		-> Y: create the 2nd cluster + reset Ndatapts = 1
%	-> N: is Ndatapts > Neverycluster
%		-> N: keep adding + subtract + (Ndatapts += 1) + fit every Nfit
%		-> Y: create the next cluster + reset Ndatapts = 1 

if lastClusterKrig < 2 
	if metamodel.cluster(lastClusterKrig).Ndatapts < Neverycluster 
		fprintf(fidKrig,'debug flag: adding data points to the 1st cluster. Ndatapts = %d\n',metamodel.cluster(lastClusterKrig).Ndatapts);
		% retrieve last dataset
		p_site = metamodel.cluster(lastClusterKrig).origS;
		T_site = metamodel.cluster(lastClusterKrig).origY;
		% add p_site + T_site
		p_site = [customSortX(new_p_site);p_site];
		T_site = [new_T_site;T_site];
		p_site = customSortX(p_site); % AT
		% counter += 1
		metamodel.cluster(lastClusterKrig).Ndatapts = metamodel.cluster(lastClusterKrig).Ndatapts + 1; 
		tic;
		% fit every Nfit
		if mod(metamodel.cluster(lastClusterKrig).Ndatapts , Nfit) == 0
			theta = metamodel.cluster(lastClusterKrig).theta;
			metamodel.cluster(lastClusterKrig).origS = p_site;
			metamodel.cluster(lastClusterKrig).origY = T_site;
			[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
			[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
			[dmodel, ~] = dacefit(p_site,T_site,metamodel.cluster(lastClusterKrig).Ndatapts,@regpoly0,@corrgauss,theta,deflob,defupb);
			metamodel.cluster(lastClusterKrig) = dmodel;
		else
			metamodel.cluster(lastClusterKrig).origS = p_site;
			metamodel.cluster(lastClusterKrig).origY = T_site;
			[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
			[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
		end
		toc;
	else
		% refine cluster 1
		p_site = metamodel.cluster(lastClusterKrig).origS;
		T_site = metamodel.cluster(lastClusterKrig).origY;
		theta = metamodel.cluster(lastClusterKrig).theta;
		% delete from bottom
		for j = length(p_site):-1:Neverycluster
			p_site(j,:) = [];
			T_site(j,:) = [];
		end
		[metamodel.cluster(lastClusterKrig), ~] = dacefit(p_site,T_site,metamodel.cluster(lastClusterKrig).Ndatapts,@regpoly0,@corrgauss,theta,deflob,defupb);
		fprintf(fidKrig,'debug flag: truncating 1st cluster to Neverycluster size.\n');
		[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
		[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);

		% create 2nd cluster based on 1st
		fprintf(fidKrig,'debug flag: creating 2nd cluster based on the 1st cluster. last cluster Ndatapts = %d\n',metamodel.cluster(lastClusterKrig).Ndatapts);
		% clone + update index
		metamodel.cluster(lastClusterKrig+1) = metamodel.cluster(lastClusterKrig);
		lastClusterKrig = lastClusterKrig+1;
		% retrieve last dataset
		p_site = metamodel.cluster(lastClusterKrig).origS;
		T_site = metamodel.cluster(lastClusterKrig).origY;
		% add p_site + T_site from top
		p_site = [customSortX(new_p_site);p_site];
		T_site = [new_T_site;T_site];
		p_site = customSortX(p_site); % AT
		% reset counter
		metamodel.cluster(lastClusterKrig).Ndatapts = 1;
		[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
		[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
	end
else
	if metamodel.cluster(lastClusterKrig).Ndatapts < Neverycluster 
		fprintf(fidKrig,'debug flag: updating = adding + deleting data points on the %d cluster. Ndatapts = %d\n',lastClusterKrig,metamodel.cluster(lastClusterKrig).Ndatapts);
		% retrieve last dataset
		p_site = metamodel.cluster(lastClusterKrig).origS;
		T_site = metamodel.cluster(lastClusterKrig).origY;
		% add p_site + T_site
		p_site = [customSortX(new_p_site);p_site];
		T_site = [new_T_site;T_site];
		p_site = customSortX(p_site); % AT
		% counter += 1
		metamodel.cluster(lastClusterKrig).Ndatapts = metamodel.cluster(lastClusterKrig).Ndatapts + 1;
		% delete from bottom
		for j = length(p_site):-1:Neverycluster
			p_site(j,:) = [];
			T_site(j,:) = [];
		end
		tic;
		% fit every Nfit
		if mod(metamodel.cluster(lastClusterKrig).Ndatapts , Nfit) == 0
			theta = metamodel.cluster(lastClusterKrig).theta;
			metamodel.cluster(lastClusterKrig).origS = p_site;
			metamodel.cluster(lastClusterKrig).origY = T_site;
			fprintf(fidKrig, 'Optimizing hyper-params...\n');
			[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
			[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
			[dmodel, ~] = dacefit(p_site,T_site,metamodel.cluster(lastClusterKrig).Ndatapts,@regpoly0,@corrgauss,theta,deflob,defupb);
			metamodel.cluster(lastClusterKrig) = dmodel;
		else
			metamodel.cluster(lastClusterKrig).origS = p_site;
			metamodel.cluster(lastClusterKrig).origY = T_site;
			[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
			[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
		end
		toc;
	else
		fprintf(fidKrig,'debug flag: creating cluster %d based on cluster %d. last cluster Ndatapts = %d\n',lastClusterKrig + 1,lastClusterKrig,metamodel.cluster(lastClusterKrig).Ndatapts);
		% clone + update index
		metamodel.cluster(lastClusterKrig+1) = metamodel.cluster(lastClusterKrig);
		lastClusterKrig = lastClusterKrig+1;
		% retrieve last dataset
		p_site = metamodel.cluster(lastClusterKrig).origS;
		T_site = metamodel.cluster(lastClusterKrig).origY;
		% add p_site + T_site from top
		p_site = [customSortX(new_p_site);p_site];
		T_site = [new_T_site;T_site];
		p_site = customSortX(p_site); % AT
		% reset counter
		metamodel.cluster(lastClusterKrig).Ndatapts = 1;
		[norows,nocols] = size(p_site); fprintf(fidKrig,'size(p_site) = [%d,%d]\n',norows,nocols);
		[norows,nocols] = size(T_site); fprintf(fidKrig,'size(T_site) = [%d,%d]\n',norows,nocols);
	end
end


fclose(fidKrig);

end
