incmpList = dlmread('log.incomplete.dat');

% ------------------------------ random sampling ------------------------------

for j = 1:length(incmpList)
	i = incmpList(j);
	% system(sprintf('mkdir %s_Iter%d',pumpAssem,i));
	currentDirectory = sprintf('%s_Iter%d',pumpAssem,i);
	cd(currentDirectory);
	system(strcat('copy ..\',sprintf('%s',pumpAssem),'\* .')); % copy inputs from template folder
	system('del g*.dat'); % remove g*.dat file
	x = getGoodInitSamplingPoint(parentPath); % get new sampling points via acquisition function -- write outputs to file
	writeControlPointsFile(x);  % write control points to file
	system('python writeGFile.py'); % gernate new gFile -- SOLVE z and FIX r
	system('..\Imp64bit.exe < dumInp.txt')
	cd(parentPath); % go back to the parent path
end

