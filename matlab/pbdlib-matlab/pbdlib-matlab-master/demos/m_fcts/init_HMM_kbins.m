function [model,bin] = init_HMM_kbins(sData, model)
% Initialization based on k-bins. 
% Data is assumed to be sequential and
% divided in k-bins of equal size. 
% Inputs:
% s : Structure with data of each demonstration
%
% Authors:	Martijn Zeestraten, 2015
%         http://programming-by-demonstration.org/

% Initialize bins:
for s= 1:model.nbStates
	bin(s).Data = [];
end

% Split each demonstration in K equal bins:
for dem = 1:length(sData)
	[~,nbData] = size(sData(dem).Data);
	Data = sData(dem).Data;
	BinSep = round(linspace(1, nbData, model.nbStates+1));
	
	for s = 1:model.nbStates
		bin(s).Data = [bin(s).Data, Data(:,BinSep(s):BinSep(s+1))];		
	end
end

% Calculate statistics on bin data:
model.Mu    = zeros(model.nbVar,model.nbStates);
model.Sigma = zeros(model.nbVar,model.nbVar,model.nbStates);
for s = 1:model.nbStates
	model.Mu(:,s) = mean(bin(s).Data,2);
	model.Sigma(:,:,s) = cov(bin(s).Data');
	model.Priors(s) = length(bin(s).Data);
end
model.Priors = model.Priors / sum(model.Priors);

end  % end function


