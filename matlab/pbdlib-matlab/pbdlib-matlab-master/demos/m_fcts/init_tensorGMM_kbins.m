function [model,bin] = init_tensorGMM_kbins(s, model)
% Initialization based on k-bins, where data are assumed to be sequential 
% and divided in k-bins of equal size. 
%
% Written by Martijn Zeestraten, 2015
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


% Initialize bins
for i=1:model.nbStates
	bin(i).Data = [];
end

% Split each demonstration in K equal bins
for n=1:length(s)
	[nbVar,nbFrames,nbData] = size(s(n).Data);
	Data = reshape(s(n).Data,nbVar*nbFrames,nbData);
	BinSep = round(linspace(1, nbData, model.nbStates+1));
	for i=1:model.nbStates
		bin(i).Data = [bin(i).Data, Data(:,BinSep(i):BinSep(i+1))];		
	end
end

% Calculate statistics on bin data
for i=1:model.nbStates
	bin(i).Mu = mean(bin(i).Data,2);
	bin(i).Sigma = cov(bin(i).Data');
	model.Priors(i) = length(bin(i).Data);
end

% Reshape GMM into a tensor
for m=1:model.nbFrames
	for i=1:model.nbStates
		model.Mu(:,m,i) = bin(i).Mu((m-1)*model.nbVar+1:m*model.nbVar);
		model.Sigma(:,:,m,i) = bin(i).Sigma((m-1)*model.nbVar+1:m*model.nbVar,(m-1)*model.nbVar+1:m*model.nbVar);
	end
end
model.Priors = model.Priors / sum(model.Priors);



