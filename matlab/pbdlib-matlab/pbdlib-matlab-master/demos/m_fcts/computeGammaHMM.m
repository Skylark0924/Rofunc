function [GAMMA, ALPHA] = computeGammaHMM(s, model)
% Compute Gamma probability in HMM by using forward and backward variables.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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

%Initialization of the parameters
nbSamples = length(s);
Data = [];
for n=1:nbSamples
	Data = [Data s(n).Data];
	s(n).nbData = size(s(n).Data,2);
end

for n=1:nbSamples
	%Emission probabilities
	for i=1:model.nbStates
		s(n).B(i,:) = model.Priors(i) * gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
	end
	%Forward variable ALPHA
	s(n).ALPHA(:,1) = model.StatesPriors .* s(n).B(:,1);
	%Scaling to avoid underflow issues
	s(n).c(1) = 1 / sum(s(n).ALPHA(:,1)+realmin);
	s(n).ALPHA(:,1) = s(n).ALPHA(:,1) * s(n).c(1);
	for t=2:s(n).nbData
		s(n).ALPHA(:,t) = (s(n).ALPHA(:,t-1)'*model.Trans)' .* s(n).B(:,t); 
		%Scaling to avoid underflow issues
		s(n).c(t) = 1 / sum(s(n).ALPHA(:,t)+realmin);
		s(n).ALPHA(:,t) = s(n).ALPHA(:,t) * s(n).c(t);
	end
	%Backward variable BETA
	s(n).BETA(:,s(n).nbData) = ones(model.nbStates,1) * s(n).c(end); %Rescaling
	for t=s(n).nbData-1:-1:1
		s(n).BETA(:,t) = model.Trans * (s(n).BETA(:,t+1) .* s(n).B(:,t+1));
		s(n).BETA(:,t) = min(s(n).BETA(:,t) * s(n).c(t), realmax); %Rescaling
	end
	%Intermediate variable GAMMA
	s(n).GAMMA = (s(n).ALPHA.*s(n).BETA) ./ repmat(sum(s(n).ALPHA.*s(n).BETA)+realmin, model.nbStates, 1); 
end

%Concatenation of GAMMAs
GAMMA=[]; ALPHA=[]; 
for n=1:nbSamples
	GAMMA = [GAMMA s(n).GAMMA];
	ALPHA = [ALPHA s(n).ALPHA];
end


