function model = init_GMM_timeBased(Data, model)
% Initialization of Gaussian Mixture Model (GMM) parameters by clustering 
% the data into equal bins based on the first variable (time steps).
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


%Parameters 
nbVar = size(Data,1);
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
end

TimingSep = linspace(min(Data(1,:)), max(Data(1,:)), model.nbStates+1);

for i=1:model.nbStates
	idtmp = find( Data(1,:)>=TimingSep(i) & Data(1,:)<TimingSep(i+1));
	model.Priors(i) = length(idtmp);
	model.Mu(:,i) = mean(Data(:,idtmp)');
	model.Sigma(:,:,i) = cov(Data(:,idtmp)');
	%Optional regularization term to avoid numerical instability
	model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(nbVar) * model.params_diagRegFact;
end
model.Priors = model.Priors / sum(model.Priors);
