function model = init_GMM_kbins2(Data, model, indices)
% Initialization of Gaussian Mixture Model (GMM) parameters by clustering 
% an arbitrary dataset into equal bins.
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
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
% Written by Andras Kupcsik, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distrib uted in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


%Parameters 
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
end

indices = [0, indices];

for i = 1:model.nbStates
    data{i} = [];
end

%Compute statistics for each bin
for d = 1:length(indices)-1
    id = indices(d):indices(d+1);
    idLoc = round(linspace(1, length(id), model.nbStates+1));
    for i=1:model.nbStates       
        data{i} = [data{i}, Data(:, id(idLoc(i)+1:idLoc(i+1)))];
%         id(idLoc(i)+1:idLoc(i+1))
    end
    
    
end

for i = 1:model.nbStates
   
	model.Mu(:,i) = mean(data{i},2);
    tmpMat = data{i};
	model.Sigma(:,:,i) = cov(tmpMat') + eye(size(Data,1)) * model.params_diagRegFact;
end
model.Priors = ones(1, model.nbStates)/model.nbStates;


