function model = init_tensorGMM_kmeans(Data, model)
% Initialization of TP-GMM with k-means.
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


diagRegularizationFactor = 1E-4; %Optional regularization term

nbVar = size(Data,1);
DataAll = reshape(Data, size(Data,1)*size(Data,2), size(Data,3)); %Matricization/flattening of tensor

%k-means clustering
[Data_id, Mu] = kmeansClustering(DataAll, model.nbStates);

for i=1:model.nbStates
	idtmp = find(Data_id==i);
	model.Priors(i) = length(idtmp);
	Sigma(:,:,i) = cov([DataAll(:,idtmp) DataAll(:,idtmp)]') + eye(size(DataAll,1))*diagRegularizationFactor;
end
model.Priors = model.Priors / sum(model.Priors);

%Reshape GMM parameters into tensor data
for m=1:model.nbFrames
	for i=1:model.nbStates
		model.Mu(:,m,i) = Mu((m-1)*nbVar+1:m*nbVar,i);
		model.Sigma(:,:,m,i) = Sigma((m-1)*nbVar+1:m*nbVar,(m-1)*nbVar+1:m*nbVar,i);
	end
end
