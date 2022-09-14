function [idList, Mu] = kmeansClustering(Data, nbStates)
% k-means clustering.
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


%Criterion to stop the EM iterative update
cumdist_threshold = 1e-10;
maxIter = 100;

%Initialization of the parameters
[~, nbData] = size(Data);
cumdist_old = -realmax;
nbStep = 0;

idTmp = randperm(nbData);
Mu = Data(:,idTmp(1:nbStates));

%k-means iterations
while 1
	%E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for i=1:nbStates
		%Compute distances
		distTmp(:,i) = sum((Data-repmat(Mu(:,i),1,nbData)).^2, 1);
	end
	[vTmp,idList] = min(distTmp,[],2);
	cumdist = sum(vTmp);
	%M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for i=1:nbStates
		%Update the centers
		Mu(:,i) = mean(Data(:,idList==i),2);
	end
	%Stopping criterion %%%%%%%%%%%%%%%%%%%%
	if abs(cumdist-cumdist_old) < cumdist_threshold
		break;
	end
	cumdist_old = cumdist;
	nbStep = nbStep+1;
	if nbStep>maxIter
		disp(['Maximum number of kmeans iterations, ' num2str(maxIter) 'is reached']);
		break;
	end
end
%disp(['Kmeans stopped after ' num2str(nbStep) ' steps.']);
