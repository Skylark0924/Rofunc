function [model, GAMMA2] = EM_HDGMM(Data, model)
% EM for High Dimensional Data Clustering (HDDC, HD-GMM) model proposed by Bouveyron (2007). 
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
% @article{Bouveyron07,
% 	author = "Bouveyron, C. and Girard, S. and Schmid, C.",
% 	title = "High-dimensional data clustering",
% 	journal = "Computational Statistics and Data Analysis",
% 	year = "2007",
% 	volume = "52",
% 	number = "1",
% 	pages = "502--519"
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


%Parameters of the EM iterations
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,2);

diagRegularizationFactor = 1E-8; %Regularization term is optional

%EM loop
for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[Lik, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2), 1, nbData);
	
	%M-step
	%Update Priors
	model.Priors = sum(GAMMA,2)/nbData;
	
	%Update Mu
	model.Mu = Data * GAMMA2';
	
	%Update factor analyser parameters
	for i=1:model.nbStates
		%Compute covariance
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		S(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVar) * diagRegularizationFactor;

		%HDGMM update
		[V,D] = eig(S(:,:,i)); 
		[~,id] = sort(diag(D),'descend');
% 		model.D(:,:,i) = D(id(1:model.nbFA), id(1:model.nbFA));
% 		model.V(:,:,i) = V(:, id(1:model.nbFA)); 
		d = diag(D);
		model.D(:,:,i) = diag([d(id(1:model.nbFA)); repmat(mean(d(id(model.nbFA+1:end))), model.nbVar-model.nbFA, 1)]);
		model.V(:,:,i) = V(:,id); 
	
		%Reconstruct Sigma
		model.Sigma(:,:,i) = model.V(:,:,i) * model.D(:,:,i) * model.V(:,:,i)' + eye(model.nbVar) * diagRegularizationFactor;
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(Lik,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lik, GAMMA] = computeGamma(Data, model)
Lik = zeros(model.nbStates,size(Data,2));
for i=1:model.nbStates
	Lik(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
end
GAMMA = Lik ./ repmat(sum(Lik,1)+realmin, model.nbStates, 1);
end
