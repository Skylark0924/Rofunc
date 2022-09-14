function [model, GAMMA2, LL] = EM_GMM(Data, model)
% Training of a Gaussian mixture model (GMM) with an expectation-maximization (EM) algorithm.
%
% If this code is useful for your research, please cite the related publication:
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


%Parameters of the EM algorithm
nbData = size(Data,2);
if ~isfield(model,'params_nbMinSteps')
	model.params_nbMinSteps = 5; %Minimum number of iterations allowed
end
if ~isfield(model,'params_nbMaxSteps')
	model.params_nbMaxSteps = 100; %Maximum number of iterations allowed
end
if ~isfield(model,'params_maxDiffLL')
	model.params_maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
end
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-4; %Regularization term is optional
end
if ~isfield(model,'params_updateComp')
	model.params_updateComp = ones(3,1); %pi,Mu,Sigma
end	

for nbIter=1:model.params_nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	
	%M-step
	for i=1:model.nbStates
		%Update Priors
		if model.params_updateComp(1)
			model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		end
		%Update Mu
		if model.params_updateComp(2)
			model.Mu(:,i) = Data * GAMMA2(i,:)';
		end
		%Update Sigma
		if model.params_updateComp(3)
			DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
			model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data,1)) * model.params_diagRegFact;
		end
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>model.params_nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<model.params_maxDiffLL || nbIter==model.params_nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end
disp(['The maximum number of ' num2str(model.params_nbMaxSteps) ' EM iterations has been reached.']);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L, GAMMA] = computeGamma(Data, model)
L = zeros(model.nbStates,size(Data,2));
for i=1:model.nbStates
	L(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
end
GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
end



