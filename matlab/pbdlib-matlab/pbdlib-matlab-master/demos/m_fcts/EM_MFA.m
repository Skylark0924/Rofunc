function [model, GAMMA2] = EM_MFA(Data, model)
% EM for Mixture of factor analysis (implementation based on "Parsimonious Gaussian 
% Mixture Models" by McNicholas and Murphy, Appendix 8, p.17, UUU version).
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
	model.params_diagRegFact = 1E-6; %Regularization term is optional
end
if isfield(model,'params_updateComp')==0
	model.params_updateComp = ones(3,1);
end	


% %Circular initialization of the MFA parameters
% Itmp = eye(size(Data,1))*1E-2;
% model.P = repmat(Itmp, [1 1 model.nbStates]);
% model.L = repmat(Itmp(:,1:model.nbFA), [1 1 model.nbStates]);

%Initialization of the MFA parameters from eigendecomposition estimate
for i=1:model.nbStates
	model.P(:,:,i) = diag(diag(model.Sigma(:,:,i))); %Dimension-wise variance
	[V,D] = eig(model.Sigma(:,:,i)-model.P(:,:,i)); 
	[~,id] = sort(diag(D),'descend');
	V = V(:,id)*D(id,id).^.5;
	model.L(:,:,i) = V(:,1:model.nbFA); %-> Sigma=LL'+P
end
for nbIter=1:model.params_nbMaxSteps
	for i=1:model.nbStates
		%Update B,L,P
		B(:,:,i) = model.L(:,:,i)' / (model.L(:,:,i) * model.L(:,:,i)' + model.P(:,:,i));
		model.L(:,:,i) = model.Sigma(:,:,i) * B(:,:,i)' / (eye(model.nbFA) - B(:,:,i) * model.L(:,:,i) + B(:,:,i) * model.Sigma(:,:,i) * B(:,:,i)');
		model.P(:,:,i) = diag(diag(model.Sigma(:,:,i) - model.L(:,:,i) * B(:,:,i) * model.Sigma(:,:,i)));
	end
end

%EM loop
for nbIter=1:model.params_nbMaxSteps
	fprintf('.');
	%E-step 
	[Lik, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	
	%M-step (cycle 1)
	%Update Priors
	if model.params_updateComp(1)
		model.Priors = sum(GAMMA,2) / nbData;
	end
	
	%Update Mu
	if model.params_updateComp(2)
		model.Mu = Data * GAMMA2';
	end
	
	%M-step (cycle 2)
	%Update factor analysers parameters
	if model.params_updateComp(3)
		for i=1:model.nbStates
			%Compute covariance
			DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
			S(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data,1)) * model.params_diagRegFact;

			%Update B
			B(:,:,i) = model.L(:,:,i)' / (model.L(:,:,i) * model.L(:,:,i)' + model.P(:,:,i));
			%Update Lambda
			model.L(:,:,i) = S(:,:,i) * B(:,:,i)' / (eye(model.nbFA) - B(:,:,i) * model.L(:,:,i) + B(:,:,i) * S(:,:,i) * B(:,:,i)');
			%Update Psi
			model.P(:,:,i) = diag(diag(S(:,:,i) - model.L(:,:,i) * B(:,:,i) * S(:,:,i))) + eye(size(Data,1)) * model.params_diagRegFact;

			%Reconstruct Sigma
			model.Sigma(:,:,i) = real(model.L(:,:,i) * model.L(:,:,i)' + model.P(:,:,i));
		end
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(Lik,1))) / nbData;
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
function [Lik, GAMMA] = computeGamma(Data, model)
Lik = zeros(model.nbStates,size(Data,2));
for i=1:model.nbStates
	Lik(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
end
GAMMA = Lik ./ repmat(sum(Lik,1)+realmin, model.nbStates, 1);
end
