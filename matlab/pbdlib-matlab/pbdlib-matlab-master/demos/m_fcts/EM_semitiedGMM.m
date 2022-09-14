function [model, LL] = EM_semitiedGMM(Data, model)
% EM procedure to estimate parameters of a semi-tied Gaussian Mixture Model
%
% Writing code takes time. Polishing it and making it available to others takes longer!
% If some parts of the code were useful for your research of for a better understanding
% of the algorithms, please reward the authors by citing the related publications,
% and consider making your own research available in this way.
%
% @article{Tanwani16RAL,
%   author="Tanwani, A. K. and Calinon, S.",
%   title="Learning Robot Manipulation Tasks with Task-Parameterized Semi-Tied Hidden Semi-{M}arkov Model",
%   journal="{IEEE} Robotics and Automation Letters ({RA-L})",
%   year="2016",
%   month="January",
%   volume="1",
%   number="1",
%   pages="235--242",
%   doi="10.1109/LRA.2016.2517825"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Ajay Tanwani and Sylvain Calinon
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
if ~isfield(model,'params_Bsf')
	model.params_Bsf = 1E-1;
end
if ~isfield(model,'params_nbVariationSteps')
	model.params_nbVariationSteps = 50;
end

if ~isfield(model,'B')
	model.B = eye(model.nbVar) * model.params_Bsf;
	model.InitH = pinv(model.B) + eye(model.nbVar) * model.params_diagRegFact;
	for i=1:model.nbStates
		%model.InitSigmaDiag(:,:,i) = diag(diag(model.B*model.Sigma(:,:,i)*model.B'));
		[~,model.InitSigmaDiag(:,:,i)] = eig(model.Sigma(:,:,i));
	end
end

for nbIter=1:model.params_nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);

	%M-step 
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		%Update Mu
		model.Mu(:,i) = Data * GAMMA2(i,:)';
		%Compute sample covariance
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.S(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVar) * model.params_diagRegFact;
	end
	%Update SigmaDiag and compute B
	for lp=1:model.params_nbVariationSteps
		for i=1:model.nbStates
			model.SigmaDiag(:,:,i) = diag(diag(model.B * model.S(:,:,i) * model.B')); %Eq.(9)
		end
		for k=1:model.nbVar
			C = pinv(model.B') * det(model.B); %Or C=cof(model.B), Eq.(6)
			G = zeros(model.nbVar);
			for i=1:model.nbStates
				G = G + model.S(:,:,i) * sum(GAMMA(i,:),2) / model.SigmaDiag(k,k,i); %Eq.(7)
			end
			model.B(k,:) = C(k,:) * pinv(G) * (sqrt(sum(sum(GAMMA,2) / (C(k,:) * pinv(G) * C(k,:)')))); %Eq.(5)
		end
	end
	%Update H and compute Sigma
	model.H = pinv(model.B) + eye(model.nbVar) * model.params_diagRegFact;
	for i=1:model.nbStates
		model.Sigma(:,:,i) = model.H * model.SigmaDiag(:,:,i) * model.H'; %Eq.(3)
	end

	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>model.params_nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<model.params_maxDiffLL || nbIter==model.params_nbMaxSteps-1
			%figure; plot(LL); title('Likelihood Plot'); xlabel('Iterations'); ylabel('log-likelihood')
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
