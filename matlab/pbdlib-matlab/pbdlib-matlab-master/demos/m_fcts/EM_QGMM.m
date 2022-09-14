function [model, GAMMA2, LL] = EM_QGMM(Data, model)
% Training of a quantum Gaussian mixture model (QGMM) with an expectation-maximization (EM) algorithm.
% Sylvain Calinon, 2017

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
	
	P = computePosterior(Data, model);
	W = zeros(model.nbStates, model.nbStates, size(Data,2));
	%lPriors = logm(model.Priors);
	[V,D] = eig(model.Priors);
	lPriors = V * diag(log(diag(D+realmin))) * V';	
	for t=1:nbData
		%lP = logm(P(:,:,t));
		[V,D] = eig(P(:,:,t));
		lP = V * diag(log(diag(D+realmin))) * V';
		%W(:,:,t) = expm(lPriors + lP);
		[V,D] = eig(lPriors + lP);
		W(:,:,t) = V * diag(exp(diag(D+realmin))) * V';
		W(:,:,t) = W(:,:,t) / trace(W(:,:,t));
	end
	
	%M-step
	%Update Priors
	if model.params_updateComp(1)
		model.Priors = sum(W,3) / nbData;
	end
	for i=1:model.nbStates
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
	L = zeros(model.nbStates, size(Data,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i,i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = computePosterior(Data, model)
	P = zeros(model.nbStates, model.nbStates, size(Data,2));
	for i=1:model.nbStates
		P(i,i,:) = gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
	end
end



