function [model, GAMMA2, LL] = EM_weighted_univariateGMM(Data, w, model)
% Training of a univariate Gaussian mixture model (GMM) with an expectation-maximization (EM) algorithm, 
% where each datapoint Data(1,t) is weighted by a weight w(t).
% Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,2);

w = w(:); %Force weight to be a row vector

diagRegularizationFactor = 1E-8; %Regularization term is optional

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA = GAMMA .* repmat(w',model.nbStates,1);
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		
		%Update Mu
		model.Mu(:,i) = Data * (GAMMA2(i,:))';
		
		%Update Sigma (regularization term is optional)
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVar) * diagRegularizationFactor;
	end
	
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
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
function [L, GAMMA] = computeGamma(Data, model)
	L = zeros(model.nbStates,size(Data,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
end