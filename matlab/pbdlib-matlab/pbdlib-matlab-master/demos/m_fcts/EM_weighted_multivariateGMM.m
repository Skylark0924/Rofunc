function [model, GAMMA2, LL] = EM_weighted_multivariateGMM(Data, w, model)
% Training of a multivariate Gaussian mixture model (GMM) with an expectation-maximization (EM) algorithm, 
% where each datapoint Data(:,t) is weighted by a weight w(:,t).
% Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,2);

%w = w(:); %Force weight to be a row vector

diagRegularizationFactor = 1E-8; %Regularization term is optional

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA0] = computeGamma(Data, model); %See 'computeGamma' function below
	for k=1:model.nbVar
		GAMMA(k,:,:) = squeeze(GAMMA0(k,:,:)) .* repmat(w(k,:),model.nbStates,1);
		GAMMA2(k,:,:) = squeeze(GAMMA(k,:,:)) ./ repmat(sum(GAMMA(k,:,:),3)',1,nbData);
	end
	
	
	%M-step
	for i=1:model.nbStates
		%Update Priors
		for k=1:model.nbVar
			model.Priors(k,i) = sum(GAMMA(k,i,:)) / nbData;
		end
		
		%Update Mu
		model.Mu(:,i) = sum(Data.*squeeze(GAMMA2(:,i,:)),2);
		
		%Update Sigma (regularization term is optional)
		DataTmp = (Data-repmat(model.Mu(:,i),1,nbData)) .* squeeze(GAMMA2(:,i,:).^.5);
		model.Sigma(:,:,i) = DataTmp * DataTmp' + eye(model.nbVar) * diagRegularizationFactor;
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
	L = zeros(model.nbVar,model.nbStates,size(Data,2));
	for k=1:model.nbVar
		for i=1:model.nbStates
			L(k,i,:) = model.Priors(k,i) * gaussPDF(Data(k,:), model.Mu(k,i), model.Sigma(k,k,i));
		end
		GAMMA(k,:,:) = squeeze(L(k,:,:)) ./ repmat(sum(squeeze(L(k,:,:)),1)+realmin, model.nbStates,1);
	end
	L = squeeze(prod(L));
end