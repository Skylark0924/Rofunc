function [model, GAMMA2, LL] = EM_WGMM(Data, model, v)
% EM estimation of a wrapped Gaussian mixture model (WGMM), for partially periodic signals, with periodic dimensions defined in 'v'.
% Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 50; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
[nbVar, nbData] = size(Data);

diagRegularizationFactor = 1E-4; %Regularization term is optional

for j=1:nbVar
	nr(j) = length(v(j).rg);
end

Xtmp = computeHypergrid_WGMM(v, nbData);
DataGrid = repmat(Data, 1, prod(nr)) - Xtmp;

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	L0 = zeros(model.nbStates, nbData*prod(nr));
	L = zeros(model.nbStates, nbData);
	for i=1:model.nbStates
		L0(i,:) = model.Priors(i) * gaussPDF(DataGrid, model.Mu(:,i), model.Sigma(:,:,i));
		L(i,:) = sum(reshape(L0(i,:), nbData, prod(nr)), 2);
		%L(i,:) = gaussPDF_WGMM(Data, model.Mu(:,i), model.Sigma(:,:,i), v);
	end
	GAMMA = L0 ./ repmat(sum(L,1)+realmin, model.nbStates, prod(nr));
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*prod(nr));
	
	%M-step 
	for i=1:model.nbStates
		%Update the prior
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;

		%Update the center
		model.Mu(:,i) = DataGrid * GAMMA2(i,:)';
		
		%Update the covariance
		DataTmp = DataGrid - repmat(model.Mu(:,i), 1, nbData*prod(nr));
		model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(nbVar)*diagRegularizationFactor;
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