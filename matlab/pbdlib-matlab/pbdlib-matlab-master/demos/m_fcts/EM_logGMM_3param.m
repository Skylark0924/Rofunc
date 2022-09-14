function [model, GAMMA2, LL] = EM_logGMM_3param(Data, model)
% Training of a 3-parameter lognormal mixture model (GMM) with an expectation-maximization (EM) algorithm.
% Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 100; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm
nbData = size(Data,2);

%diagRegularizationFactor = 1E-6; %Regularization term is optional
diagRegularizationFactor = 1E-4; %Regularization term is optional

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	
	%List to find closest point to zero in each cluster 
	[~,id] = max(GAMMA);
	
% 	figure; hold on;
% 	for i=1:model.nbStates
% 		DataTmp = Data(1,id==i);
% 		plot(DataTmp, zeros(size(DataTmp)), '.');
% 		plot(min(DataTmp), 0, 'ko');
% 	end

	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		
		%Update Mu
		Data0 = Data - repmat(model.Gamma(:,i),1,nbData);
		model.Mu(:,i) = log(Data0) * GAMMA2(i,:)';
		
		%Update Sigma
		DataTmp = log(Data0) - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data0,1)) * diagRegularizationFactor;
		
% 		%Update Gamma (by finding closest point to zero in each cluster)
% 		%[~,t] = min((sum(Data(:,id==i).^2,1)).^.5);
% 		DataTmp = min(Data(:,id==i));
% 		model.Gamma(:,i) = DataTmp - 5E-1; %To avoid the inappropriate solution of 3-param lognormal
% 		%plot(model.Gamma(:,i),0, 'kx');
	end
	%model.Gamma
	%pause
	
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
	L = zeros(model.nbStates, size(Data,2));
	for i=1:model.nbStates
		DataTmp = Data - repmat(model.Gamma(:,i),1,size(Data,2));
		id = (DataTmp>0);

		%Removal of smallest entries
		%for nb=1:10
		[~,id0] = min(DataTmp(:,id));
		id(id0)=0;
		%end

		L(i,id) = model.Priors(i) * logGaussPDF(DataTmp(:,id), model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
end