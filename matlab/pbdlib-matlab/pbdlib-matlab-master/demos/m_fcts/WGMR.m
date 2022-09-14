function [expData, expSigma, H] = WGMR(model, DataIn, vIn, in, out)
% Wrapped Gaussian mixture regression (WGMR)
% Sylvain Calinon, 2015

nbData = size(DataIn,2);
nbVarIn = length(in);
nbVarOut = length(out);

diagRegularizationFactor = 1E-8; %Regularization term is optional

for j=1:nbVarIn
	nr(j) = length(vIn(j).rg);
end

expData = zeros(nbVarOut,nbData);
expSigma = zeros(nbVarOut,nbVarOut,nbData);
for t=1:nbData
	
	Xtmp = computeHypergrid_WGMM(vIn, 1);
	DataGrid = repmat(DataIn(:,t), 1, prod(nr)) - Xtmp; 
	
	%Compute activation weight
	H0 = zeros(model.nbStates, prod(nr));
	H = zeros(model.nbStates, nbData);
	for i=1:model.nbStates
		H0(i,:) = model.Priors(i) * gaussPDF(DataGrid, model.Mu(in,i), model.Sigma(in,in,i));
		H(i,t) = sum(H0(i,:));
	end
	H0 = H0 / sum(H(:,t)+realmin);
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	%Compute conditional means
	for i=1:model.nbStates	
		MuTmp(:,:,i) = (repmat(model.Mu(out,i), 1, prod(nr)) + model.Sigma(out,in,i)/model.Sigma(in,in,i) * ...
			(DataGrid - repmat(model.Mu(in,i), 1, prod(nr))));	
		%for m=1:prod(nr)
		%	expData(:,t) = expData(:,t) + MuTmp(:,m,i) * H0(i,m);
		%end
		expData(:,t) = expData(:,t) + MuTmp(:,:,i) * H0(i,:)';
	end
	
	%Compute conditional covariances
	for i=1:model.nbStates
		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i)/model.Sigma(in,in,i) * model.Sigma(in,out,i);
		%for m=1:prod(nr)
		%	expSigma(:,:,t) = expSigma(:,:,t) + (SigmaTmp + MuTmp(:,m,i) * MuTmp(:,m,i)') * H0(i,m);
		%end
		expSigma(:,:,t) = expSigma(:,:,t) + SigmaTmp * sum(H0(i,:)) + MuTmp(:,:,i) * diag(H0(i,:)) * MuTmp(:,:,i)';
	end
	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t)*expData(:,t)' + eye(nbVarOut)*diagRegularizationFactor;
end	