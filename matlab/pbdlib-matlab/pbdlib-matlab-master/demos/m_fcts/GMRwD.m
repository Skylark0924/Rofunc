function [expData, expSigma, H, expData_dot] = GMRwD(model, DataIn, in, out, diagRegularizationFactor)
% Gaussian mixture regression (GMR) with additional derivative of the outpur data for one-dimensional input // check for
% higher dimensions
%
% Ajay Tanwani and Sylvain Calinon, 2017

nbData = size(DataIn,2);
nbVarOut = length(out);

if nargin < 5
	diagRegularizationFactor = 1E-8; %Regularization term is optional
end

MuTmp = zeros(nbVarOut,model.nbStates);
expData = zeros(nbVarOut,nbData);
expData_dot = zeros(nbVarOut,nbData);
expSigma = zeros(nbVarOut,nbVarOut,nbData);
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(DataIn(:,t), model.Mu(in,i), model.Sigma(in,in,i));
		
		G_dot(i,t) = -model.Priors(i) * (gaussPDF(DataIn(:,t), model.Mu(in,i), model.Sigma(in,in,i))/model.Sigma(in,in,i))*(DataIn(:,t) - model.Mu(in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	for i=1:model.nbStates		
		H_dot(i,t) = G_dot(i,t) * sum(H(:,t) + realmin) - H(i,t)* sum(G_dot(:,t) + realmin) ;
	end
	
	H_dot(:,t) = H_dot(:,t) / ((sum(H(:,t) + realmin))^2);
	
	%Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i)/(model.Sigma(in,in,i) + eye(length(in))* diagRegularizationFactor) * (DataIn(:,t)-model.Mu(in,i));
		expData(:,t) = expData(:,t) + H(i,t) * MuTmp(:,i);
		expData_dot(:,t) = expData_dot(:,t) + H_dot(i,t) * MuTmp(:,i) + H(i,t) * model.Sigma(out,in,i)/(model.Sigma(in,in,i) + eye(length(in))* diagRegularizationFactor);
	end

	%Compute conditional covariances Method 2
	for i=1:model.nbStates
		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i)/(model.Sigma(in,in,i) + eye(length(in))* diagRegularizationFactor) * model.Sigma(in,out,i);
 		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaTmp + MuTmp(:,i)*MuTmp(:,i)');
	end
	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t)*expData(:,t)'; 
end