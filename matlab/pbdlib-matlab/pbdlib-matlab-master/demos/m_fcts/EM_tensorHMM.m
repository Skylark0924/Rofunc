function [model, GAMMA] = EM_tensorHMM(s, model)
% Estimation of TP-HMM parameters with an EM algorithm
% Sylvain Calinon, 2015

%Parameters of the EM algorithm
nbMinSteps = 5; %Minimum number of iterations allowed
nbMaxSteps = 50; %Maximum number of iterations allowed
maxDiffLL = 1E-4; %Likelihood increase threshold to stop the algorithm

diagRegularizationFactor = 1E-6; %Regularization term is optional

%Initialization of the parameters
nbSamples = length(s);

Data=[];
for n=1:nbSamples
	Data = cat(3, Data, s(n).Data);
end
nbData = size(Data,3);

for nbIter=1:nbMaxSteps
	fprintf('.');
	
	%E-step
	for n=1:nbSamples
		
		%Emission probabilities
		s(n).B = computeGamma(s(n).Data, model); %See 'computeGamma' function below
		
		%Forward variable ALPHA
		s(n).ALPHA(:,1) = model.StatesPriors .* s(n).B(:,1);
		%Scaling to avoid underflow issues
		s(n).c(1) = 1 / sum(s(n).ALPHA(:,1)+realmin);
		s(n).ALPHA(:,1) = s(n).ALPHA(:,1) * s(n).c(1);
		for t=2:s(n).nbData
			s(n).ALPHA(:,t) = (s(n).ALPHA(:,t-1)'*model.Trans)' .* s(n).B(:,t); 
			%Scaling to avoid underflow issues
			s(n).c(t) = 1 / sum(s(n).ALPHA(:,t)+realmin);
			s(n).ALPHA(:,t) = s(n).ALPHA(:,t) * s(n).c(t);
		end
		
		%Backward variable BETA
		s(n).BETA(:,s(n).nbData) = ones(model.nbStates,1) * s(n).c(end); %Rescaling
		for t=s(n).nbData-1:-1:1
			s(n).BETA(:,t) = model.Trans * (s(n).BETA(:,t+1) .* s(n).B(:,t+1));
			s(n).BETA(:,t) = min(s(n).BETA(:,t) * s(n).c(t), realmax); %Rescaling
		end
		
		%Intermediate variable GAMMA
		s(n).GAMMA = (s(n).ALPHA.*s(n).BETA) ./ repmat(sum(s(n).ALPHA.*s(n).BETA)+realmin, model.nbStates, 1); 
		
		%Intermediate variable XI (fast version, by considering scaling factor)
		for i=1:model.nbStates
			for j=1:model.nbStates
				s(n).XI(i,j,:) = model.Trans(i,j) * (s(n).ALPHA(i,1:end-1) .* s(n).B(j,2:end) .* s(n).BETA(j,2:end)); 
			end
		end
	end
	
	%Concatenation of HMM intermediary variables
	GAMMA=[]; GAMMA_TRK=[]; GAMMA_INIT=[]; XI=[];
	for n=1:nbSamples
		GAMMA = [GAMMA s(n).GAMMA];
		GAMMA_INIT = [GAMMA_INIT s(n).GAMMA(:,1)];
		GAMMA_TRK = [GAMMA_TRK s(n).GAMMA(:,1:end-1)];
		XI = cat(3,XI,s(n).XI);
	end
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, size(GAMMA,2));
	
	%M-step
	for i=1:model.nbStates	
		for m=1:model.nbFrames
			%Matricization/flattening of tensor
			DataMat(:,:) = Data(:,m,:);
			
			%Update Mu
			model.Mu(:,m,i) = DataMat * GAMMA2(i,:)';
			
			%Update Sigma (regularization term is optional)
			DataTmp = DataMat - repmat(model.Mu(:,m,i),1,nbData);
			model.Sigma(:,:,m,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(model.nbVar) * diagRegularizationFactor;
		end
	end
	
	%Update initial state probability vector
	model.StatesPriors = mean(GAMMA_INIT,2); 
	
	%Update transition probabilities
	model.Trans = sum(XI,3)./ repmat(sum(GAMMA_TRK,2)+realmin, 1, model.nbStates); 
	
	%Compute the average log-likelihood through the ALPHA scaling factors
	LL(nbIter)=0;
	for n=1:nbSamples
		LL(nbIter) = LL(nbIter) - sum(log(s(n).c));
	end
	LL(nbIter) = LL(nbIter)/nbSamples;
	%Stop the algorithm if EM converged
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			return;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Lik, GAMMA, GAMMA0] = computeGamma(Data, model)
	nbData = size(Data, 3);
	Lik = ones(model.nbStates, nbData);
	GAMMA0 = zeros(model.nbStates, model.nbFrames, nbData);
	for i=1:model.nbStates
		for m=1:model.nbFrames
			DataMat(:,:) = Data(:,m,:); %Matricization/flattening of tensor
			GAMMA0(i,m,:) = gaussPDF(DataMat, model.Mu(:,m,i), model.Sigma(:,:,m,i));
			Lik(i,:) = Lik(i,:) .* squeeze(GAMMA0(i,m,:))';
		end
		Lik(i,:) = Lik(i,:) * model.Priors(i);
	end
	GAMMA = Lik ./ repmat(sum(Lik,1)+realmin, size(Lik,1), 1);
end