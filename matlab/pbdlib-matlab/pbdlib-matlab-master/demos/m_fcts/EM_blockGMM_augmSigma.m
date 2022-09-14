function [model,GAMMA] = EM_blockGMM_augmSigma(Data, model)
% Evaluation of GMM parameters as a single Gaussian parameter estimation problem
% in a hyperdimensional space (with an augmented covariance representation and by replicating 
% the dataset K times and using a weighted estimation of average and covariance with a mask 
% on the data given by the E step of the EM process.
%
% Sylvain Calinon, 2016

%Parameters
nbMaxSteps = 50;
diagRegularizationFactor = 1E-8;
nbData = size(Data,2);

%Augmented data representation
Data = [Data; ones(1,nbData)];

%Augmented covariance representation
model.nbVar = model.nbVar + 1;
Sigma = zeros(model.nbVar, model.nbVar, model.nbStates);
for i=1:model.nbStates
	Sigma(:,:,i) = [model.Sigma(:,:,i)+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1];
end
model.Sigma = Sigma;
model.Mu = zeros(model.nbVar, model.nbStates);

%Initialization of SigmaBlock, MuBlock and DataBlock
model.SigmaBlock = zeros(model.nbStates*model.nbVar);
for i=1:model.nbStates
	model.SigmaBlock((i-1)*model.nbVar+1:i*model.nbVar, (i-1)*model.nbVar+1:i*model.nbVar) = model.Sigma(:,:,i);
end
DataBlock = repmat(Data, model.nbStates, 1);

M = repmat(DataBlock,1,model.nbStates);  %bsxfun() could be used to speed up computation
DataBlk = M .* kron(eye(model.nbStates), ones(model.nbVar,nbData));
%DataBlk = M; %Would there be a way to consider here off-diagonal inter-state correlations?
	
%EM (without loop over each GMM component as in standard EM)
for nbIter=1:nbMaxSteps

	%Compute determinant of SigmaBlock
	U = chol(model.SigmaBlock);
	D = reshape(diag(U), model.nbVar, model.nbStates);
	detSigma = prod(D).^2;
	
	%Compute GAMMA (E-step)
	A = sum((DataBlk'/model.SigmaBlock).*DataBlk',2);
	dst = reshape(A, nbData, model.nbStates);
	h = exp(-0.5*dst) ./ repmat(sqrt((detSigma+realmin)),nbData,1);
	GAMMA = (h ./ repmat(sum(h,2), 1, model.nbStates))';
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	W = diag(reshape(GAMMA2', 1, nbData*model.nbStates));

	%Update Priors (M-step)
	model.Priors = sum(GAMMA,2)/nbData;
	
% 	%Update SigmaBlock without off-diagonal elements (M-step)
% 	model.SigmaBlock = DataBlk * W * DataBlk'; 

	%Update SigmaBlock with off-diagonal elements (M-step)
	%%%model.SigmaBlock2 = M * W * M' / model.nbStates;
	%The version below is faster and also provides off-diagonal elements
	W = repmat( reshape(repmat(GAMMA2.^.5, 1,model.nbVar)', nbData, model.nbVar*model.nbStates)' ,1, model.nbStates);
	model.SigmaBlock = (M.*W) * (M.*W)' / model.nbStates;
	
	%Regularization term
	model.SigmaBlock = model.SigmaBlock + eye(model.nbStates*model.nbVar) * diagRegularizationFactor;
end

%Transform SigmaBlock back to standard mixture representation
for i=1:model.nbStates
	model.Sigma0(:,:,i) = model.SigmaBlock((i-1)*model.nbVar+1:i*model.nbVar, (i-1)*model.nbVar+1:i*model.nbVar);
end

%Transform augmented covariance back to standard normal distribution representation
model.Mu = [];
model.Sigma = [];
for i=1:model.nbStates
	model.Mu(:,i) = model.Sigma0(1:end-1,end,i);
	model.Sigma(:,:,i) = model.Sigma0(1:end-1,1:end-1,i) - model.Mu(:,i)*model.Mu(:,i)';
end
model.nbVar = model.nbVar - 1;

