function [model,GAMMA] = EM_blockMFA(Data, model)
%Evaluation of MFA parameters as a single Gaussian parameter estimation problem
%in a hyperdimensional space (namely, by replicating the dataset K times and using
%a weighted estimation of average and covariance with a mask on the data given by
%the E step of the EM process.
%
%Sylvain Calinon, 2015

%Thresholds for the EM iterations
nbMaxSteps = 50;
nbMinSteps = 2;
maxDiffLL = 1E-5;

diagRegularizationFactor = 1E-8;

nbData = size(Data,2);

%Initialization of SigmaBlock, MuBlock and DataBlock
model.SigmaBlock = zeros(model.nbStates*model.nbVar);
for i=1:model.nbStates
	model.SigmaBlock((i-1)*model.nbVar+1:i*model.nbVar, (i-1)*model.nbVar+1:i*model.nbVar) = model.Sigma(:,:,i);
end
model.MuBlock = reshape(model.Mu, model.nbStates*model.nbVar, 1);
DataBlock = repmat(Data, model.nbStates, 1);

%Initialization of the MFA parameters from eigendecomposition estimate
S = model.SigmaBlock;
[V,D] = eig(S);
[~,id] = sort(diag(D),'descend');
model.L = V(:,id(1:model.nbFA));
model.P = diag(diag(S));
for nbIter=1:nbMaxSteps
	for i=1:model.nbStates
		%Update B,L,P
		B = model.L' / (model.L * model.L' + model.P);
		model.L = S * B' / (eye(model.nbFA) - B * model.L + B * S * B');
		model.P = diag(diag(S - model.L * B * S)) + eye(model.nbVar*model.nbStates) * diagRegularizationFactor;
	end
end

%EM (without loops for each component of the GMM)
for nbIter=1:nbMaxSteps

	%Compute determinant
	U = chol(model.SigmaBlock);
	D = reshape(diag(U), model.nbVar, model.nbStates);
	detSigma = prod(D).^2;

	%Compute GAMMA (E-step)
	M = repmat( DataBlock-repmat(model.MuBlock,1,nbData) ,1,model.nbStates);  %bsxfun() could be used to speed up computation
	DataBlk = M .* kron(eye(model.nbStates), ones(model.nbVar,nbData));
	%DataBlk = M; %Could we consider here off-diagonal inter-state correlations?
	A = sum((DataBlk'/model.SigmaBlock).*DataBlk',2);
	dst = reshape(A, nbData, model.nbStates);
	h = exp(-0.5*dst) ./ repmat(sqrt((detSigma+realmin)),nbData,1);
	GAMMA = (h ./ repmat(sum(h,2), 1, model.nbStates))';
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);

	%Update Priors (M-step)
	model.Priors = sum(GAMMA,2)/nbData;

	%Update Mu (M-step)
	model.Mu = Data * GAMMA2';
	model.MuBlock = reshape(model.Mu, model.nbStates*model.nbVar, 1);

	%Update Sigma (M-step)
	%W = diag(reshape(GAMMA2',1,nbData*nbStates));
	%SigmaBlock = DataBlk * W * DataBlk'; %Update with no off-diagonal elements
	%The version below is faster and provides off-diagonal elements
	W = repmat( reshape(repmat(GAMMA2.^.5, 1,model.nbVar)', nbData, model.nbVar*model.nbStates)' ,1, model.nbStates);
	S = (M.*W) * (M.*W)' / model.nbStates;
	
	%Regularization term
	S = S + eye(model.nbStates*model.nbVar) * diagRegularizationFactor;
	
	%Update B
	B = model.L' / (model.L * model.L' + model.P);
	%Update Lambda
	model.L = S * B' / (eye(model.nbFA) - B * model.L + B * S * B');
	%Update Psi
	model.P = diag(diag(S - model.L * B * S)) + eye(model.nbVar*model.nbStates) * diagRegularizationFactor;
	
	%Reconstruct SigmaBlock
	model.SigmaBlock = model.L * model.L' + model.P;
end

%Transform SigmaBlock back to standard representation
for i=1:model.nbStates
	model.Sigma(:,:,i) = model.SigmaBlock((i-1)*model.nbVar+1:i*model.nbVar, (i-1)*model.nbVar+1:i*model.nbVar);
end


