function [model,GAMMA] = EM_blockGMM2(Data, model)
%Evaluation of GMM parameters as a single Gaussian parameter estimation problem
%in a hyperdimensional space (namely, by replicating the dataset K times and using
%a weighted estimation of average and covariance with a mask on the data given by
%the E step of the EM process.
%
%Sylvain Calinon, 2014

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


%EM (without loop over each GMM component as in standard EM)
for nbIter=1:nbMaxSteps

	%Compute determinant
	U = chol(model.SigmaBlock);
	D = reshape(diag(U), model.nbVar, model.nbStates);
	detSigma = prod(D).^2;

	%Compute GAMMA (E-step)
	M = repmat( DataBlock-repmat(model.MuBlock,1,nbData) ,1,model.nbStates);  %bsxfun() could be used to speed up computation
	DataBlk = M .* kron(eye(model.nbStates), ones(model.nbVar,nbData));
	
	A = sum((DataBlk'/model.SigmaBlock).*DataBlk',2);
	%DataBlk'/model.SigmaBlock*DataBlk
	
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
	M = DataBlock - repmat(model.MuBlock,1,nbData);
	W = reshape(repmat(GAMMA2.^.5, 1,model.nbVar)', nbData, model.nbVar*model.nbStates)';
	model.SigmaBlock = (M.*W) * (M.*W)';
	
% 	%Regularization term
% 	model.SigmaBlock = model.SigmaBlock + eye(model.nbStates*model.nbVar) * diagRegularizationFactor;
	
end

figure; hold on;
colormap(flipud(gray));
pcolor(abs(W));
shading flat; axis ij; axis tight;

M = M.*W;
figure; 
subplot(1,2,1); hold on;
plotGMM(zeros(2,1), model.SigmaBlock(1:2,1:2),[1,0,0], .4);
plotGMM(zeros(2,1), model.SigmaBlock(3:4,3:4),[0,1,0], .4);
plotGMM(zeros(2,1), model.SigmaBlock(5:6,5:6),[0,0,1], .4);
plot(M(1,:), M(2,:), '.','color',[1 0 0]);
plot(M(3,:), M(4,:), '.','color',[0 1 0]);
plot(M(5,:), M(6,:), '.','color',[0 0 1]);
plot(0,0,'k+');
axis equal;

subplot(1,2,2); hold on;
plotGMM(zeros(2,1), model.SigmaBlock([1,5],[1,5]),[1,0,1], .4);
plot(M(1,:), M(5,:), '.','color',[1 0 1]);
axis equal;

pause

model.SigmaBlock

% W = [];
% for i=1:model.nbStates
% 	%W = blkdiag(W, diag(GAMMA2(i,:)));
% 	W = [W; kron(ones(1,model.nbStates), diag(GAMMA2(i,:)))]; 
% end
% model.SigmaBlock = DataBlk * W * DataBlk';
	
%model.SigmaBlock = M * W * M'; 

%Transform SigmaBlock back to standard representation
for i=1:model.nbStates
	model.Sigma(:,:,i) = model.SigmaBlock((i-1)*model.nbVar+1:i*model.nbVar, (i-1)*model.nbVar+1:i*model.nbVar);
end


