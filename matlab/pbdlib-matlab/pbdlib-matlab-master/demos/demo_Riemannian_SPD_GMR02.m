function demo_Riemannian_SPD_GMR02
% GMR with time as input and position vector as output, comparison between computation in vector and matrix form
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 200; % Number of datapoints
nbSamples = 5; % Number of demonstrations
nbIter = 1; % Number of iteration for the Gauss Newton algorithm
nbIterEM = 5; % Number of iteration for the EM algorithm
dt = 0.001; % Time step duration


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('./data/2Dletters/G.mat');
x=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	x = [x [[1:nbData]*dt; s(n).Data*1.3E-1]]; 
end
xIn = x(1,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Computation in vector form %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
modelv.nbStates = 4; % Number of states in the GMM
modelv.nbVar = 3; % Dimension of the tangent space (1D input + 2^2 covariance output)
modelv.params_diagRegFact = 1E-4; % Regularization of covariance


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning...');
modelv = init_GMM_kbins(x, modelv,nbSamples);

L = zeros(modelv.nbStates,size(x,2));
for nb=1:nbIterEM	
	% E-step	
	for i=1:modelv.nbStates
		L(i,:) = modelv.Priors(i) * gaussPDF(x, modelv.Mu(:,i), modelv.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, modelv.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	
	% M-step
	for i=1:modelv.nbStates
		% Update Priors
		modelv.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		% Update Mu
		modelv.Mu(:,i) = x * H(i,:)';
		% Update Sigma
		DataTmp = x - repmat(modelv.Mu(:,i),1,nbData*nbSamples);
		modelv.Sigma(:,:,i) = DataTmp * diag(H(i,:)) * DataTmp' + eye(size(x,1)) * modelv.params_diagRegFact;
	end
end


%% GMR (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Regression...');

in=1; out=2:modelv.nbVar;
nbVarOut = length(out);

xhatv = zeros(nbVarOut,nbData);
MuTmp = zeros(nbVarOut,modelv.nbStates);
expSigmav = zeros(nbVarOut,nbVarOut,nbData);
H = [];

for t=1:nbData
	% Compute activation weight
	for i=1:modelv.nbStates
		H(i,t) = modelv.Priors(i) * gaussPDF(xIn(:,t), modelv.Mu(in,i), modelv.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	% Compute conditional means
	for i=1:modelv.nbStates
		MuTmp(:,i) = modelv.Mu(out,i) + modelv.Sigma(out,in,i)/modelv.Sigma(in,in,i) * (xIn(:,t)-modelv.Mu(in,i));
		xhatv(:,t) = xhatv(:,t) + H(i,t) * MuTmp(:,i);
	end
	% Compute conditional covariances
	for i=1:modelv.nbStates
		SigmaTmp = modelv.Sigma(out,out,i) - modelv.Sigma(out,in,i)/modelv.Sigma(in,in,i) * modelv.Sigma(in,out,i);
		expSigmav(:,:,t) = expSigmav(:,:,t) + H(i,t) * (SigmaTmp + MuTmp(:,i)*MuTmp(:,i)');
	end
	expSigmav(:,:,t) = expSigmav(:,:,t) - xhatv(:,t) * xhatv(:,t)'; % + eye(nbVarOut) * modelv.params_diagRegFact; 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Computation in matrix form %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Model parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 4; % Number of states in the GMM
model.nbVar = 3; % Dimension of the manifold and tangent space (1D input + 2^2 output)
model.nbVarCovOut = model.nbVar + model.nbVar*(model.nbVar-1)/2; % Dimension of the output covariance
model.params_diagRegFact = 1E-4; % Regularization of covariance
in = 1; 
out = in(end)+1:model.nbVar;
nbVarIn = length(in); 
nbVarOut = length(out);

% Initialisation of the covariance transformation for input+output
% covariance
[covOrder4to2, covOrder2to4] = set_covOrder4to2(model.nbVar);
% Initialisation of the covariance transformation for output covariance
[covOrder4to2_out, ~] = set_covOrder4to2(model.nbVar-nbVarIn);

% Tensor regularization term
tensor_diagRegFact_mat = eye(model.nbVar + model.nbVar * (model.nbVar-1)/2);
tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);


%% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = zeros(3,3,nbData*nbSamples);
for n = 1:nbData*nbSamples
	X(:,:,n) = diag(x(:,n));
end


%% GMM parameters estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning...');
% Initialisation in vector form
xvec = symMat2Vec(X);
model = init_GMM_kbins(xvec, model, nbSamples);
% Conversion to matrix form
mu = model.Mu;
sigma = model.Sigma;
model.Mu = zeros(model.nbVar,model.nbVar,model.nbStates);
model.Sigma = zeros(model.nbVar,model.nbVar,model.nbVar,model.nbVar,model.nbStates);
for i = 1:model.nbStates
	model.Mu(:,:,i) = vec2symMat(mu(:,i));
	model.Sigma(:,:,:,:,i) = covOrder2to4(sigma(:,:,i));
end

in=1; out=2:model.nbVar;

L = zeros(model.nbStates, nbData*nbSamples);
L2 = zeros(model.nbStates, nbData*nbSamples);
for nb=1:nbIterEM
	
	% E-step
	for i=1:model.nbStates
		
		% Compute probabilities using the reduced form (computationally
		% less expensive than complete form)
		xts = symMat2Vec(X);
		MuVec = symMat2Vec(model.Mu(:,:,i));
		SigmaVec = covOrder4to2(model.Sigma(:,:,:,:,i));

		L(i,:) = model.Priors(i) * gaussPDF2(xts, MuVec, SigmaVec);
% 		L(i,:) = model.Priors(i) * gaussPDF(xts([1:3,6],:), MuVec([1:3,6],:), SigmaVec([1:3,6],[1:3,6]));
		
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
	% M-step
	for i=1:model.nbStates
		% Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		
		% Update MuMan
		for n=1:nbIter
			model.Mu(:,:,i) = zeros(model.nbVar,model.nbVar);
			for k = 1:nbData*nbSamples
				model.Mu(:,:,i) = model.Mu(:,:,i) + X(:,:,k) .* H(i,k);
			end
		end
		
		% Update Sigma
		Xtmp = X - repmat(model.Mu(:,:,i),1,1,nbData*nbSamples);
		model.Sigma(:,:,:,:,i) = zeros(model.nbVar,model.nbVar,model.nbVar,model.nbVar);
		for k = 1:nbData*nbSamples
			model.Sigma(:,:,:,:,i) = model.Sigma(:,:,:,:,i) + H(i,k) .* outerprod(Xtmp(:,:,k),Xtmp(:,:,k));
		end
		model.Sigma(:,:,:,:,i) = model.Sigma(:,:,:,:,i) + tensor_diagRegFact.*model.params_diagRegFact;
	end
end

% Eigendecomposition of Sigma
for i=1:model.nbStates
	[~, V(:,:,:,i), D(:,:,i)] = covOrder4to2(model.Sigma(:,:,:,:,i));
end


%% GMR (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Regression...');

xhat = zeros(nbVarOut,nbVarOut,nbData);
MuTmp = zeros(nbVarOut,nbVarOut,model.nbStates,nbData);
expSigma = zeros(nbVarOut,nbVarOut,nbVarOut,nbVarOut,nbData);
H = [];
for t=1:nbData
	% Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(xIn(:,t), model.Mu(in,in,i), model.Sigma(in,in,in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	% Compute conditional mean
	for i=1:model.nbStates
		% Gaussian conditioning on the tangent space
		MuTmp(:,:,i,t) = model.Mu(out,out,i) + tensor4o_mult(tensor4o_div ...
			(model.Sigma(out,out,in,in,i),model.Sigma(in,in,in,in,i)), (xIn(:,t)-model.Mu(in,in,i)));
% 		MuTmp(:,:,i,t) = model.Mu(out,out,i) + tensor4o_div ...
% 			(model.Sigma(out,out,in,in,i),model.Sigma(in,in,in,in,i)) * (xIn(:,t)-model.Mu(in,in,i));

		xhat(:,:,t) = xhat(:,:,t) + H(i,t) * MuTmp(:,:,i,t);
	end
	
	% Compute conditional covariances 
	for i=1:model.nbStates
		SigmaOutTmp = model.Sigma(out,out,out,out,i) ...
			- tensor4o_mult(tensor4o_div(model.Sigma(out,out,in,in,i), model.Sigma(in,in,in,in,i)), model.Sigma(in,in,out,out,i));
		expSigma(:,:,:,:,t) = expSigma(:,:,:,:,t) + H(i,t) * (SigmaOutTmp + outerprod(MuTmp(:,:,i,t),MuTmp(:,:,i,t)));
	end
	expSigma(:,:,:,:,t) = expSigma(:,:,:,:,t) - outerprod(xhat(:,:,t),xhat(:,:,t));	
end

model.MuVec = symMat2Vec(model.Mu);
for i = 1:model.nbStates
	model.SigmaVec(:,:,i) = covOrder4to2(model.Sigma(:,:,:,:,i));
end
xhatv2 = symMat2Vec(xhat);
xhatv2 = xhatv2(1:2,:);
for t = 1:nbData
	expSigmav2(:,:,t) = covOrder4to2_out(expSigma(:,:,:,:,t));
end
expSigmav2 = expSigmav2(1:2,1:2,:);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,500]); 
% Plot GMM
subplot(1,2,1); hold on; axis off; title('GMM');
plot(x(2,:),x(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(modelv.Mu(2:modelv.nbVar,:), modelv.Sigma(2:modelv.nbVar,2:modelv.nbVar,:), [.8 0 0], .5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% Plot GMR
subplot(1,2,2); hold on; axis off; title('GMR');
plot(x(2,:),x(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(xhatv, expSigmav, [0 .8 0], .05);
plot(xhatv(1,:),xhatv(2,:),'-','linewidth',2,'color',[0 .4 0]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

figure('position',[10,10,1300,500]); 
% Plot GMM
subplot(1,2,1); hold on; axis off; title('GMM');
plot(x(2,:),x(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(model.MuVec(2:modelv.nbVar,:), model.SigmaVec(2:modelv.nbVar,2:modelv.nbVar,:), [.8 0 0], .5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% Plot GMR
subplot(1,2,2); hold on; axis off; title('GMR');
plot(x(2,:),x(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(xhatv2, expSigmav2, [0 .8 0], .05);
plot(xhatv2(1,:),xhatv2(2,:),'-','linewidth',2,'color',[0 .4 0]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function v = symMat2Vec(S)
% Reduced vectorisation of a symmetric matrix
[d,~,N] = size(S);

v = zeros(d+d*(d-1)/2,N);
for n = 1:N
	v(1:d,n) = diag(S(:,:,n));
	
	row = d+1;
	for i = 1:d-1
		v(row:row + d-1-i,n) = sqrt(2).*S(i+1:end,i,n);
		row = row + d-i;
	end
end
end

function S = vec2symMat(v)
% Matricisation of a vector to a symmetric matrix
[t, N] = size(v);

d = (-1 + sqrt(1+8*t))/2;
S = zeros(d,d,N);

for n= 1:N
	% Side elements
	i = d+1;
	for row = 1:d-1
		S(row,row+1:d,n) = v(i:i+d-1-row,n)./sqrt(2);
		i = i+d-row;
	end
	S(:,:,n) = S(:,:,n) + S(:,:,n)';
	% Diagonal elements
	S(:,:,n) = S(:,:,n) + diag(v(1:d,n));
end
end

function M = tensor2mat(T, rows, cols)
% Matricisation of a tensor (the rows, respectively columns of the matrix are 'rows', 
% respectively 'cols' of the tensor)
if nargin <=2
	cols = [];
end

sizeT = size(T);
N = ndims(T);

if isempty(rows)
	rows = 1:N;
	rows(cols) = [];
end
if isempty(cols)
	cols = 1:N;
	cols(rows) = [];
end

if any(rows(:)' ~= 1:length(rows)) || any(cols(:)' ~= length(rows)+(1:length(cols)))
	T = permute(T,[rows(:)' cols(:)']);
end

M = reshape(T,prod(sizeT(rows)), prod(sizeT(cols)));

end

function T = tensor4o_mult(A,B)
% Multiplication of two 4th-order tensors A and B
if ndims(A) == 4 || ndims(B) == 4
	sizeA = size(A);
	sizeB = size(B);
	if ismatrix(A) 
		sizeA(3:4) = [1,1];
	end
	if ismatrix(B)
		sizeB(3:4) = [1,1];
	end

	if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
		error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
	end

% multiplication with outer product, more expensive (loops)	
% 	T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));
% 
% 	for i = 1:sizeA(3)
% 		for j = 1:sizeA(4)
% 			T = T + outerprod(A(:,:,i,j),permute(B(i,j,:,:),[3,4,1,2]));
% 		end
% 	end
	T = reshape(tensor2mat(A,[1,2]) * tensor2mat(B,[1,2]), [sizeA(1),sizeA(2),sizeB(3),sizeB(4)]);
	
else
	if ismatrix(A) && isscalar(B)
		T = A*B;
	else
		error('Dimensions mismatch.');
	end
end
end

function T = tensor4o_div(A,B)
% Division of two 4th-order tensors A and B
if ndims(A) == 4 || ndims(B) == 4
	sizeA = size(A);
	sizeB = size(B);
	if ismatrix(A)
		sizeA(3:4) = [1,1];
	end
	if ismatrix(B)
		T = A/B;
	else
		if sizeA(3) ~= sizeB(1) || sizeA(4) ~= sizeB(2)
			error('Dimensions mismatch: two last dim of A should be the same than two first dim of B.');
		end

		[~, V, D] = covOrder4to2(B);
		invB = zeros(size(B));
		for j = 1:size(V,3)
			invB = invB + D(j,j)^-1 .* outerprod(V(:,:,j),V(:,:,j));
		end
		
% multiplication with outer product, more expensive (loops)
% 		T = zeros(sizeA(1),sizeA(2),sizeB(3),sizeB(4));
% 
% 		for i = 1:sizeA(3)
% 			for j = 1:sizeA(4)
% 				T = T + outerprod(A(:,:,i,j),permute(invB(i,j,:,:),[3,4,1,2]));
% 			end
% 		end

		% multiplication using matricisation of tensors
		T = reshape(tensor2mat(A,[1,2]) * tensor2mat(invB,[1,2]), [sizeA(1),sizeA(2),sizeB(3),sizeB(4)]);
	end
else
	if ismatrix(A) && isscalar(B)
		T = A/B;
	else
		error('Dimensions mismatch.');
	end
end
end

function prob = gaussPDF2(Data, Mu, Sigma)
% Likelihood of datapoint(s) to be generated by a Gaussian parameterized by
% center and covariance. The inverse and determinant of the covariance are
% computed using the eigenvalue decomposition.
[nbVar,nbData] = size(Data);
Data = Data' - repmat(Mu',nbData,1);
[V,D] = eig(Sigma);
SigmaInv = V*diag(diag(D).^-1)*V';
prob = sum((Data*SigmaInv).*Data, 2);
prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * abs(det(Sigma)) + realmin);
end

function [covOrder4to2, covOrder2to4] = set_covOrder4to2(dim)
% Set the factors necessary to simplify a 4th-order covariance of symmetric
% matrix to a 2nd-order covariance. The dimension ofthe 4th-order covariance is 
% dim x dim x dim x dim. Return the functions covOrder4to2 and covOrder2to4. 
% This function must be called one time for each covariance's size.
newDim = dim+dim*(dim-1)/2;

% Computation of the indices and coefficients to transform 4th-order
% covariances to 2nd-order covariances
sizeS = [dim,dim,dim,dim];
sizeSred = [newDim,newDim];
id = [];
idred = [];
coeffs = [];

% left-up part
for k = 1:dim
	for m = 1:dim
		id = [id,sub2ind(sizeS,k,k,m,m)];
		idred = [idred,sub2ind(sizeSred,k,m)];
		coeffs = [coeffs,1];
	end
end

% right-down part
row = dim+1; col = dim+1;
for k = 1:dim-1
	for m = k+1:dim
		for p = 1:dim-1
			for q = p+1:dim
				id = [id,sub2ind(sizeS,k,m,p,q)];
				idred = [idred,sub2ind(sizeSred,row,col)];
				coeffs = [coeffs,2];
				col = col+1;
			end
		end
		row = row + 1;
		col = dim+1;
	end
end

% side-parts
for k = 1:dim
	col = dim+1;
	for p = 1:dim-1
		for q = p+1:dim
			id = [id,sub2ind(sizeS,k,k,p,q)];
			idred = [idred,sub2ind(sizeSred,k,col)];
			id = [id,sub2ind(sizeS,k,k,p,q)];
			idred = [idred,sub2ind(sizeSred,col,k)];
			coeffs = [coeffs,sqrt(2),sqrt(2)];
			col = col+1;
		end
	end
end

% Computation of the indices and coefficients to transform eigenvectors to
% eigentensors
sizeV = [dim,dim,newDim];
sizeVred = [newDim,newDim];
idEig = [];
idredEig = [];
coeffsEig = [];

for n = 1:newDim
	% diagonal part
	for j = 1:dim
		idEig = [idEig,sub2ind(sizeV,j,j,n)];
		idredEig = [idredEig,sub2ind(sizeVred,j,n)];
		coeffsEig = [coeffsEig,1];
	end
	
	% side parts
	j = dim+1;
	for k = 1:dim-1
		for m = k+1:dim
			idEig = [idEig,sub2ind(sizeV,k,m,n)];
			idredEig = [idredEig,sub2ind(sizeVred,j,n)];
			idEig = [idEig,sub2ind(sizeV,m,k,n)];
			idredEig = [idredEig,sub2ind(sizeVred,j,n)];
			coeffsEig = [coeffsEig,1/sqrt(2),1/sqrt(2)];
			j = j+1;
		end
	end
end

function [Sred, V, D] = def_covOrder4to2(S)
	Sred = zeros(newDim,newDim);
	Sred(idred) = bsxfun(@times,S(id),coeffs);
	[v,D] = eig(Sred);
	V = zeros(dim,dim,newDim);
	V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);
end
function [S, V, D] = def_covOrder2to4(Sred) 
	[v,D] = eig(Sred);
	V = zeros(dim,dim,newDim);
	V(idEig) = bsxfun(@times,v(idredEig),coeffsEig);

	S = zeros(dim,dim,dim,dim);
	for i = 1:size(V,3)
		S = S + D(i,i).*outerprod(V(:,:,i),V(:,:,i));
	end
end

covOrder4to2 = @def_covOrder4to2;
covOrder2to4 = @def_covOrder2to4;
end