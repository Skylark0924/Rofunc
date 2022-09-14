function demo_Riemannian_SPD_GMM_tensor01
% GMM for covariance data by relying on Riemannian manifold computation with tensor method
%
% If this code is useful for your research, please cite the related publication:
% @article{Jaquier17IROS,
%   author="Jaquier, N. and Calinon, S.",
%   title="Gaussian Mixture Regression on Symmetric Positive Definite Matrices Manifolds: 
%	    Application to Wrist Motion Estimation with s{EMG}",
%   year="2017",
%	  booktitle = "{IEEE/RSJ} Intl. Conf. on Intelligent Robots and Systems ({IROS})",
%	  address = "Vancouver, Canada"
% }
% 
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 30; % Number of datapoints
nbSamples = 1; % Number of demonstrations
nbIter = 5; % Number of iteration for the Gauss Newton algorithm
nbIterEM = 5; % Number of iteration for the EM algorithm

model.nbStates = 3; % Number of states in the GMM
model.nbVar = 2; % Dimension of the manifold and tangent space (2^2 data)
model.params_diagRegFact = 1E-4; % Regularization of covariance
e0 = eye(model.nbVar);

% Initialisation of the covariance transformation
[covOrder4to2, covOrder2to4] = set_covOrder4to2(model.nbVar);

% Tensor regularization term
tensor_diagRegFact_mat = eye(model.nbVar + model.nbVar * (model.nbVar-1)/2);
tensor_diagRegFact = covOrder2to4(tensor_diagRegFact_mat);


%% Generate covariance data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:model.nbStates
	xn = randn(model.nbVar,3);
	S(:,:,i) = cov(xn');
end
x = zeros(model.nbVar, model.nbVar, nbData*nbSamples);

idList = repmat(kron(1:model.nbStates,ones(1,ceil(nbData/model.nbStates))),1,nbSamples);
for t=1:nbData*nbSamples
	xn = randn(model.nbVar,5) * 3E-1;
	x(:,:,t) = S(:,:,idList(t)) + cov(xn');
end

xvec = reshape(x, [model.nbVar^2, nbData*nbSamples]); % Data on the manifold
uvec = logmap_vec(xvec, reshape(e0,[model.nbVar^2,1])); % Data on the tangent space


%% GMM parameters estimation (computation in matrix form)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ucvec = symMat2Vec(logmap(x,e0));
% Initialisation in vector form
model = init_GMM_kbins(ucvec, model, nbSamples);
% Conversion to matrix form
model.MuMan = expmap(vec2symMat(model.Mu), e0);
model.Mu = zeros(size(model.MuMan));
sigma = model.Sigma;
model.Sigma = zeros(model.nbVar,model.nbVar,model.nbVar,model.nbVar,model.nbStates);
for i = 1:model.nbStates
	model.Sigma(:,:,:,:,i) = covOrder2to4(sigma(:,:,i));
end

L = zeros(model.nbStates, nbData*nbSamples);
L2 = zeros(model.nbStates, nbData*nbSamples);

Xts = zeros(model.nbVar, model.nbVar, nbData*nbSamples, model.nbStates);
for nb=1:nbIterEM
	% E-step
	for i=1:model.nbStates
		Xts(:,:,:,i) = logmap(x, model.MuMan(:,:,i));
		
		% L2 and L are equivalent
		L2(i,:) = model.Priors(i) * gaussPDF_tensor(Xts(:,:,:,i), model.Mu(:,:,i), model.Sigma(:,:,:,:,i), covOrder4to2);
		
		xts = symMat2Vec(Xts(:,:,:,i));
		MuVec = symMat2Vec(model.Mu(:,:,i));
		SigmaVec = covOrder4to2(model.Sigma(:,:,:,:,i));
		
		L(i,:) = model.Priors(i) * gaussPDF(xts, MuVec, SigmaVec);
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
	
	% M-step
	for i=1:model.nbStates
		% Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		% Update MuMan
		for n=1:nbIter
			uTmpTot = zeros(model.nbVar,model.nbVar);
			uTmp = logmap(x, model.MuMan(:,:,i));
			for k = 1:nbData*nbSamples
				uTmpTot = uTmpTot + uTmp(:,:,k) .* H(i,k);
			end
			model.MuMan(:,:,i) = expmap(uTmpTot, model.MuMan(:,:,i));
		end
		% Update Sigma
		model.Sigma(:,:,:,:,i) = zeros(model.nbVar,model.nbVar,model.nbVar,model.nbVar);
		for k = 1:nbData*nbSamples
			model.Sigma(:,:,:,:,i) = model.Sigma(:,:,:,:,i) + H(i,k) .* outerprod(uTmp(:,:,k),uTmp(:,:,k));
		end
		model.Sigma(:,:,:,:,i) = model.Sigma(:,:,:,:,i) + tensor_diagRegFact.*model.params_diagRegFact;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Matrix form
figure('PaperPosition',[0 0 8 8],'position',[10,10,1650,1250],'Name', 'Computation in matrix form'); 
hold on; axis off; 
clrmap = lines(model.nbStates);

for t=1:nbData
	plotGMM(zeros(2,1), x(:,:,t), [.6 .6 .6], .05);
end
axis equal;
%print('-dpng','graphs/demo_Riemannian_cov_GMM01a.png');

for i=1:model.nbStates
	plotGMM(zeros(2,1), model.MuMan(:,:,i)*.8, clrmap(i,:), .3);
end
%print('-dpng','graphs/demo_Riemannian_cov_GMM01b.png');

% %Plot activation functions
% figure; hold on;
% clrmap = lines(model.nbStates);
% for i=1:model.nbStates
% 	plot(1:nbData, GAMMA(i,:),'linewidth',2,'color',clrmap(i,:));
% end
% axis([1, nbData, 0, 1.02]);
% set(gca,'xtick',[],'ytick',[]);
% xlabel('t'); ylabel('h_i');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
% Exponential map (SPD manifold)
N = size(U,3);
for n = 1:N
	X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
end
end

function U = logmap(X,S)
% Logarithm map 
N = size(X,3);
for n = 1:N
% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
% 	U(:,:,n) = S * logm(S\X(:,:,n));
	[v,d] = eig(S\X(:,:,n));
	U(:,:,n) = S * v*diag(log(diag(d)))*v^-1;
end
end

function x = expmap_vec(u,s)
% Exponential map for the first vector form (SPD manifold)
	nbData = size(u,2);
	d = size(u,1)^.5;
	U = reshape(u, [d, d, nbData]);
	S = reshape(s, [d, d]);
	x = zeros(d^2, nbData);
	for t=1:nbData
		x(:,t) = reshape(expmap(U(:,:,t),S), [d^2, 1]);
	end
end

function u = logmap_vec(x,s)
% Logarithm map for the first vector form (SPD manifold)
	nbData = size(x,2);
	d = size(x,1)^.5;
	X = reshape(x, [d, d, nbData]);
	S = reshape(s, [d, d]);
	u = zeros(d^2, nbData);
	for t=1:nbData
		u(:,t) = reshape(logmap(X(:,:,t),S), [d^2, 1]);
	end
end

function x = expmap_vec2(u,s)
% Exponential map for the second vector form (SPD manifold)
	U = vec2symMat(u);
	S = vec2symMat(s);
	x = symMat2Vec(expmap(U,S));
end

function u = logmap_vec2(x,s)
% Logarithm map for the second vector form (SPD manifold)
	X = vec2symMat(x);
	S = vec2symMat(s);
	u = symMat2Vec(logmap(X,S));
end

function prob = gaussPDF_tensor(Data, Mu, Sigma, covOrder4to2)
% Likelihood of matrix datapoint(s) to be generated by a Gaussian 
% parameterized by center and 4th-order covariance tensor.
[nbVar, ~, nbData] = size(Data);

% Substract mean
Data = Data - repmat(Mu,1,1,nbData);

% Compute inverse and determinant of covariance
[~, V, D] = covOrder4to2(Sigma);

detSigmaInv = sum(diag(D).^-1);

Sinv = zeros(size(Sigma));
for j = 1:size(V,3)
	Sinv = Sinv + D(j,j)^-1 .* outerprod(V(:,:,j),V(:,:,j));
end

% Compute Gaussian PDF
prob = zeros(nbData,1);

for n = 1:nbData
	prob(n,1) = tensor4o_mult(permute(Data(:,:,n),[3,4,1,2]),tensor4o_mult(Sinv,Data(:,:,n)));
end

prob = exp(-0.5*prob) / sqrt((2*pi)^(nbVar+nbVar*(nbVar-1)/2) * abs(detSigmaInv) + realmin);

end

function v = symMat2Vec(S)
% Reduced vectorisation of a symmetric matrix.
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
% Matricisation of a vector to a symmetric matrix.
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
% Matricisation of a tensor (he rows, respectively columns of the matrix
% are 'rows', respectively 'cols' of the tensor)
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
	
	T = reshape(tensor2mat(A,[1,2]) * tensor2mat(B,[1,2]), [sizeA(1),sizeA(2),sizeB(3),sizeB(4)]);
	
else
	if ismatrix(A) && isscalar(B)
		T = A*B;
	else
		error('Dimensions mismatch.');
	end
end

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