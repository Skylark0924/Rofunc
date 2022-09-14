function demo_Riemannian_SPD_GMM_augmSigma01
% GMM to encode ellipsoid datapoints (centers and covariance matrices) by relying on 
% augmented covariance embeddings and Riemannian manifold 
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
nbData = 30; %Number of datapoints
nbSamples = 1; %Number of demonstrations
nbIter = 5; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 5; %Number of iteration for the EM algorithm

model.nbStates = 3; %Number of states in the GMM
model.nbVar = 3; %Dimension of the tangent space
model.nbVarVec = model.nbVar + model.nbVar*(model.nbVar-1)/2; % Dimension in vector form
model.params_diagRegFact = 1E1; %Regularization of covariance


% %% Generate augmented covariance data 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for i=1:model.nbStates
% 	S(:,:,i) = cov(randn(3,model.nbVar-1));
% end
% m = randn(model.nbVar-1, model.nbStates) * 1E0;
% x = zeros(model.nbVar, model.nbVar, nbData*nbSamples);
% xMu = zeros(model.nbVar-1, nbData*nbSamples);
% xSigma = zeros(model.nbVar-1, model.nbVar-1, nbData*nbSamples);
% idList = repmat(kron(1:model.nbStates,ones(1,ceil(nbData/model.nbStates))),1,nbSamples);
% for t=1:nbData*nbSamples
% 	xn = randn(model.nbVar-1,5) * 3E-1;
% 	xMu(:,t) = m(:,idList(t)) + randn(model.nbVar-1,1) * 3E-1;
% 	xSigma(:,:,t) = S(:,:,idList(t)) + cov(xn');
% 	x(:,:,t) = [xSigma(:,:,t) + xMu(:,t)*xMu(:,t)', xMu(:,t); xMu(:,t)', 1];
% end
% xvec = reshape(x, [model.nbVar^2, nbData*nbSamples]);
% uvec = logmap_vec(xvec, reshape(e0,[model.nbVar^2,1]));


%% Generate augmented covariance datapoints from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data [s(n).Data]]; 
end
for t=1:nbData*nbSamples
	xMu(:,t) = Data(:,t);
	xSigma(:,:,t) = eye(model.nbVar-1) * 1E1;
	%xSigma(:,:,t) = Data(:,t) * Data(:,t)' + eye(model.nbVar-1) * 1E0;
	X(:,:,t) = [xSigma(:,:,t) + xMu(:,t)*xMu(:,t)', xMu(:,t); xMu(:,t)', 1];
% 	X(:,:,t) = [xSigma(:,:,t) + xMu(:,t)*xMu(:,t)', xMu(:,t); xMu(:,t)', 1] .* (det(xSigma(:,:,t)).^(-1./(model.nbVar+1)));
end
x = symMat2vec(X);


%% GMM parameters estimation (Mandel notation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = spd_init_GMM_kbins(x, model, nbSamples);
model.Mu = zeros(size(model.MuMan));

L = zeros(model.nbStates, nbData*nbSamples);
xts = zeros(model.nbVarVec, nbData*nbSamples, model.nbStates);
for nb=1:nbIterEM
	% E-step
	for i=1:model.nbStates
		xts(:,:,i) = logmap_vec(x, model.MuMan(:,i));
		L(i,:) = model.Priors(i) * gaussPDF(xts(:,:,i), model.Mu(:,i), model.Sigma(:,:,i));
		
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
	
	% M-step
	for i=1:model.nbStates
		% Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		
		% Update MuMan
		for n=1:nbIter
			uTmp = logmap_vec(x, model.MuMan(:,i));
			uTmpTot = sum(uTmp.*repmat(H(i,:),model.nbVarVec,1),2);
			
			model.MuMan(:,i) = expmap_vec(uTmpTot, model.MuMan(:,i));
		end
		
		% Update Sigma
		model.Sigma(:,:,i) = uTmp * diag(H(i,:)) * uTmp' + eye(model.nbVarVec) .* model.params_diagRegFact;
	end
end

%Convert back augmented covariances to Gaussians
Mu = zeros(model.nbVar-1, model.nbStates);
Sigma = zeros(model.nbVar-1, model.nbVar-1, model.nbStates);
MuMan = vec2symMat(model.MuMan);
for i=1:model.nbStates
	%Mu(:,i) = MuMan(1:end-1,end,i);
	%Sigma(:,:,i) = MuMan(1:end-1,1:end-1,i) - Mu(:,i) * Mu(:,i)';
	beta = MuMan(end,end,i);
	Mu(:,i) = MuMan(end,1:end-1,i) ./ beta;
	Sigma(:,:,i) = MuMan(1:end-1,1:end-1,i) - beta * Mu(:,i) * Mu(:,i)';
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; 
clrmap = lines(model.nbStates);

plotGMM(xMu, xSigma, [.6 .6 .6], .05);
for i=1:model.nbStates
	plotGMM(Mu(:,i), Sigma(:,:,i)*.8, clrmap(i,:), .3);
end
axis equal;
%print('-dpng','graphs/demo_Riemannian_cov_GMM_augmSigma01.png');

%Plot activation function
figure; hold on;
clrmap = lines(model.nbStates);
for i=1:model.nbStates
	plot(1:nbData, GAMMA(i,:),'linewidth',2,'color',clrmap(i,:));
end
axis([1, nbData, 0, 1.02]);
set(gca,'xtick',[],'ytick',[]);
xlabel('t'); ylabel('h_i');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
% Exponential map (SPD manifold)
N = size(U,3);
for n = 1:N
% 	X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
	[v,d] = eig(S\U(:,:,n));
	X(:,:,n) = S * v*diag(exp(diag(d)))*v^-1;
end
end

function x = expmap_vec(u,s)
% Exponential map (SPD manifold)
U = vec2symMat(u);
S = vec2symMat(s);
X = expmap(U,S);
x = symMat2vec(X);
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

function u = logmap_vec(x,s)
% Exponential map (SPD manifold)
X = vec2symMat(x);
S = vec2symMat(s);
U = logmap(X,S);
u = symMat2vec(U);
end

function Ac = transp(S1,S2)
% Parallel transport (SPD manifold)
% t = 1;
% U = logmap(S2,S1);
% Ac = S1^.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^-.5;
Ac = (S2/S1)^.5;
end

function M = spdMean(setS, nbIt)
% Mean of SPD matrices on the manifold
if nargin == 1
	nbIt = 10;
end
M = setS(:,:,1);

for i=1:nbIt
	L = zeros(size(setS,1),size(setS,2));
	for n = 1:size(setS,3)
		L = L + logm(M^-.5 * setS(:,:,n)* M^-.5);
	end
	M = M^.5 * expm(L./size(setS,3)) * M^.5;
end

end

function model = spd_init_GMM_kbins(Data, model, nbSamples, spdDataId)
% K-Bins initialisation by relying on SPD manifold
nbData = size(Data,2) / nbSamples;
if ~isfield(model,'params_diagRegFact')
	model.params_diagRegFact = 1E-4; %Optional regularization term to avoid numerical instability
end

% Delimit the cluster bins for the first demonstration
tSep = round(linspace(0, nbData, model.nbStates+1));

% Compute statistics for each bin
for i=1:model.nbStates
	id=[];
	for n=1:nbSamples
		id = [id (n-1)*nbData+[tSep(i)+1:tSep(i+1)]];
	end
	model.Priors(i) = length(id);
	
	% Mean computed on SPD manifold for parts of the data belonging to the
	% manifold
	if nargin < 4
		model.MuMan(:,i) = symMat2vec(spdMean(vec2symMat(Data(:,id))));
	else
		model.MuMan(:,i) = mean(Data(:,id),2);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				model.MuMan(spdDataId{c},i) = symMat2vec(spdMean(vec2symMat(Data(spdDataId{c},id)),3));
			end
		else
			model.MuMan(spdDataId,i) = symMat2vec(spdMean(vec2symMat(Data(spdDataId,id)),3));
		end
	end
	
	% Parts of data belonging to SPD manifold projected to tangent space at
	% the mean to compute the covariance tensor in the tangent space
	DataTgt = zeros(size(Data(:,id)));
	if nargin < 4
		DataTgt = logmap_vec(Data(:,id),model.MuMan(:,i));
	else
		DataTgt = Data(:,id);
		if iscell(spdDataId)
			for c = 1:length(spdDataId)
				DataTgt(spdDataId{c},:) = logmap_vec(Data(spdDataId{c},id),model.MuMan(spdDataId{c},i));
			end
		else
			DataTgt(spdDataId,:) = logmap_vec(Data(spdDataId,id),model.MuMan(spdDataId,i));
		end
	end

	model.Sigma(:,:,i) = cov(DataTgt') + eye(model.nbVarVec).*model.params_diagRegFact;
	
end
model.Priors = model.Priors / sum(model.Priors);
end

function V = symMat2vec(S)
% Vectorization of a tensor of symmetric matrix

[D, ~, N] = size(S);

V = [];
for n = 1:N
	v = [];
	v = diag(S(:,:,n));
	for d = 1:D-1
	  v = [v; sqrt(2).*diag(S(:,:,n),d)]; % Mandel notation
	%   v = [v; diag(M,n)]; % Voigt notation
	end
	V = [V v];
end

end

function S = vec2symMat(V)
% Transforms matrix of vectors to tensor of symmetric matrices

[d, N] = size(V);
D = (-1 + sqrt(1 + 8*d))/2;
for n = 1:N
	v = V(:,n);
	M = diag(v(1:D));
	id = cumsum(fliplr(1:D));

	for i = 1:D-1
	  M = M + diag(v(id(i)+1:id(i+1)),i)./sqrt(2) + diag(v(id(i)+1:id(i+1)),-i)./sqrt(2); % Mandel notation
	%   M = M + diag(v(id(i+1)+1:id(i+1)),i) + diag(v(id(i+1)+1:id(i+1)),-i); % Voigt notation
	end
	S(:,:,n) = M;
end
end