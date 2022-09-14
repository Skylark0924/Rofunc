function demo_Riemannian_SPD_GMR03
% GMR with vector as input and covariance data as output by relying on Riemannian manifold
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
nbData = 30; % Number of datapoints
nbSamples = 1; % Number of demonstrations
nbIter = 5; % Number of iteration for the Gauss Newton algorithm
nbIterEM = 5; % Number of iteration for the EM algorithm

model.nbStates = 5; % Number of states in the GMM
model.nbVar = 5; % Dimension of the manifold and tangent space (2D input + 2^2 covariance output)
model.nbVarOut = 2; % Dimension of the output 
model.nbVarOutVec = model.nbVarOut + model.nbVarOut*(model.nbVarOut-1)/2; % Dimension of the output in vector form
model.nbVarVec = model.nbVar - model.nbVarOut + model.nbVarOutVec; % Dimension of the manifold and tangent space in vector form
model.dt = 1E-1; % Time step duration
model.params_diagRegFact = 1E-2; % Regularization of covariance

in=1:3; outMat=4:model.nbVar; out = 4:model.nbVarVec;

%% Generate covariance data from rotating covariance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time(1,:) = [1:nbData] * model.dt;

% Vdata = eye(model.nbVar-2);
% Ddata = eye(model.nbVar-2);
Vdata = eye(model.nbVarOut);
Ddata = eye(model.nbVarOut);

X = zeros(model.nbVar,model.nbVar,nbData*nbSamples);
% Input as first eigenvector, output as covariance matrix
for t = 1:nbData
	Ddata(1,1) = t * 1E-1;
	a = pi/2 * t * 1E-1;
	R = [cos(a) -sin(a); sin(a) cos(a)];
	V2 = R * Vdata;
	X(1,1,t) = time(1,t);
%   X(1:2,1:2,t) = diag(V2(:,end));
% 	X(3:4,3:4,t) = V2 * Ddata * V2';
  X(2:3,2:3,t) = diag(V2(:,end));
	X(4:5,4:5,t) = V2 * Ddata * V2';
	x(:,t) = [diag(X(1:3,1:3,t)); symMat2vec(X(4:5,4:5,t))];
end

xIn = x(1:3,:);

%% GMM parameters estimation 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning...');
model = spd_init_GMM_kbins(x, model, nbSamples,out);
model.Mu = zeros(size(model.MuMan));

L = zeros(model.nbStates, nbData*nbSamples);
Xts = zeros(model.nbVar, model.nbVar, nbData*nbSamples, model.nbStates);
for nb=1:nbIterEM
	% E-step
	for i=1:model.nbStates
		xts(in,:,i) = x(in,:)-repmat(model.MuMan(in,i),1,nbData*nbSamples);
		xts(out,:,i) = logmap_vec(x(out,:), model.MuMan(out,i));
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
			uTmpTot = zeros(model.nbVarVec,1);
			uTmp = zeros(model.nbVarVec,nbData*nbSamples);
			uTmp(in,:) = x(in,:) - repmat(model.MuMan(in,i),1,nbData*nbSamples);
			uTmp(out,:) = logmap_vec(x(out,:), model.MuMan(out,i));
			uTmpTot = sum(uTmp.*repmat(H(i,:),model.nbVarVec,1),2);
			
			model.MuMan(in,i) = uTmpTot(in,:) + model.MuMan(in,i);
			model.MuMan(out,i) = expmap_vec(uTmpTot(out,:), model.MuMan(out,i));
		end
		
		% Update Sigma
		model.Sigma(:,:,i) = uTmp * diag(H(i,:)) * uTmp' + eye(model.nbVarVec) .* model.params_diagRegFact;
	end
end

% Eigendecomposition of Sigma
for i=1:model.nbStates
	[V(:,:,i), D(:,:,i)] = eig(model.Sigma(:,:,i));
end


%% GMR (version with single optimization loop)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Regression...');
nbVarOut = model.nbVarOutVec;
uhat = zeros(nbVarOut,nbData);
xhat = zeros(nbVarOut,nbData);
uOut = zeros(nbVarOut,model.nbStates,nbData);
expSigma = zeros(nbVarOut,nbVarOut,nbData);
H = [];
for t=1:nbData
	% Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(xIn(:,t)-model.MuMan(in,i), model.Mu(in,i), model.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	% Compute conditional mean (with covariance transportation)
	if t==1
		[~,id] = max(H(:,t));
		xhat(:,t) = model.MuMan(out,id); % Initial point
	else
		xhat(:,t) = xhat(:,t-1);
	end
	for n=1:nbIter
		uhat(:,t) = zeros(nbVarOut,1);
		for i=1:model.nbStates
			% Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t)
			S1 = vec2symMat(model.MuMan(out,i));
			S2 = vec2symMat(xhat(:,t));
			Ac = blkdiag(eye(model.nbVar-model.nbVarOut),transp(S1,S2));
			
			% Parallel transport of eigenvectors
			for j = 1:size(V,2)
				vMat(:,:,j,i) = blkdiag(diag(V(in,j,i)),vec2symMat(V(out,j,i)));
				pvMat(:,:,j,i) = Ac * D(j,j,i)^.5 * vMat(:,:,j,i) * Ac';
				pV(:,j,i) = [diag(pvMat(in,in,j,i)); symMat2vec(pvMat(outMat,outMat,j,i))];
			end
			
			% Parallel transported sigma (reconstruction from eigenvectors)
			pSigma(:,:,i) = pV(:,:,i)*pV(:,:,i)';
						
			% Gaussian conditioning on the tangent space
			uOut(:,i,t) = logmap_vec(model.MuMan(out,i), xhat(:,t)) + ...
				pSigma(out,in,i)/pSigma(in,in,i)*(xIn(:,t)-model.MuMan(in,i));

			uhat(:,t) = uhat(:,t) + uOut(:,i,t) * H(i,t);
		end
		
		xhat(:,t) = expmap_vec(uhat(:,t), xhat(:,t));
	end
	
	% Compute conditional covariances (note that since uhat=0, the final part in the GMR computation is dropped)
	for i=1:model.nbStates
		SigmaOutTmp = pSigma(out,out,i) - pSigma(out,in,i)/pSigma(in,in,i)*pSigma(in,out,i);
		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaOutTmp + uOut(:,i,t)*uOut(:,i,t)');
	end
end

model.MuMan = real(model.MuMan);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 8],'position',[10,10,1350,650]);
clrmap = lines(model.nbStates);

MuManMatOut = vec2symMat(model.MuMan(out,:));
xhatMat = vec2symMat(xhat);

subplot(2,1,1); hold on; axis off;
sc = 1E1;
for t=1:size(X,3)
	plotGMM(diag(X(in(2:3),in(2:3),t))*sc, X(4:5,4:5,t), [.6 .6 .6], .1);
end
for i=1:model.nbStates
	plotGMM(model.MuMan(in(2:3),i)*sc, MuManMatOut(:,:,i), clrmap(i,:), .3);
end
for t=1:nbData
	plotGMM(diag(X(in(2:3),in(2:3),t))*sc, xhatMat(:,:,t), [0 1 0], .1);
end

subplot(2,1,2); hold on;
for i=1:model.nbStates
	plot(time, H(i,:),'linewidth',2,'color',clrmap(i,:));
end
axis([time(1), time(end), 0, 1.02]);
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