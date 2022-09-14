function demo_TPLQT01
% Linear quadratic control acting in multiple frames,
% which is equivalent to a product of Gaussian controllers through a TP-GMM representation
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
% 	publisher="Springer Berlin Heidelberg",
% 	doi="10.1007/s11370-015-0187-9",
% 	year="2016",
% 	volume="9",
% 	number="1",
% 	pages="1--29"
% }
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 3; %Number of Gaussians in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.rfactor = 1E-3;	%Control cost in LQR
model.dt = 0.01; %Time step duration
nbData = 200; %Number of datapoints in a trajectory
nbRepros = 4; %Number of reproductions in new situations
nbStochRepros = 10; %Number of reproductions with stochastic sampling


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD

%Construct Su and Sx matrices (transfer matrices in batch LQR)
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B;
for n=2:nbData
	%Build Sx matrix
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	%Build Su matrix
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M];
end

%Control cost matrix in LQR
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);


%% Create 3rd order tensor data (trajectories in multiple coordinate systems)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Create 3rd order tensor data (trajectories in multiple coordinate systems)...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension DC x P x N, with D=2 the dimension of a
% datapoint, C=2 the number of derivatives (incl. position), P=2 the number of candidate frames, and N=TM 
% the number of datapoints in a trajectory (T=200) multiplied by the number of demonstrations (M=5).
load('data/DataWithDeriv01.mat');

%Compute 3rd order tensor data (trajectories in multiple coordinate systems)
D = (diag(ones(1,nbData-1),-1)-eye(nbData)) / model.dt;
D(end,end) = 0;
Data = zeros(model.nbVar, model.nbFrames, nbSamples*nbData);
for n=1:nbSamples
	s(n).Data = zeros(model.nbVar,model.nbFrames,nbData);
	DataTmp = s(n).Data0;
	for k=1:model.nbDeriv-1
		DataTmp = [DataTmp; s(n).Data0*D^k]; %Compute derivatives
	end
	for m=1:model.nbFrames
		s(n).Data(:,m,:) = s(n).p(m).A \ (DataTmp - repmat(s(n).p(m).b, 1, nbData));
		Data(:,m,(n-1)*nbData+1:n*nbData) = s(n).Data(:,m,:);
	end
end


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of TP-GMM with EM...');
%model = init_tensorGMM_kmeans(Data, model); %Initialization
model = init_tensorGMM_kbins(s, model);
model = EM_tensorGMM(Data, model);

%Precomputation of eigendecompositions and inverses
for m=1:model.nbFrames
	for i=1:model.nbStates
		[V,D] = eig(model.Sigma(1:model.nbVarPos,1:model.nbVarPos,m,i));
		d = diag(D); 
		[~,id] = sort(d,'descend');
		model.V(:,:,m,i) = V(:,id);
		model.D(:,:,m,i) = diag(d(id));
		model.invSigma(:,:,m,i) = inv(model.Sigma(:,:,m,i));
	end
end


%% Reproductions for the same situations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions for the same situations...');
for n=1:nbSamples
	%GMM projection
	for i=1:model.nbStates
		for m=1:model.nbFrames
			s(n).p(m).Mu(:,i) = s(n).p(m).A * model.Mu(:,m,i) + s(n).p(m).b;
			s(n).p(m).invSigma(:,:,i) = s(n).p(m).A * model.invSigma(:,:,m,i) * s(n).p(m).A';
		end
	end
	
	%Compute best path for the n-th demonstration (an HSMM can alternatively be used here)
	[~,s(n).q] = max(model.Pix(:,(n-1)*nbData+1:n*nbData),[],1); 
	
	%Build a reference trajectory for each frame
	Q = zeros(model.nbVar*nbData);
	for m=1:model.nbFrames
		s(n).p(m).MuQ = reshape(s(n).p(m).Mu(:,s(n).q), model.nbVar*nbData, 1);  
		s(n).p(m).Q = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(s(n).p(m).invSigma(:,:,s(n).q), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));
		Q = Q + s(n).p(m).Q;
	end
	
	%Batch LQR (unconstrained linear MPC in multiple frames), corresponding to a product of Gaussian controllers
	Rq = Su' * Q * Su + R;
	X = [s(1).Data0(:,1) + randn(model.nbVarPos,1)*0E0; zeros(model.nbVarPos,1)];
 	rq = zeros(model.nbVar*nbData,1);
	for m=1:model.nbFrames
		rq = rq + s(n).p(m).Q * (s(n).p(m).MuQ - Sx*X);
	end
	rq = Su' * rq; 
 	u = Rq \ rq; %can also be computed with u = lscov(Rq, rq);
	r(n).Data = reshape(Sx*X+Su*u, model.nbVar, nbData);
end


%% Reproductions for new situations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions for new situations...');
for n=1:nbRepros
	%Random generation of new task parameters
	for m=1:model.nbFrames
		id=ceil(rand(2,1)*nbSamples);
		w=rand(2); w=w/sum(w);
		rnew(n).p(m).b = s(id(1)).p(m).b * w(1) + s(id(2)).p(m).b * w(2);
		rnew(n).p(m).A = s(id(1)).p(m).A * w(1) + s(id(2)).p(m).A * w(2);
	end
	
	%GMM projection
	for i=1:model.nbStates
		for m=1:model.nbFrames
			rnew(n).p(m).Mu(:,i) = rnew(n).p(m).A * model.Mu(:,m,i) + rnew(n).p(m).b;
			rnew(n).p(m).invSigma(:,:,i) = rnew(n).p(m).A * model.invSigma(:,:,m,i) * rnew(n).p(m).A';
		end
	end
	
	%Compute best path for the 1st demonstration (an HSMM can alternatively be used here)
	[~,rnew(n).q] = max(model.Pix(:,1:nbData),[],1); 
	
	%Build a reference trajectory for each frame
	Q = zeros(model.nbVar*nbData);
	for m=1:model.nbFrames
		rnew(n).p(m).MuQ = reshape(rnew(n).p(m).Mu(:,rnew(n).q), model.nbVar*nbData, 1);  
		rnew(n).p(m).Q = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(rnew(n).p(m).invSigma(:,:,rnew(n).q), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));
		Q = Q + rnew(n).p(m).Q;
	end
	
	%Batch LQR (unconstrained linear MPC in multiple frames), corresponding to a product of Gaussian controllers
	Rq = Su' * Q * Su + R;
	X = [s(1).Data0(:,1) + randn(model.nbVarPos,1)*0E0; zeros(model.nbVarPos,1)];
 	rq = zeros(model.nbVar*nbData,1);
	for m=1:model.nbFrames
		rq = rq + rnew(n).p(m).Q * (rnew(n).p(m).MuQ - Sx*X);
	end
	rq = Su' * rq; 
 	u = Rq \ rq; %can also be computed with u = lscov(Rq, rq);
	rnew(n).Data = reshape(Sx*X+Su*u, model.nbVar, nbData);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2500,500]);
xx = round(linspace(1,64,nbSamples));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];

%Demonstrations
subplot(1,5,1); hold on; box on; title('Demonstrations');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
	%Plot trajectories
	plot(s(n).Data0(1,1), s(n).Data0(2,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(1,:), s(n).Data0(2,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%Reproductions in same situations
subplot(1,5,2); hold on; box on; title('Reproductions');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
end
for n=1:nbSamples
	%Plot trajectories
	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',12,'color',clrmap(n,:));
	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%Reproductions in new situations 
subplot(1,5,3); hold on; box on; title('Reproductions in new situations');
for n=1:nbRepros
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(rnew(n).p(m), colPegs(m,:));
	end
end
for n=1:nbRepros
	%Plot trajectories
	plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[.2 .2 .2]);
	plot(rnew(n).Data(1,:), rnew(n).Data(2,:),'-','linewidth',1.5,'color',[.2 .2 .2]);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%Model
p0.A = eye(2);
p0.b = zeros(2,1);
for m=1:model.nbFrames
	subplot(1,5,3+m); hold on; grid on; box on; title(['Model - Frame ' num2str(m)]);
	for n=1:nbSamples
		plot(squeeze(Data(1,m,(n-1)*s(1).nbData+1)), squeeze(Data(2,m,(n-1)*s(1).nbData+1)), '.','markersize',15,'color',clrmap(n,:));
		plot(squeeze(Data(1,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), squeeze(Data(2,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), '-','linewidth',1.5,'color',clrmap(n,:));
	end
	plotGMM(squeeze(model.Mu(1:2,m,:)), squeeze(model.Sigma(1:2,1:2,m,:)+eye(2).*1E-2), [.5 .5 .5], .4);
	plotPegs(p0, colPegs(m,:));
	axis equal; axis([-4.5 4.5 -1 8]); set(gca,'xtick',[0],'ytick',[0]);
end

%Timeline plot
figure('position',[10,600,1500,680]); hold on;
qList = s(1).q;
for m=1:model.nbFrames
	labList = {['$x^{(' num2str(m) ')}_1$'], ['$x^{(' num2str(m) ')}_2$'], ['$\dot{x}^{(' num2str(m) ')}_1$'], ['$\dot{x}^{(' num2str(m) ')}_2$']};
	for j=1:model.nbVar
		subplot(model.nbVar, model.nbFrames, (j-1)*model.nbFrames+m); hold on;
		limAxes = [1, nbData, min(Data(j,m,:))-4E0, max(Data(j,m,:))+4E0];
		msh=[]; x0=[];
		for t=1:nbData-1
			if size(msh,2)==0
				msh(:,1) = [t; model.Mu(j,m,qList(t))];
			end
			if t==nbData-1 || qList(t+1)~=qList(t)
				i = qList(t);
				msh(:,2) = [t+1; model.Mu(j,m,i)];
				sTmp = model.Sigma(j,j,m,qList(t))^.5;
				msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
				patch(msh2(1,:), msh2(2,:), [.7 .7 .7],'edgecolor',[.6 .6 .6],'facealpha', .4, 'edgealpha', .4);
				plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.6 .6 .6]);
				if msh(1,1)>1
					plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.5 .5 .5]);
				end
				x0 = [x0 msh];
				msh=[];
			end
		end
		for n=1:nbSamples
			plot(1:nbData, squeeze(s(n).Data(j,m,:)), '-','linewidth',.5,'color',[.6 .6 .6]);
		end
		if j<7
			ylabel(labList{j},'fontsize',14,'interpreter','latex');
		end
		axis(limAxes);
	end
	xlabel('$t$','fontsize',14,'interpreter','latex');
end

pause;
close all;
end

%Function to plot pegs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotPegs(p, colPegs, fa)
	if ~exist('colPegs')
		colPegs = [0.2863    0.0392    0.2392; 0.9137    0.4980    0.0078];
	end
	if ~exist('fa')
		fa = .6;
	end
	pegMesh = [-4 -3.5; -4 10; -1.5 10; -1.5 -1; 1.5 -1; 1.5 10; 4 10; 4 -3.5; -4 -3.5]' *1E-1;
	for m=1:length(p)
		dispMesh = p(m).A(1:2,1:2) * pegMesh + repmat(p(m).b(1:2),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end