function demo_OC_LQT_infHor03
% Continuous infinite horizon linear quadratic tracking, by relying on a GMM encoding of position and velocity data.
% (see also demo_MPC_infHor02.m for the corresponding discrete version, recommended for stability)
%
% If this code is useful for your research, please cite the related publication:
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
% Written by Sylvain Calinon (http://calinon.ch/) and Danilo Bruno (danilo.bruno@iit.it)
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
model.nbStates = 6; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step duration
model.rfactor = 1E-3;	%Control cost in LQR (to be set carefully because infinite horizon LQR can suffer mumerical instability)

nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions
nbData = 200; %Number of datapoints

%Continuous dynamical System settings (Integration with Euler method) 
A = kron(diag(ones(model.nbDeriv-1,1),1), eye(model.nbVarPos)); 
B = kron([zeros(model.nbDeriv-1,1); 1], eye(model.nbVarPos));

%A = kron([0 1; 0 0], eye(model.nbVarPos));
%B = kron([0; 1], eye(model.nbVarPos));
%C = kron([1, 0], eye(model.nbVarPos));

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data=[];
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).Data = [s(n).Data; dTmp];
	end
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data,model);
model = init_GMM_kbins(Data,model,nbSamples);

% %Initialization based on position data
% model0 = init_GMM_kmeans(Data(1:model.nbVarPos,:), model);
% [~, GAMMA2] = EM_GMM(Data(1:model.nbVarPos,:), model0);
% model.Priors = model0.Priors;
% for i=1:model.nbStates
% 	model.Mu(:,i) = Data * GAMMA2(i,:)';
% 	DataTmp = Data - repmat(model.Mu(:,i),1,nbData*nbSamples);
% 	model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp';
% end

%Refinement of parameters
model.params_diagRegFact = 1E-3;
[model, H] = EM_GMM(Data, model);

% %Create single Gaussian N(MuQ,SigmaQ) based on h	
% h = H(:,1:nbData);
% h = h ./ repmat(sum(h,1),model.nbStates,1);
% MuQ = zeros(model.nbVar*nbData,1);
% SigmaQ = zeros(model.nbVar*nbData);
% for t=1:nbData
% 	id = (t-1)*model.nbVar+1:t*model.nbVar;
% 	for i=1:model.nbStates
% 		MuQ(id) = MuQ(id) + model.Mu(:,i) * h(i,t);
% 		SigmaQ(id,id) = SigmaQ(id,id) + model.Sigma(:,:,i) * h(i,t);
% 	end
% end


%% Iterative continuous LQR with infinite horizon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

tar = model.Mu(:,qList);
%tar = reshape(MuQ,model.nbVar,nbData);
%dtar = gradient(tar,1,2)/model.dt;
dtar = zeros(model.nbVar,nbData);

for n=1:nbRepros
	X = Data(:,1) + [randn(model.nbVarPos,1)*2E0; zeros(model.nbVar-model.nbVarPos,1)];
	for t=1:nbData
		%id = (t-1)*model.nbVar+1:t*model.nbVar;
		%Q = inv(SigmaQ(id,id));
		Q = inv(model.Sigma(:,:,qList(t)));
		
		P = solveAlgebraicRiccati_eig(A, B*(R\B'), (Q+Q')/2); 
		%P = solveAlgebraicRiccati_Schur(A, B*(R\B'), (Q+Q')/2); 
		%P = care(A, B, (Q+Q')/2, R); %[P,~,L]=...
		
		%Variable for feedforward term computation
		d = (P*B*(R\B')-A') \ (P*dtar(:,t) - P*A*tar(:,t)); 
	
		L = R\B'*P; %Feedback gain (for continuous systems)
% 		M = R\B'*d; %Feedforward term
% 		u =  -L * (X - model.Mu(:,qList(t))) + M; %Compute acceleration (with feedback and feedforward terms)
		u =  -L * (X - tar(:,t)); %Compute acceleration (with feedback terms only)
		DX = A*X + B*u;
		X = X + DX * model.dt;
		r(n).Data(:,t) = X;
	end
end


% %% Compute acceleration map
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbGrid = 40;
% [X,Y] = meshgrid(linspace(min(Data(1,:)),max(Data(1,:)),nbGrid), linspace(min(Data(2,:)),max(Data(2,:)),nbGrid));
% xs(1,:) = reshape(X,1,size(X,1)^2);
% xs(2,:) = reshape(Y,1,size(Y,1)^2);
% H = [];
% for i=1:model.nbStates
% 	H(i,:) = model.Priors(i) * gaussPDF(xs, model.Mu(1:2,i), model.Sigma(1:2,1:2,i));
% end
% [~,qList] = max(H,[],1);
% for t=1:size(xs,2)
% 	Q = inv(model.Sigma(:,:,qList(t)));
% 	P = solveAlgebraicRiccati_eig(A, B/R*B', (Q+Q')/2); 
% 	L = R\B'*P; %Feedback term
% 	xs(3:4,t) =  -L * ([xs(1:2,t); zeros(2,1)] - model.Mu(:,qList(t))); 
% end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot position
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; 
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [0.5 0.5 0.5], .3);
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','color',[.7 .7 .7]);
end
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
	plot(r(n).Data(1,1), r(n).Data(2,1), '.','markersize',18,'color',[.6 0 0]);
end
%quiver(xs(1,:),xs(2,:),xs(3,:),xs(4,:)); %Plot acceleration map
axis equal; 
xlabel('x_1'); ylabel('x_2');

% %Plot velocity
% figure('position',[1020 10 1000 1000],'color',[1 1 1]); hold on;  
% plotGMM(model.Mu(3:4,:), model.Sigma(3:4,3:4,:), [0.5 0.5 0.5], .3);
% for n=1:nbSamples
% 	plot(Data(3,(n-1)*nbData+1:n*nbData), Data(4,(n-1)*nbData+1:n*nbData), '-','color',[.7 .7 .7]);
% end
% for n=1:nbRepros
% 	plot(r(n).Data(3,:), r(n).Data(4,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
% 	plot(r(n).Data(3,1), r(n).Data(4,1), '.','markersize',18,'color',[.6 0 0]);
% end
% plot(0,0,'k+');
% axis equal;
% xlabel('dx_1'); ylabel('dx_2');

%print('-dpng','graphs/demo_LQR_infHor03.png');
pause;
close all;