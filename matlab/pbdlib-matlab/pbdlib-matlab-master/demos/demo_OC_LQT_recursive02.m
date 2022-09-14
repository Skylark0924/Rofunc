function demo_OC_LQT_recursive02
% Recursive computation of linear quadratic tracking (with feedback and feedforward terms),
% by relying on a GMM encoding of position and velocity data, including comparison with batch LQT.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 5; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E-2; %Time step duration
model.rfactor = 0.1 * model.dt^model.nbDeriv;	%Control cost in LQR
nbSamples = 3; %Number of demonstrations
nbRepros = 5; %Number of reproductions
nbData = 200; %Number of datapoints

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); 
% Bc1d = [zeros(model.nbDeriv-1,1); 1];
% A = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% B = kron(Bc1d*model.dt, eye(model.nbVarPos));

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

% %Conversion with control toolbox
% Ac1d = diag(ones(model.nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(model.nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,model.nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt);
% A = kron(sysd.a, eye(model.nbVarPos)); %Discrete nD
% B = kron(sysd.b, eye(model.nbVarPos)); %Discrete nD


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
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
[model, H] = EM_GMM(Data, model);
%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1


%% Iterative LQT (finite horizon, discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = zeros(model.nbVar,model.nbVar,nbData);
P(:,:,end) = inv(model.Sigma(:,:,qList(nbData)));
d = zeros(model.nbVar, nbData);

%Backward computation
for t=nbData-1:-1:1
	Q = inv(model.Sigma(:,:,qList(t)));
	P(:,:,t) = Q - A' * (P(:,:,t+1) * B / (B' * P(:,:,t+1) * B + R) * B' * P(:,:,t+1) - P(:,:,t+1)) * A;
	d(:,t) = (A' - A' * P(:,:,t+1) * B / (R + B' * P(:,:,t+1) * B) * B' ) * (P(:,:,t+1) * (A * model.Mu(:,qList(t)) - model.Mu(:,qList(t+1))) + d(:,t+1));
end

%Reproduction with feedback (FB) and feedforward (FF) terms
for n=1:nbRepros
	X = Data(:,1) + [randn(model.nbVarPos,1)*2E0; zeros(model.nbVar-model.nbVarPos,1)];
	r(n).X0 = X;
	for t=1:nbData
		r(n).Data(:,t) = X; %Log data
		K = (B' * P(:,:,t) * B + R) \ B' * P(:,:,t) * A; %FB gain
		
% 		%Test ratio between kp and kv
% 		kp = eigs(K(:,1:2));
% 		kv = eigs(K(:,3:4));
% 		ratio = kv ./ (2.*kp).^.5
% 		figure; hold on;
% 		plotGMM(zeros(2,1), K(:,1:2), [.8 0 0],.3);
% 		plotGMM(zeros(2,1), K(:,3:4), [.8 0 0],.3);
% 		axis equal;
% 		pause;
% 		close all;

		uff = -(B' * P(:,:,t) * B + R) \ B' * (P(:,:,t) * (A * model.Mu(:,qList(t)) - model.Mu(:,qList(t))) + d(:,t)); %Feedforward term
		u = K * (model.Mu(:,qList(t)) - X) + uff; %Acceleration command with FB and FF terms
		X = A * X + B * u; %Update of state vector
		
		r(n).K(:,:,t) = K;
		r(n).uff(:,t) = uff;
	end
end


% %% Iterative LQT (as in Table 2 on p.2 of http://web.mst.edu/~bohner/papers/tlqtots.pdf)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K = zeros(model.nbVarPos, model.nbVar, nbData-1);
% Kff = zeros(model.nbVarPos, model.nbVar, nbData-1);
% P = zeros(model.nbVar, model.nbVar, nbData);
% P(:,:,end) = inv(model.Sigma(:,:,qList(nbData)));
% v = zeros(model.nbVar, nbData);
% v(:,end) = inv(model.Sigma(:,:,qList(nbData))) * model.Mu(:,qList(nbData));
% % r(1).xx = zeros(model.nbVar, nbData-1);
% %Backward computation
% for t=nbData-1:-1:1
% 	Q = inv(model.Sigma(:,:,qList(t)));
% 	Kff(:,:,t) = (B' * P(:,:,t+1) * B + R) \ B'; %FF
% 	K(:,:,t) = Kff(:,:,t) * P(:,:,t+1) * A; %FB
% 	P(:,:,t) = A' * P(:,:,t+1) * (A - B * K(:,:,t)) + Q;
% 	v(:,t) = (A - B * K(:,:,t))' * v(:,t+1) + Q * model.Mu(:,qList(t));
% % 	r(1).xx(:,t) = (P(:,:,t) * A) \ v(:,t);
% end
% %Reproduction with feedback (FB) and feedforward (FF) terms
% for n=1:nbRepros
% 	X = r(n).X0;
% 	r2(n).Data(:,1) = X; %Log data
% 	for t=2:nbData
% 		r2(n).Data(:,t) = X; %Log data
% 		u = -K(:,:,t-1) * X + Kff(:,:,t-1) * v(:,t); %Acceleration command with FB and FF terms
% 		X = A * X + B * u; %Update of state vector
% 	end
% end


%% Batch LQT (for comparison)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q, see Eq. (27)
MuQ = reshape(model.Mu(:,qList), model.nbVar*nbData, 1); 
SigmaQ = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.Sigma(:,:,qList), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));
%Build Sx and Su matrices for batch LQR, see Eq. (35)
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M];
end
%Set matrices to compute the damped weighted least squares estimate, see Eq. (37)
SuInvSigmaQ = Su' / SigmaQ;
Rq = SuInvSigmaQ * Su + kron(eye(nbData-1),R);

%Reproductions
for n=1:nbRepros
	X = r(n).X0;
 	rq = SuInvSigmaQ * (MuQ-Sx*X);
 	u = Rq \ rq; 
	r2(n).Data = reshape(Sx*X+Su*u, model.nbVar, nbData);
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot position
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; 
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [0.5 0.5 0.5], .3);
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','color',[.7 .7 .7]);
end
for n=1:nbRepros
	h(1) = plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
	h(2) = plot(r2(n).Data(1,:), r2(n).Data(2,:), '--','linewidth',2,'color',[0 .8 0]); %Reproduction with batch LQR
end
% plot(r(1).xx(1,:), r(1).xx(2,:), '-','linewidth',2,'color',[0 0 .8]); 
axis equal; 
xlabel('x_1'); ylabel('x_2');
legend(h,'Iterative LQR','Batch LQR');

% %Plot velocity
% figure('position',[1020 10 1000 1000],'color',[1 1 1]); hold on;  
% plotGMM(model.Mu(3:4,:), model.Sigma(3:4,3:4,:), [0.5 0.5 0.5], .3);
% for n=1:nbSamples
% 	plot(Data(3,(n-1)*nbData+1:n*nbData), Data(4,(n-1)*nbData+1:n*nbData), '-','color',[.7 .7 .7]);
% end
% for n=1:nbRepros
% 	plot(r(n).Data(3,:), r(n).Data(4,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
% 	plot(r(n).Data(3,1), r(n).Data(4,1), '.','markersize',18,'color',[.6 0 0]);
% 	plot(r2(n).Data(3,:), r2(n).Data(4,:), '--','linewidth',2,'color',[0 .8 0]); %Reproduction with batch LQR
% end
% % plot(r(1).xx(3,:), r(1).xx(4,:), '-','linewidth',2,'color',[0 0 .8]); 
% plot(0,0,'k+');
% axis equal;
% xlabel('dx_1'); ylabel('dx_2');

%print('-dpng','graphs/demo_iterativeLQR02.png');
pause;
close all;