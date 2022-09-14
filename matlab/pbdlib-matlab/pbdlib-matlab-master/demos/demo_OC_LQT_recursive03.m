function demo_OC_LQT_recursive03
% Recursive computation of linear quadratic tracking (with feedback and feedforward terms),
% by relying on a GMM encoding of only position data.
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
nbSamples = 3; %Number of demonstrations
nbRepros = 5; %Number of reproductions
nbData = 100; %Number of datapoints

model.nbStates = 5; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 1; %Number of static & dynamic features (D=1 for [x1,x2])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.rfactor = 1E-6;	%Control cost in LQR
model.dt = 0.01; %Time step duration

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDeriv = model.nbDeriv + 1; %For definition of dynamical system

% %Integration with Euler method 
% Ac1d = diag(ones(nbDeriv-1,1),1); 
% Bc1d = [zeros(nbDeriv-1,1); 1];
% A = kron(eye(nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% B = kron(Bc1d*model.dt, eye(model.nbVarPos));
% C = kron([1, 0], eye(model.nbVarPos));

%Integration with higher order Taylor series expansion
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD
C = kron([1, 0], eye(model.nbVarPos));

% %Conversion with control toolbox
% Ac1d = diag(ones(nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt);
% A = kron(sysd.a, eye(model.nbVarPos)); %Discrete nD
% B = kron(sysd.b, eye(model.nbVarPos)); %Discrete nD
% C = kron([1, 0], eye(model.nbVarPos));


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
model = init_GMM_kmeans(Data,model);
%Refinement of parameters
[model, H] = EM_GMM(Data, model);
%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1


%% Iterative LQT (finite horizon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = zeros(model.nbVarPos*2,model.nbVarPos*2,nbData);
P(1:model.nbVarPos,1:model.nbVarPos,end) = inv(model.Sigma(:,:,qList(nbData)));
d = zeros(model.nbVarPos*2, nbData);
Q = zeros(model.nbVarPos*2);
for t=nbData-1:-1:1
	Q(1:model.nbVarPos,1:model.nbVarPos) = inv(model.Sigma(:,:,qList(t)));
	P(:,:,t) = Q - A' * (P(:,:,t+1) * B / (B' * P(:,:,t+1) * B + R) * B' * P(:,:,t+1) - P(:,:,t+1)) * A;
	d(:,t) = (A' - A'*P(:,:,t+1) * B / (R + B' * P(:,:,t+1) * B) * B' ) * (P(:,:,t+1) * ...
		( A * [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - [model.Mu(:,qList(t+1)); zeros(model.nbVarPos,1)] ) + d(:,t+1));
end
%Reproduction with feedback (FB) and feedforward (FF) terms
for n=1:nbRepros
	X = [Data(:,1) + randn(model.nbVarPos,1)*2E0; zeros(model.nbVarPos,1)];
	r(n).X0 = X;
	for t=1:nbData
		r(n).Data(:,t) = X; %Log data
		K = (B' * P(:,:,t) * B + R) \ B' * P(:,:,t) * A; %FB term
		M = -(B' * P(:,:,t) * B + R) \ B' * (P(:,:,t) * ...
			(A * [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - [model.Mu(:,qList(t)); zeros(model.nbVarPos,1)]) + d(:,t)); %FF term
		u = K * ([model.Mu(:,qList(t)); zeros(model.nbVarPos,1)] - X) + M; %Acceleration command with FB and FF terms
		X = A * X + B * u; %Update of state vector
	end
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
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
end
axis equal;
xlabel('x_1'); ylabel('x_2');

% %Plot velocity
% figure('position',[1020 10 1000 1000],'color',[1 1 1]); hold on; 
% for n=1:nbSamples
% 	DataVel = gradient(Data(:,(n-1)*nbData+1:n*nbData)) / model.dt;
% 	plot(DataVel(1,:), DataVel(2,:), '-','color',[.7 .7 .7]);
% end
% for n=1:nbRepros
% 	plot(r(n).Data(3,:), r(n).Data(4,:), '-','linewidth',2,'color',[.8 0 0]); %Reproduction with iterative LQR
% 	plot(r(n).Data(3,1), r(n).Data(4,1), '.','markersize',18,'color',[.6 0 0]);
% end
% plot(0,0,'k+');
% axis equal;
% xlabel('dx_1'); ylabel('dx_2');

%print('-dpng','graphs/demo_iterativeLQR02.png');
pause;
close all;