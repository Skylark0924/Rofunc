function demo_DMP_batchLQR01
% Emulation of DMP with a spring system controlled by batch LQR.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
% }
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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
model.nbStates = 4; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * (model.nbDeriv+1); %Dimension of state vector
model.dt = 0.01; %Time step duration
model.alpha = 3.0; %Decay factor
model.rfactor = 1E-6;	%Control cost in LQR
model.kP = 50; %Stiffness gain (initial estimate)
%model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.kV = 2*model.kP^.5; %Damping (for critically damped system)

nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbData = 200; %Number of datapoints

% %Dynamical System settings (discrete version)
% Ac = kron([0, 1; -model.kP, -model.kV], eye(model.nbVarPos));
% A = eye(model.nbVar) + Ac * model.dt;
% B = kron([0; model.dt], eye(model.nbVarPos));

%Dynamical System settings (discrete version)
Ac = kron([0, 1, 0; -model.kP, -model.kV, model.kP; 0, 0, 0], eye(model.nbVarPos));
A = eye(model.nbVar) + Ac * model.dt;
B = kron([0; model.dt; 0], eye(model.nbVarPos));

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);

%Evolution of DMP decay term
sIn(1) = 1; 
for t=2:nbData
	sIn(t) = sIn(t-1) - model.alpha * sIn(t-1) * model.dt; %Update of decay term (ds/dt=-alpha s)
end

%Build Sx and Su matrices for batch LQR to have ddx = kP*(xTar-x(:,t)) - kV*dx(:,t) + u(:,t) * sIn(t)
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B*sIn(1); %*sIn(1)
for t=2:nbData
	id1 = (t-1)*model.nbVar+1:nbData*model.nbVar;
	
	%Ac = kron([0, 1; -model.kP*(1-sIn(t)), -model.kV*(1-sIn(t))], eye(model.nbVarPos));
	Ac = kron([0, 1, 0; -model.kP*(1-sIn(t)), -model.kV*(1-sIn(t)), model.kP*(1-sIn(t)); 0, 0, 0], eye(model.nbVarPos));
	A = eye(model.nbVar) + Ac * model.dt;

	Sx(id1,:) = Sx(id1,:) * A; 
	id1 = (t-1)*model.nbVar+1:t*model.nbVar; 
	id2 = 1:(t-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [Sx(id1,:)*B*sIn(t), M]; %*sIn(t)
end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
	if n==1
		model.xTar = s(n).Data(1:model.nbVarPos,end); %Equilibrium point of the spring
	end
	Data = [Data [s(n).Data; repmat(model.xTar,1,nbData)]]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
model = init_GMM_kbins(Data,model,nbSamples);
[model, H] = EM_GMM(Data, model);


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1
MuQ = reshape(model.Mu(:,qList), model.nbVar*nbData, 1); 
SigmaQ = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.Sigma(:,:,qList), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));
SuInvSigmaQ = Su' / SigmaQ;
Rq = SuInvSigmaQ * Su + R;
for n=1:nbRepros
	X = Data(:,1); 
 	rq = SuInvSigmaQ * (MuQ-Sx*X);
 	u = Rq \ rq; 
	r(n).Data = reshape(Sx*X+Su*u, model.nbVar, nbData);
	r(n).u = reshape(u, model.nbVarPos, nbData-1);
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 700 700],'color',[1 1 1]); hold on; axis off;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [0.5 0.5 0.5]);
plot(Data(1,:), Data(2,:), 'k.');
plot(model.xTar(1),model.xTar(2),'k+','markersize',30,'linewidth',2);
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis equal; 


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$u_1$','$u_2$'};
figure('position',[720 10 600 700],'color',[1 1 1]); 
for j=1:model.nbVar
subplot(model.nbVar+1,1,j); hold on; grid on;
if j<5
	for n=1:nbSamples
		plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[0 0 0]);
	end
	for n=1:nbRepros
		plot(r(n).Data(j,:), '-','linewidth',1,'color',[.8 0 0]);
	end
else
	for n=1:nbRepros
		plot(r(n).u(j-4,:), '-','linewidth',1,'color',[.8 0 0]);
	end
end
if j<7
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
end
subplot(model.nbVar+1,1,model.nbVar+1); hold on; grid on;
plot(sIn,'k-');
xlabel('t','fontsize',14,'interpreter','latex');
ylabel('s','fontsize',14,'interpreter','latex');

%print('-dpng','graphs/demo_DMP_batchLQR01.png');
pause;
close all;

