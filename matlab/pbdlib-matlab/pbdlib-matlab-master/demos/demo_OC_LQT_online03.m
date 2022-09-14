function demo_OC_LQT_online03
% Obstacle avoidance with MPC recomputed in an online manner.
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
nbSamples = 5; %Number of demonstrations
nbRepros = 3; %Number of reproductions in new situations
nbData = 200; %Number of datapoints
nbD = 60; %Time window for LQR computation

model.nbStates = 4; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E-2; %Time step duration
model.rfactor = 1E-6;	%Control cost in LQR

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbD-1),R);


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

%Build Sx and Su matrices for batch LQR, see Eq. (35)
Su = zeros(model.nbVar*nbD, model.nbVarPos*(nbD-1));
Sx = kron(ones(nbD,1),eye(model.nbVar));
M = B;
for n=2:nbD
	id1 = (n-1)*model.nbVar+1:nbD*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


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
[model, H] = EM_GMM(Data, model);

%Precomputation of inverse 
for i=1:model.nbStates
	model.Q(:,:,i) = inv(model.Sigma(:,:,i));
% 	model.Q(:,:,i) = blkdiag(inv(model.Sigma(1:2,1:2,i)), zeros(2));
end
	

%% MPC (batch LQT recomputed in an online manner)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu0 = [1; -1]; %Obstacle
th = 1E-5; %gaussPDF([0;0],[16^.5;0],eye(2)*16);
sc = 40;
for n=1:nbRepros
	%Increase repulsive force at each reproduction
	sc = sc * 10;
	if n==nbRepros
		sc = 0;
	end
	
	x = Data(:,1); 	
	for t=1:nbData
		%Set list of states according to first demonstration (alternatively, an HSMM can be used)
		id = [t:min(t+nbD-1,nbData), repmat(nbData,1,t-nbData+nbD-1)];
		[~,q] = max(H(:,id),[],1); %works also for nbStates=1
		
		%Compute tracking controller
		MuQ = reshape(model.Mu(:,q), model.nbVar*nbD, 1); 
		Q = (kron(ones(nbD,1), eye(model.nbVar)) * reshape(model.Q(:,:,q), model.nbVar, model.nbVar*nbD)) .* kron(eye(nbD), ones(model.nbVar));
		
		%Compute obstacle avoidance controller
		w0 = (max(gaussPDF(x(1:2,1), Mu0, eye(2).*1E0),th)-th) .* sc;
		
		e = x(1:2,1) - Mu0; 
		e1 = e ./ norm(e); 
		e2 = -(e1*e1') * x(3:4,1); %e2 = [e1(2); -e1(1)];
		u_rep1 = [w0 .* e1; zeros((nbD-2)*2, 1)]; %Controller moving away from obstacle
		R_rep1 = blkdiag((e1*e1').*w0, zeros((nbD-2)*2)); 
		u_rep2 = [w0 .* e2; zeros((nbD-2)*2, 1)]; %Controller slowing down when moving around obstacle
		R_rep2 = blkdiag((e2*e2').*w0, zeros((nbD-2)*2)); 
		
		%Log data
		r(n).Data(:,t) = x;
		r(n).w(t) = w0;
		r(n).q(t) = q(1);
		r(n).MuQ(:,t) = model.Mu(:,q(1));
		r(n).SigmaQ(:,:,t) = model.Sigma(:,:,q(1));
		
		%Fusion of controllers (tracking of stepwise reference while avoiding an obstacle)
		u = (Su' * Q * Su + R + R_rep1 + R_rep2) \ (Su' * Q * (MuQ-Sx*x) + R_rep1 * u_rep1 + R_rep2 * u_rep2); %includes obstacle avoidance
		
		%Update X with first control command
		x = A * x + B * u(1:model.nbVarPos);
	end
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; axis off;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.8 .8 .8]);
% for n=1:nbSamples
% 	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
% end	
plotGMM(Mu0(1:2,1), eye(2)*3, [.8 0 0], .3);
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[1-n/nbRepros 0 0]);
	for t=1:nbData
		if r(n).w(t)>0
			plot(r(n).Data(1,t), r(n).Data(2,t), '.','markersize',12,'color',[1-n/nbRepros 0 0]);
		end
	end
end
% plot(r(3).Data(1,:), r(3).Data(2,:), '-','linewidth',6,'color',[.4 .4 .4]);
% plot(r(2).Data(1,:), r(2).Data(2,:), '-','linewidth',3,'color',[.8 0 0]);
axis equal; 
% print('-dpng','graphs/demo_MPC_online03a.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'};
figure('position',[820 10 800 800],'color',[1 1 1]); 
for j=1:model.nbVar
	subplot(model.nbVar+1,1,j); hold on;
	limAxes = [1, nbData, min(Data(j,:))-4E0, max(Data(j,:))+4E0];
	msh=[]; x0=[];
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; r(1).MuQ(j,t)];
		end
		if t==nbData-1 || r(1).q(t+1)~=r(1).q(t)
			i = r(1).q(t);
			msh(:,2) = [t+1; r(1).MuQ(j,t)];
			sTmp = r(1).SigmaQ(j,j,t)^.5;
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), [.7 .7 .7],'edgecolor',[.6 .6 .6],'facealpha', .4, 'edgealpha', .4);
			plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.5 .5 .5]);
			if msh(1,1)>1
				plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.5 .5 .5]);
			end
			x0 = [x0 msh];
			msh=[];
		end
		if r(1).w(t)==0 && r(1).w(t+1)>0
			plot([t t], limAxes(3:4), '--','linewidth',1,'color',[.8 0 0]);
		elseif r(1).w(t)>0 && r(1).w(t+1)==0
			plot([t t], limAxes(3:4), '--','linewidth',1,'color',[0 .7 0]);
		end
	end
	if j<3
		plot([1,nbData],[Mu0(j),Mu0(j)],'-','linewidth',2,'color',[.8 0 0]);
	end
% 	for n=1:nbSamples
% 		plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[.6 .6 .6]);
% 	end
	for n=1:nbRepros
		plot(r(n).Data(j,:), '-','linewidth',2,'color',[1-n/nbRepros 0 0]);
	end
	if j<7
		ylabel(labList{j},'fontsize',14,'interpreter','latex');
	end
	axis(limAxes);
end

subplot(model.nbVar+1,1,model.nbVar+1); hold on;
for n=1:nbRepros
	plot(r(n).w,'-','color',[1-n/nbRepros 0 0]);
end
ylabel('$w$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');
% print('-dpng','graphs/demo_MPC_online03b.png');

pause;
close all;