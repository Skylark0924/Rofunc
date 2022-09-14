function demo_OC_LQT_online02
% MPC recomputed in an online manner with a time horizon, by relying on a GMM encoding of position and velocity data (with animation).
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
nbSamples = 8; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbData = 100; %Number of datapoints
nbD = 30; %Time window for LQR computation

model.nbStates = 5; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step duration
model.rfactor = 1E-8;	%Control cost in LQR

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

%Build Sx and Su matrices for batch LQR
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
load('data/2Dletters/M.mat');
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
% model = init_GMM_kmeans(Data,model);
model = init_GMM_kbins(Data,model,nbSamples);
[model, H] = EM_GMM(Data, model);

%Precomputation of inverse 
for i=1:model.nbStates
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end	

%List of viapoints with time and error information 
[~,q] = max(H(:,1:nbData),[],1); 
xDes = [];
qCurr = q(1);
MuQ0 = zeros(model.nbVar*(nbData+nbD),1);
Q0 = zeros(model.nbVar*(nbData+nbD));
t_old = 1; 
for t=1:nbData
	if qCurr~=q(t) || t==nbData
% 		tm = t;
		tm = t_old + floor((t-t_old)/2);
		xDes = [xDes, [tm; model.Mu(:,qCurr); diag(model.Sigma(:,:,qCurr)).^.5 .* 2E0]];	%for plots
		id = [1:model.nbVar] + (tm-1) * model.nbVar;
		MuQ0(id) = model.Mu(:,qCurr);
		Q0(id,id) = model.Sigma(:,:,qCurr);
		qCurr = q(t);
		t_old = t;
	end
end	

% %% Task setting
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tl = linspace(1,nbData,model.nbStates+1);
% tl = round(tl(2:end)); 
% MuQ0 = zeros(model.nbVar*nbData,1); 
% Q0 = zeros(model.nbVar*nbData);
% j = 1;
% for t=1:length(tl)
% 	model.Mu(:,j) = rand(model.nbVarPos,1) - 0.5;
% 	model.Sigma(:,:,j) = eye(model.nbVarPos) .* 1E-2;
% 	id(:,t) = [1:model.nbVarPos] + (tl(t)-1) * model.nbVar;
% 	Q0(id(:,t), id(:,t)) = inv(model.Sigma(:,:,j));
% 	MuQ0(id(:,t)) = model.Mu(:,j);
% 	j = j + 1;
% end


%% MPC (batch LQT recomputed in an online manner) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbRepros
	x = [Data(1:model.nbVarPos,1); zeros(model.nbVar-model.nbVarPos,1)];
	for t=1:nbData
		
% 		%Version with stepwise tracking
% 		id = [t:min(t+nbD-1,nbData), repmat(nbData,1,t-nbData+nbD-1)]; %Time steps involved in the MPC computation
% 		%Set list of states according to first demonstration (alternatively, an HSMM can be used)
% 		[~,q] = max(H(:,id),[],1); 
% 		%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q
% 		MuQ = reshape(model.Mu(:,q), model.nbVar*nbD, 1); 		
% 		Q = (kron(ones(nbD,1), eye(model.nbVar)) * reshape(model.invSigma(:,:,q), model.nbVar, model.nbVar*nbD)) .* kron(eye(nbD), ones(model.nbVar));

		%Version with viapoints
		id2 = (t-1)*model.nbVar+1:(t+nbD-1)*model.nbVar;
		MuQ = MuQ0(id2);
		Q = Q0(id2,id2);
		
		%Control command
		u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x);
		%Log current estimate (for visualization purpose)
		r(n).s(t).u = reshape(u, model.nbVarPos, nbD-1);
		r(n).s(t).x = reshape(Sx*x+Su*u, model.nbVar, nbD);
		r(n).x(:,t) = x;
		r(n).u(:,t) = u(1:model.nbVarPos);
		%Update x with first control command
		x = A * x + B * u(1:model.nbVarPos);
	end
end


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1600 800],'color',[1 1 1]); 
%Plot 2D
subplot(model.nbVar+model.nbVarPos, 2, [1:2:(model.nbVar+model.nbVarPos)*2]); hold on; axis off;
limAxes = [min(r(1).x(1,:))-8E0, max(r(1).x(1,:))+8E0, min(r(1).x(2,:))-8E0, max(r(1).x(2,:))+8E0];
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.8 .8 .8]);
% for n=1:nbSamples
% 	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
% end
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis equal; axis(limAxes);

%State profile
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'}; 
for j=1:model.nbVar
	subplot(model.nbVar+model.nbVarPos, 2, j*2); hold on;
	v(j).limAxes = [1, nbData+nbD, min(r(1).x(j,:))-8E0, max(r(1).x(j,:))+8E0];
	
% 	%Plot stepwise reference
% 	msh=[]; x0=[];
% 	for t=1:nbData-1
% 		if size(msh,2)==0
% 			msh(:,1) = [t; model.Mu(j,q(t))];
% 		end
% 		if t==nbData-1 || q(t+1)~=q(t)
% 			i = q(t);
% 			msh(:,2) = [t+1; model.Mu(j,i)];
% 			sTmp = model.Sigma(j,j,qList(t))^.5;
% 			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
% 			patch(msh2(1,:), msh2(2,:), [.7 .7 .7],'edgecolor',[.6 .6 .6],'facealpha', .4, 'edgealpha', .4);
% 			plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.6 .6 .6]);
% 			if msh(1,1)>1
% 				plot([msh(1,1) msh(1,1)], v(j).limAxes(3:4), ':','linewidth',1,'color',[.5 .5 .5]);
% 			end
% 			x0 = [x0 msh];
% 			msh=[];
% 		end
% 	end
	
	%Plot viapoints reference
	for t=1:size(xDes,2)
		errorbar(xDes(1,t), xDes(1+j,t), xDes(1+model.nbVar+j,t), 'color',[.7 .7 .7]);
		plot(xDes(1,t), xDes(1+j,t), '.','markersize',15,'color',[.5 .5 .5]);
	end
	
% 	for n=1:nbSamples
% 		plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[0 0 0]);
% 	end
	for n=1:nbRepros
		plot(r(n).x(j,:), '-','linewidth',2,'color',[.8 0 0]);
	end
	if j<7
		ylabel(labList{j},'fontsize',22,'interpreter','latex');
	end
	axis(v(j).limAxes);
	set(gca,'xtick',[],'ytick',[]);
end

%Control profile
for j=1:model.nbVarPos
	subplot(model.nbVar+model.nbVarPos, 2, (model.nbVar+j)*2); hold on;
	v(model.nbVar+j).limAxes = [1, nbData+nbD, min(r(1).u(j,:))-8E1, max(r(1).u(j,:))+8E1];
% 	patch([1:nbData-1 nbData-1:-1:1], [r(1).u(j,:)-S(:)' fliplr(r(1).u(j,:)+S(:)')], [.8 0 0],'edgecolor',[.6 0 0],'facealpha', .2, 'edgealpha', .2);
	%Plot reproduction samples
	for n=1:nbRepros
		plot(r(n).u(j,:), '.','markersize',10,'color',[.8 0 0]);
	end
	ylabel(['$u_' num2str(j) '$'],'fontsize',22,'interpreter','latex');
	axis(v(model.nbVar+j).limAxes);
	set(gca,'xtick',[],'ytick',[]);
end
xlabel('$t$','fontsize',22,'interpreter','latex');

%Time window anim
for tt=1:nbData
	subplot(model.nbVar+model.nbVarPos, 2, [1:2:(model.nbVar+model.nbVarPos)*2]); hold on; axis off;
	h = plot(r(1).s(tt).x(1,:), r(1).s(tt).x(2,:), '-','linewidth',2,'color',[0 .6 0]);
	h = [h, plot(r(1).s(tt).x(1,1), r(1).s(tt).x(2,1), '.','markersize',18,'color',[0 .6 0])];
	for j=1:model.nbVar+model.nbVarPos
		subplot(model.nbVar+model.nbVarPos, 2, j*2); hold on;
		msh = [tt tt+nbD-1 tt+nbD-1 tt tt; v(j).limAxes([3,3]) v(j).limAxes([4,4]) v(j).limAxes(3)];
		h = [h, plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.6 .6 .6])];
		if j<=model.nbVar
			h = [h, plot(tt:tt+nbD-1, r(1).s(tt).x(j,:), '-','linewidth',2,'color',[0 .6 0])];
		else
			h = [h, plot(tt:tt+nbD-2, r(1).s(tt).u(j-model.nbVar,:), '-','linewidth',2,'color',[0 .6 0])];
		end
	end
	drawnow;
	delete(h);
end
	
% %Speed profile
% if model.nbDeriv>1
% 	subplot(model.nbVar+model.nbVarPos+1,1,model.nbVar+1); hold on;
% % 	for n=1:nbSamples
% % 		sp = sqrt(Data(3,(n-1)*nbData+1:n*nbData).^2 + Data(4,(n-1)*nbData+1:n*nbData).^2);
% % 		plot(sp, '-','linewidth',.5,'color',[.6 .6 .6]);
% % 	end
% 	for n=1:nbRepros
% 		sp = sqrt(r(n).x(3,:).^2 + r(n).x(4,:).^2);
% 		plot(sp, '-','linewidth',2,'color',[.8 0 0]);
% 	end
% 	ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
% end

%print('-dpng','graphs/demo_MPC_online02.png');
pause;
close all;