function demo_OC_LQT03
% Batch computation of linear quadratic tracking problem, by tracking only position references.
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
nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbNewRepros = 0; %Number of stochastically generated reproductions
nbData = 200; %Number of datapoints

model.nbStates = 4; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (nbDeriv=1 for just x)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E-2; %Time step duration
model.rfactor = 1E-4;	%Control cost in LQR

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); 
% Bc1d = [zeros(model.nbDeriv-1,1); 1];
% A = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% B = kron(Bc1d*model.dt, eye(model.nbVarPos));
% C = kron([1, 0], eye(model.nbVarPos));

%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) .* model.dt^i ./ factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i ./ factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD
C = kron([1, zeros(1,model.nbDeriv-1)], eye(model.nbVarPos));

% %Conversion with control toolbox
% Ac1d = diag(ones(nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt);
% A = kron(sysd.a, eye(model.nbVarPos)); %Discrete nD
% B = kron(sysd.b, eye(model.nbVarPos)); %Discrete nD
% C = kron([1, 0], eye(model.nbVarPos));


%Build CSx and CSu matrices for batch LQR
CSu = zeros(model.nbVarPos*nbData, model.nbVarPos*(nbData-1));
CSx = kron(ones(nbData,1), [eye(model.nbVarPos) zeros(model.nbVarPos, model.nbVarPos*(model.nbDeriv-1))]);
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos;
	CSx(id1,:) = CSx(id1,:) * A;
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos; 
	id2 = 1:(n-1)*model.nbVarPos;
	CSu(id1,id2) = C * M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

%Refinement of parameters
[model, H] = EM_GMM(Data, model);

%Precomputation of inverse and eigencomponents (optional)
for i=1:model.nbStates
	[model.V(:,:,i), model.D(:,:,i)] = eigs(model.Sigma(:,:,i));
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end

%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q, see Eq. (27)
MuQ = reshape(model.Mu(:,qList), model.nbVarPos*nbData, 1); 
SigmaQ = (kron(ones(nbData,1), eye(model.nbVarPos)) * reshape(model.Sigma(:,:,qList), model.nbVarPos, model.nbVarPos*nbData)) .* kron(eye(nbData), ones(model.nbVarPos));


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set matrices to compute the damped weighted least squares estimate
CSuInvSigmaQ = CSu' / SigmaQ;
Rq = CSuInvSigmaQ * CSu + R;
%Reproductions
for n=1:nbRepros
	%x = [Data(:,1)+randn(model.nbVarPos,1)*2E0; zeros(model.nbVarPos,1)]; 
	x = [Data(:,1); zeros(model.nbVarPos*(model.nbDeriv-1),1)]; 
	rq = CSuInvSigmaQ * (MuQ-CSx*x);
	u = Rq \ rq; %Can also be computed with u = lscov(Rq, rq);
	r(n).Data = reshape(CSx*x+CSu*u, model.nbVarPos, nbData);
end


% %% Stochastic sampling by exploiting the GMM representation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbEigs = 2; %Number of principal eigencomponents to keep
% X = [Data(:,1); zeros(model.nbVarPos,1)]; 
% CSuInvSigmaQ = CSu' / SigmaQ;
% Rq = CSuInvSigmaQ * CSu + R;
% for n=1:nbNewRepros
% 	N = randn(nbEigs,model.nbStates) * 1E0; %Noise on all components
% 	%N = [randn(nbEigs,1), zeros(nbEigs,model.nbStates-1)] * 2E0; %Noise on first component
% 	%N = [zeros(nbEigs,model.nbStates-3), randn(nbEigs,1), zeros(nbEigs,2)] * 1E0; %Noise on second component
% 	for i=1:model.nbStates
% 		MuTmp(:,i) = model.Mu(:,i) + model.V(:,1:nbEigs,i) * model.D(1:nbEigs,1:nbEigs,i).^.5 * N(:,i);
% 	end
% 	MuQ2 = reshape(MuTmp(:,qList), model.nbVarPos*nbData, 1); 
% 	rq = CSuInvSigmaQ * (MuQ2-CSx*X);
% 	u = Rq \ rq;
% 	rnew(n).Data = reshape(CSx*X+CSu*u, model.nbVarPos, nbData); %Reshape data for plotting
% end


% %% Stochastic sampling by exploiting the GMM representation (version 2)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X = [Data(:,1); zeros(model.nbVarPos,1)]; 
% CSuInvSigmaQ = CSu' / SigmaQ;
% M = CSu * ((CSuInvSigmaQ * CSu + R) \ CSuInvSigmaQ);
% [V,D] = eig(SigmaQ);
% for n=1:nbNewRepros
% 	
% % 	N = randn(model.nbVarPos,model.nbStates) * 1E0; %Noise on all components
% % 	for i=1:model.nbStates
% % 		MuTmp(:,i) = model.Mu(:,i) + model.V(:,:,i) * model.D(:,:,i).^.5 * N(:,i);
% % 	end
% % 	MuQ2 = reshape(MuTmp(:,qList), model.nbVarPos*nbData, 1); 
% 	
% 	MuQ2 = MuQ + V * D.^.5 * randn(size(D,1),1) * 5;
% 	
% 	xtmp = CSx * X + M * (MuQ2 - CSx*X);
% 	rnew(n).Data = reshape(xtmp, model.nbVarPos, nbData); %Reshape data for plotting
% end


%% Stochastic sampling by exploiting distribution on x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbEigs = 2; %Number of principal eigencomponents to keep
x = [Data(:,1); zeros(model.nbVarPos*(model.nbDeriv-1),1)]; 
CSuInvSigmaQ = CSu' / SigmaQ;
Rq = CSuInvSigmaQ * CSu + R;
rq = CSuInvSigmaQ * (MuQ-CSx*x);
u = Rq \ rq;
Mu = CSx * x + CSu * u;
mse =  1; %abs(MuQ'/SigmaQ*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
Sigma = CSu/Rq*CSu' * mse + eye(nbData*model.nbVarPos) * 0E-10;
%[V,D] = eigs(Sigma);
[V,D] = eig(Sigma);
for n=1:nbNewRepros
	%xtmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1) * 0.1;
	xtmp = Mu + V * D.^.5 * randn(size(D,1),1) * 0.9;
	rnew(n).Data = reshape(xtmp, model.nbVarPos, nbData); %Reshape data for plotting
end


% %% Stochastic sampling by exploiting distribution on u
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbEigs = 2; %Number of principal eigencomponents to keep
% X = [Data(:,1); zeros(model.nbVarPos,1)]; 
% CSuInvSigmaQ = CSu' / SigmaQ;
% Rq = CSuInvSigmaQ * CSu + R;
% rq = CSuInvSigmaQ * (MuQ-CSx*X);
% Mu = Rq \ rq;
% mse =  abs(MuQ'/SigmaQ*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
% Sigma = inv(Rq) * mse + eye((nbData-1)*model.nbVarPos) * 1E-10;
% %[V,D] = eigs(Sigma);
% [V,D] = eig(Sigma);
% for n=1:nbNewRepros
% 	%utmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1) * 0.1;
% 	utmp = real(Mu + V * D.^.5 * randn(size(D,1),1) * 0.1);
% 	xtmp = CSx * X + CSu * utmp;
% 	rnew(n).Data = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
% end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; axis off;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.8 .8 .8]);
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
end	
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]);
	plot(r(n).Data(1,:), r(n).Data(2,:), '.','markersize',12,'color',[.8 0 0]);
end
for n=1:nbNewRepros
	plot(rnew(n).Data(1,:), rnew(n).Data(2,:), '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
end
axis equal;
% print('-dpng','graphs/demo_batchLQR02a.png');

% %% Plot timeline
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 600 600]);
% for m=1:model.nbVarPos
% 	limAxes = [1, nbData, min(Data(m,:))-4E0, max(Data(m,:))+4E0];
% 	subplot(model.nbVarPos,1,m); hold on;
% 	msh=[]; x0=[];
% 	for t=1:nbData-1
% 		if size(msh,2)==0
% 			msh(:,1) = [t; model.Mu(m,qList(t))];
% 		end
% 		if t==nbData-1 || qList(t+1)~=qList(t)
% 			msh(:,2) = [t+1; model.Mu(m,qList(t))];
% 			sTmp = model.Sigma(m,m,qList(t))^.5;
% 			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
% 			patch(msh2(1,:), msh2(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7]);
% 			plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
% 			plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
% 			x0 = [x0 msh];
% 			msh=[];
% 		end
% 	end
% 	for n=1:nbSamples
% 		plot(1:nbData, Data(m,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]);
% 	end
% 	for n=1:nbNewRepros
% 		plot(1:nbData, rnew(n).Data(m,:), '-','lineWidth',.5,'color',[0 .7 0]);
% 	end
% 	for n=1:nbRepros
% 		plot(1:nbData-1, r(n).Data(m,2:end), '-','lineWidth',2,'color',[.8 0 0]);
% 	end
% 	set(gca,'xtick',[],'ytick',[]);
% 	xlabel('$t$','interpreter','latex','fontsize',18);
% 	ylabel(['$x_' num2str(m) '$'],'interpreter','latex','fontsize',18);
% 	axis(limAxes);
% end


%% Additional timeline velocity plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluate velocities
for n=1:nbSamples
	DataVel(:,(n-1)*nbData+1:n*nbData) = gradient(Data(:,(n-1)*nbData+1:n*nbData)) / model.dt;
end
Data = [Data; DataVel];
for n=1:nbRepros
	r(n).Data = [r(n).Data; gradient(r(n).Data) / model.dt];
end
for n=1:nbNewRepros
	rnew(n).Data = [rnew(n).Data; gradient(rnew(n).Data) / model.dt];
end
%Plot
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$'};
figure('position',[810 10 800 800],'color',[1 1 1]); 
for j=1:4
	subplot(5,1,j); hold on;
	if j<=model.nbVarPos
		limAxes = [1, nbData, min(Data(j,:))-4E0, max(Data(j,:))+4E0];
		msh=[]; x0=[];
		for t=1:nbData-1
			if size(msh,2)==0
				msh(:,1) = [t; model.Mu(j,qList(t))];
			end
			if t==nbData-1 || qList(t+1)~=qList(t)
				i = qList(t);
				msh(:,2) = [t+1; model.Mu(j,i)];
				sTmp = model.Sigma(j,j,qList(t))^.5;
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
		axis(limAxes);
	end
	for n=1:nbSamples
		plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[0 0 0]);
	end
	for n=1:nbNewRepros
		plot(rnew(n).Data(j,:), '-','linewidth',2,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	for n=1:nbRepros
		plot(r(n).Data(j,:), '-','linewidth',2,'color',[.8 0 0]);
	end
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
%Speed profile
subplot(5,1,5); hold on;
for n=1:nbSamples
	sp = sqrt(Data(3,(n-1)*nbData+1:n*nbData).^2 + Data(4,(n-1)*nbData+1:n*nbData).^2);
	plot(sp, '-','linewidth',.5,'color',[0 0 0]);
end
for n=1:nbNewRepros
	sp = sqrt(rnew(n).Data(3,:).^2 + rnew(n).Data(4,:).^2);
	plot(sp, '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
end
for n=1:nbRepros
	sp = sqrt(r(n).Data(3,:).^2 + r(n).Data(4,:).^2);
	plot(sp, '-','linewidth',1,'color',[.8 0 0]);
end
ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');

% print('-dpng','graphs/demo_batchLQR03.png');
pause;
close all;
