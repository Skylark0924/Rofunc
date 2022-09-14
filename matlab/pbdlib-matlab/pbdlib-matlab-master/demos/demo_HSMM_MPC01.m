function demo_HSMM_MPC01
% Use of HSMM (with lognormal duration model) and batch LQR (with position only) as motion synthesis.
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
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
model.nbStates = 4; %Number of hidden states in the HSMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.rfactor = 1E-5;	%Control cost in LQR
model.dt = 0.01; %Time step duration
nbData = 200; %Length of each trajectory
nbSamples = 10; %Number of demonstrations
nbRepros = 5; %Number of reproductions in new situations
minSigmaPd = 2E-2; %Minimum variance of state duration (regularization term)
nbD = round(2.5*nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).nbData = size(s(n).Data,2);
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1-(model.nbStates/nbData);
	model.Trans(i,i+1) = model.nbStates/nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

[model, H] = EM_HMM(s, model);
%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);


% %% Set state duration manually (for HSMM representation)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Test with cyclic data
% model.Trans = [0, 1, 0; 0, 0, 1; 1, 0, 0];
% %model.Trans = [0, .5, .5; 1, 0, 0; 1, 0, 0];
% model.Mu_Pd = [10,30,50];
% for i=1:model.nbStates
% 	model.Sigma_Pd(1,1,i) = 1E1;
% end


%% Post-estimation of the state duration from data (for HSMM representation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:model.nbStates
	st(i).d=[];
end
[~,hmax] = max(H);
currState = hmax(1);
cnt = 1;
for t=1:length(hmax)
	if (hmax(t)==currState)
		cnt = cnt+1;
	else
		%st(currState).d = [st(currState).d cnt];
		st(currState).d = [st(currState).d log(cnt)];
		cnt = 1;
		currState = hmax(t);
	end
end
%st(currState).d = [st(currState).d cnt];
st(currState).d = [st(currState).d log(cnt)];

%Compute state duration as Gaussian distribution
for i=1:model.nbStates
	model.Mu_Pd(1,i) = mean(st(i).d);
	model.Sigma_Pd(1,1,i) = cov(st(i).d) + minSigmaPd;
end


%% Reconstruction of states probability sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Precomputation of duration probabilities 
for i=1:model.nbStates
	%model.Pd(i,:) = gaussPDF([0:nbD], model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)); 
	model.Pd(i,:) = gaussPDF(log([0:nbD]), model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)); 
	%The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
	%model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
end
%Reconstruction of states sequence 
h = zeros(model.nbStates,nbData);
for t=1:nbData
	for i=1:model.nbStates
		if t<=nbD
			h(i,t) = model.StatesPriors(i) * model.Pd(i,t);
		end
		for d=1:min(t-1,nbD)
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d);
		end
	end
end
h = h ./ repmat(sum(h,1)+realmin,model.nbStates,1);


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Dynamical System settings (discrete version), see Eq. (33)
A = kron([1, model.dt; 0, 1], eye(model.nbVarPos));
B = kron([0; model.dt], eye(model.nbVarPos));
C = kron([1, 0], eye(model.nbVarPos));
%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);

%Build CSx and CSu matrices for batch LQR, see Eq. (35)
CSu = zeros(model.nbVarPos*nbData, model.nbVarPos*(nbData-1));
CSx = kron(ones(nbData,1), [eye(model.nbVarPos) zeros(model.nbVarPos)]);
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos;
	CSx(id1,:) = CSx(id1,:) * A;
	id1 = (n-1)*model.nbVarPos+1:n*model.nbVarPos; 
	id2 = 1:(n-1)*model.nbVarPos;
	CSu(id1,id2) = C * M;
	M = [A*M(:,1:model.nbVarPos), M];
end

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q, see Eq. (27)
[~,qList] = max(h,[],1); %works also for nbStates=1

n=2;
qList = hmax((n-1)*nbData+1:n*nbData); %estimation from data (instead of HSMM parameters)

% figure; hold on;
% plot(qList,'r-');
% plot(hmax,'k:');
% %pause;
% %close all;
% %return;

MuQ = reshape(model.Mu(:,qList), model.nbVarPos*nbData, 1); 
SigmaQ = (kron(ones(nbData,1), eye(model.nbVarPos)) * reshape(model.Sigma(:,:,qList), model.nbVarPos, model.nbVarPos*nbData)) .* kron(eye(nbData), ones(model.nbVarPos));

%Set matrices to compute the damped weighted least squares estimate
CSuInvSigmaQ = CSu' / SigmaQ;
Rq = CSuInvSigmaQ * CSu + R;

%Reproductions
for n=1:nbRepros
	X = [Data(:,1)+randn(model.nbVarPos,1)*2E0; zeros(model.nbVarPos,1)]; 
	%X = [Data(:,1); zeros(model.nbVarPos,1)]; 
	rq = CSuInvSigmaQ * (MuQ-CSx*X);
	u = Rq \ rq; %Can also be computed with u = lscov(Rq, rq);
	r(n).Data = reshape(CSx*X+CSu*u, model.nbVarPos, nbData);
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 20 24],'position',[10,10,600,700],'color',[1 1 1]); 
clrmap = lines(model.nbStates);

%Spatial plot of the data
subplot(5,2,[1,3]); axis off; hold on; 
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.4 .4 .4]);
end
for i=1:model.nbStates
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), .8);
end
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis tight; axis equal;

%HSMM transition and state duration plot
subplot(5,2,[2,4]); axis off; hold on; 
plotHSMM(model.Trans, model.StatesPriors, model.Pd);
axis([-1 1 -1 1]*1.9);

%Timeline plot of the state sequence probabilities
subplot(5,2,[5,6]); hold on;
for i=1:model.nbStates
	patch([1, 1:nbData, nbData], [0, h(i,:), 0], clrmap(i,:), ...
		'linewidth', 1.5, 'EdgeColor', max(clrmap(i,:)-0.2,0), 'facealpha', .7, 'edgealpha', .7);
end
%set(gca,'xtick',[10:10:nbData],'fontsize',8); 
set(gca,'xtick',[],'ytick',[]);
axis([1 nbData 0 1.1]);
ylabel('$h_i$','interpreter','latex','fontsize',18);

%Timeline plot of the movement
for m=1:model.nbVarPos
	limAxes = [1, nbData, min(Data(m,:))-4E0, max(Data(m,:))+4E0];
	if m==1
		subplot(5,2,[7,8]); hold on;
	else
		subplot(5,2,[9,10]); hold on;
	end
	for n=1:nbSamples
		plot(1:nbData, Data(m,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.4 .4 .4]);
	end
	msh=[]; x0=[];
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; model.Mu(m,qList(t))];
		end
		if t==nbData-1 || qList(t+1)~=qList(t)
			i = qList(t);
			msh(:,2) = [t+1; model.Mu(m,i)];
			sTmp = model.Sigma(m,m,qList(t))^.5;
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), clrmap(i,:),'edgecolor',max(clrmap(i,:)-0.2,0),'facealpha', .7, 'edgealpha', .7);
			plot(msh(1,:), msh(2,:), '-','linewidth',1.5,'color',[0 0 0]);
			plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
			x0 = [x0 msh];
			msh=[];
		end
	end
	for n=1:nbRepros
		plot(1:nbData-1, r(n).Data(m,2:end), '-','lineWidth',2,'color',[.8 0 0]);
	end
	set(gca,'xtick',[],'ytick',[]);
	if m==2
		xlabel('$t$','interpreter','latex','fontsize',18);
	end
	ylabel(['$q_' num2str(m) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
end
% print('-dpng','-r150','graphs/HSMM-simple01.png'); 


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
%Plot
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$'};
figure('position',[610 10 600 700],'color',[1 1 1]); 
for j=1:4
subplot(5,1,j); hold on;
for n=1:nbSamples
	plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[0 0 0]);
end
for n=1:nbRepros
	plot(r(n).Data(j,:), '-','linewidth',1,'color',[.8 0 0]);
end
ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
%Speed profile
subplot(5,1,5); hold on;
for n=1:nbSamples
	sp = sqrt(Data(3,(n-1)*nbData+1:n*nbData).^2 + Data(4,(n-1)*nbData+1:n*nbData).^2);
	plot(sp, '-','linewidth',.5,'color',[0 0 0]);
end
for n=1:nbRepros
	sp = sqrt(r(n).Data(3,:).^2 + r(n).Data(4,:).^2);
	plot(sp, '-','linewidth',1,'color',[.8 0 0]);
end
ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');

pause;
close all;