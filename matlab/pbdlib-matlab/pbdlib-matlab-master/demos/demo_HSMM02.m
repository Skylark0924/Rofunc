function demo_HSMM02
% Variable duration model implemented as a hidden semi-Markov model with lognormal duration model
% (simplified version by encoding the state duration after EM).
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
model.nbStates = 6; %Number of hidden states in the HSMM
nbData = 100; %Length of each trajectory
nbSamples = 10; %Number of demonstrations
minSigmaPd = 1E-2; %Minimum variance of state duration (regularization term)


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
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
		st(currState).d = [st(currState).d log(cnt)];
		cnt = 1;
		currState = hmax(t);
	end
end
st(currState).d = [st(currState).d log(cnt)];

%Compute state duration as Gaussian distribution
for i=1:model.nbStates
	model.Mu_Pd(1,i) = mean(st(i).d);
	model.Sigma_Pd(1,1,i) = cov(st(i).d) + minSigmaPd;
end

% %dm=P(d) for each state can be computed with:
% rho = (nbData - sum(squeeze(model.Mu_Pd))) / sum(squeeze(model.Sigma_Pd));
% dm = squeeze(model.Mu_Pd) + (squeeze(model.Sigma_Pd)' * rho);

% model.Sigma_Pd(1,1,2:3) = 1E-1;


%% Reconstruction of states probability sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbD = round(2 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)

%Precomputation of duration probabilities 
for i=1:model.nbStates
	model.Pd(i,:) = gaussPDF(log([1:nbD]), model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)); 
	%The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
	model.Pd(i,:) = model.Pd(i,:) / sum(model.Pd(i,:));
end


%Slow reconstruction of states sequence based on standard computation
%(in the iteration, a scaling factor c is used to avoid numerical underflow issues in HSMM, see Levinson'1986) 
h = zeros(model.nbStates,nbData);
c = zeros(nbData,1); %scaling factor to avoid numerical issues
c(1) = 1; %Initialization of scaling factor
for t=1:nbData
	for i=1:model.nbStates
		if t<=nbD
% 			oTmp = 1; %Observation probability for generative purpose
			oTmp = prod(c(1:t) .* gaussPDF(s(1).Data(:,1:t), model.Mu(:,i), model.Sigma(:,:,i))'); %Observation probability for standard HSMM
			h(i,t) = model.StatesPriors(i) * model.Pd(i,t) * oTmp;
		end
		for d=1:min(t-1,nbD)
% 			oTmp = 1; %Observation probability for generative purpose
			oTmp = prod(c(t-d+1:t) .* gaussPDF(s(1).Data(:,t-d+1:t), model.Mu(:,i), model.Sigma(:,:,i))'); %Observation probability for standard HSMM	
			h(i,t) = h(i,t) + h(:,t-d)' * model.Trans(:,i) * model.Pd(i,d) * oTmp;
		end
	end
	c(t+1) = 1/sum(h(:,t)+realmin); %Update of scaling factor
end
h = h ./ repmat(sum(h,1),model.nbStates,1);


% %Fast reconstruction of sequence for HSMM (version based on position and duration information)
% h = zeros(model.nbStates,nbData);
% [bmx, ALPHA, S, h(:,1)] = hsmm_fwd_init_hsum(s(1).Data(:,1), model);
% for t=2:nbData
% 	[bmx, ALPHA, S, h(:,t)] = hsmm_fwd_step_hsum(s(1).Data(:,t), model, bmx, ALPHA, S);
% end


% %Fast reconstruction of sequence for HSMM (version based on only duration information)
% h = zeros(model.nbStates,nbData);
% [ALPHA, S, h(:,1)] = hsmm_fwd_init_ts(model);
% for t=2:nbData
% 	[ALPHA, S, h(:,t)] = hsmm_fwd_step_ts(model, ALPHA, S);
% end
% h = h ./ repmat(sum(h,1),model.nbStates,1);


% %Manual reconstruction of sequence for HSMM based on stochastic sampling 
% nbSt=0; currTime=0; iList=[];
% h = zeros(model.nbStates,nbData);
% while currTime<nbData
% 	nbSt = nbSt+1;
% 	if nbSt==1
% 		[~,iList(1)] = max(model.StatesPriors.*rand(model.nbStates,1));
% 		h1 = ones(1,nbData);
% 	else
% 		h1 = [zeros(1,currTime), cumsum(model.Pd(iList(end-1),:)), ones(1,max(nbData-currTime-nbD,0))];
% 		currTime = currTime + round(model.Mu_Pd(1,iList(end-1)));
% 	end
% 	h2 = [ones(1,currTime), 1-cumsum(model.Pd(iList(end),:)), zeros(1,max(nbData-currTime-nbD,0))];
% 	h(iList(end),:) = h(iList(end),:) + min([h1(1:nbData); h2(1:nbData)]);
% 	[~,iList(end+1)] = max(model.Trans(iList(end),:).*rand(1,model.nbStates));
% end
% h = h ./ repmat(sum(h,1),model.nbStates,1);


% %% Plots
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('PaperPosition',[0 0 18 4.5],'position',[10,10,1300,450],'color',[1 1 1]); 
% %xx = round(linspace(1,64,model.nbStates));
% %clrmap = colormap('jet')*0.5;
% %clrmap = min(clrmap(xx,:),.9);
% clrmap = lines(model.nbStates);
% 
% %Spatial plot of the data
% subplot(2,4,[1,5]); hold on;
% for i=1:model.nbStates
% 	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), .6);
% end
% plot(Data(1,:), Data(2,:), '.', 'color', [.6 .6 .6]);
% %xlabel('$x_1$','fontsize',14,'interpreter','latex'); ylabel('$x_2$','fontsize',14,'interpreter','latex');
% %axis equal; axis square;
% axis tight; axis off;
% 
% %Timeline plot of the duration probabilities
% subplot(2,4,[2:4]); hold on;
% for i=1:model.nbStates
% 	yTmp = model.Pd(i,:) / max(model.Pd(i,:));
% 	patch([1, 1:nbD, nbD], [0, yTmp, 0], clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6);
% 	plot(1:nbD, yTmp, 'linewidth', 2, 'color', clrmap(i,:));
% end
% set(gca,'xtick',[10:10:nbD],'fontsize',8); axis([1 nbData 0 1]);
% ylabel('$Pd/Pd_{max}$','fontsize',16,'interpreter','latex');
% 
% %Timeline plot of the state sequence probabilities
% subplot(2,4,[6:8]); hold on;
% for i=1:model.nbStates
% 	patch([1, 1:nbData, nbData], [0, h(i,:), 0], clrmap(i,:), 'EdgeColor', 'none', 'facealpha', .6);
% 	plot(1:nbData, h(i,:), 'linewidth', 2, 'color', clrmap(i,:));
% end
% set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData 0 1]);
% xlabel('$t$','fontsize',16,'interpreter','latex'); 
% ylabel('$h$','fontsize',16,'interpreter','latex');
% 
% %% Plot transition graph
% figure('color',[1 1 1]); hold on; axis off;
% plotHSMM(model.Trans, model.StatesPriors, model.Pd);

% %% Additional plot 
% figure('PaperPosition',[0 0 36 12],'position',[10,10,1300,400],'color',[1 1 1]); 
% clrmap = lines(model.nbStates);
% %Spatial plot of the data
% subplot(1,4,1); axis off; hold on; 
% for i=1:model.nbStates
% 	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), 1);
% end
% plot(Data(1,:), Data(2,:), '.', 'color', [.3 .3 .3]);
% axis tight; axis equal;
% %GMM
% subplot(1,4,2); axis off; hold on; title('GMM','fontsize',20);
% %Plot nodes
% graphRadius = 1;
% nodeRadius = .2;
% nodePts = 40;
% nodeAngle = linspace(pi,2*pi+pi,model.nbStates+1);
% %nodeAngle = linspace(pi/2,2*pi+pi/2,model.nbStates+1);
% for i=1:model.nbStates
% 	nodePos(:,i) = [cos(nodeAngle(i)); sin(nodeAngle(i))] * graphRadius;
% end
% for i=1:model.nbStates
% 	a = linspace(0,2*pi,nodePts);
% 	meshTmp = [cos(a); sin(a)] * nodeRadius + repmat(nodePos(:,i),1,nodePts);
% 	patch(meshTmp(1,:), meshTmp(2,:), clrmap(i,:),'edgecolor',clrmap(i,:)*0.5, 'facealpha', 1,'edgealpha', 1);
% 	text(nodePos(1,i),nodePos(2,i),num2str(i),'HorizontalAlignment','center','FontWeight','bold','fontsize',20);
% end
% axis equal; axis([-1 1 -1 1]*1.9); 
% %HMM
% subplot(1,4,3); axis off; hold on; title('HMM','fontsize',20);
% plotHMM(model.Trans, model.StatesPriors);
% axis([-1 1 -1 1]*1.9); 
% %HSMM
% subplot(1,4,4); axis off; hold on; title('HSMM','fontsize',20); 
% plotHSMM(model.Trans, model.StatesPriors, model.Pd);
% axis([-1 1 -1 1]*1.9);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,600],'color',[1 1 1]); 
clrmap = lines(model.nbStates);
%Spatial plot of the data
subplot(3,2,[1,3]); axis off; hold on; 
plot(Data(1,:), Data(2,:), '.', 'color', [.3 .3 .3]);
for i=1:model.nbStates
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), .6);
end
axis tight; axis equal;
%HSMM transition and state duration plot
subplot(3,2,[2,4]); axis off; hold on; 
plotHSMM(model.Trans, model.StatesPriors, model.Pd);
axis([-1 1 -1 1]*1.9);
%Timeline plot of the state sequence probabilities
subplot(3,2,[5,6]); hold on;
for i=1:model.nbStates
	patch([1, 1:nbData, nbData], [0, h(i,:), 0], clrmap(i,:), ...
		'linewidth', 2, 'EdgeColor', max(clrmap(i,:)-0.2,0), 'facealpha', .6, 'edgealpha', .6);
end
set(gca,'xtick',[10:10:nbData],'fontsize',8); axis([1 nbData 0 1.1]);
xlabel('t','fontsize',16); 
ylabel('h_i','fontsize',16);

%print('-dpng','-r300','graphs/demo_HSMM02.png'); 
pause;
close all;
