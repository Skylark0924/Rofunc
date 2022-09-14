function demo_AR_HSMM01
% Multivariate autoregressive (AR) model implemented as a hidden semi-Markov model with lognormal duration model 
% (simplified version by encoding the state duration after EM)
%
% Sylvain Calinon, 2019
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
model.nbStates = 2; %Number of hidden states in the HSMM
nbData = 50; %Length of each trajectory
nbSamples = 3; %Number of demonstrations
minSigmaPd = 1E-2; %Minimum variance of state duration (regularization term)
nbVar = 2; %Dimension of datapoint
nbHist = 2; %Length of time window


% %% Load handwriting data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% demos = [];
% load('data/2Dletters/C.mat');
% Data = [];
% for n=1:nbSamples
% 	s(n).x = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData+1)); %Resampling
% 	s(n).x = [(rand(1)-.5) .* 2 .* s(n).x(2,:); -s(n).x(1,:)];
% 	if n>1
% 		s(n).x(1,:) = s(n).x(1,:) + (s(n-1).x(1,end) - s(n).x(1,1));
% 	end
% 	%Transformation of data for AR representation
% 	Y = s(n).x(:,2:end);
% 	X = [];
% 	for t=1:nbData
% 		xtmp = [s(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(s(n).x(:,1),1,nbHist-t)]; %Without offset
% % 		xtmp = [s(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(s(n).x(:,1),1,nbHist-t), ones(nbVar,1)]; %With offset
% 		X = [X, xtmp(:)];
% 	end
% 	s(n).Data = [X; Y];
% 	s(n).nbData = size(s(n).Data,2);
% 	Data = [Data, s(n).Data]; 
% end


%% Generate continuous data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/C.mat');
for n=1:nbSamples
	stmp(n).x = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	stmp(n).x = [(rand(1)-.5) .* 2 .* stmp(n).x(2,:); -stmp(n).x(1,:)];
	if n>1
		stmp(n).x(1,:) = stmp(n).x(1,:) + (stmp(n-1).x(1,end) - stmp(n).x(1,1));
	end
end
s(1).x = [];
for n=1:nbSamples	
	s(1).x = [s(1).x, stmp(n).x];
end
nbData = nbData * nbSamples - 1;
nbSamples = 1;

Data = [];
for n=1:nbSamples	
	%Transformation of data for AR representation
	Y = s(n).x(:,2:end);
	X = [];
	for t=1:nbData
		xtmp = [s(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(s(n).x(:,1),1,nbHist-t)]; %Without offset
% 		xtmp = [s(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(s(n).x(:,1),1,nbHist-t), ones(nbVar,1)]; %With offset
		X = [X, xtmp(:)];
	end
	s(n).Data = [X; Y];
	s(n).nbData = size(s(n).Data,2);
	Data = [Data, s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kmeans(Data, model);
% model = init_GMM_kbins(Data(:,1:100), model, nbSamples);

% model.Mu = zeros(size(model.Mu));
% model.params_updateComp = [0,1,1,1];

% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

% %Left-right model initialization
% model.Trans = zeros(model.nbStates);
% for i=1:model.nbStates-1
% 	model.Trans(i,i) = 1 - model.nbStates ./ nbData;
% 	model.Trans(i,i+1) = model.nbStates ./ nbData;
% end
% model.Trans(model.nbStates,model.nbStates) = 1.0;
% model.StatesPriors = zeros(model.nbStates,1);
% model.StatesPriors(1) = 1;
% model.Priors = ones(model.nbStates,1);

%Cyclic model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1 - model.nbStates ./ nbData;
	model.Trans(i,i+1) = model.nbStates ./ nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1-(model.nbStates/nbData);
model.Trans(model.nbStates,1) = model.nbStates/nbData;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;
model.Priors = ones(model.nbStates,1);

[model, H] = EM_HMM(s, model);
%Removal of self-transition (for HSMM representation) and normalization
model.Trans = model.Trans - diag(diag(model.Trans)) + eye(model.nbStates)*realmin;
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);

%AR parameters estimation with Gaussian conditioning
in = 1:nbHist*nbVar;
out = nbHist*nbVar+1:(nbHist+1)*nbVar;

% in = 1:(nbHist+1)*nbVar;
% out = (nbHist+1)*nbVar+1:(nbHist+2)*nbVar;
for i=1:model.nbStates
% 	model.A(:,:,i) = [model.Sigma(out,in,i) / model.Sigma(in,in,i)];

	Atmp = model.Sigma(out,in,i) / model.Sigma(in,in,i);
	model.A(:,:,i) = [Atmp, model.Mu(out,i) - Atmp * model.Mu(in,i)]; 
% 	model.A(:,:,i) = model.Mu(out,i) / model.Mu(in,i);
 	
% % 	model.A(:,:,i) = Data(out,:) / Data(in,:);
% % 	model.A(:,:,i) = Data(out,:) * ((Data(in,:)' * Data(in,:)) \ Data(in,:)');
% 	model.A(:,:,i) = Data(out,:) * diag(H(i,:)) * Data(in,:)' / (Data(in,:) * diag(H(i,:)) * Data(in,:)' + eye(length(in)).*1E-8);	
end


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
% nbD = round(2 * nbData/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)
nbD = round(2 * 50/model.nbStates); %Number of maximum duration step to consider in the HSMM (2 is a safety factor)

%Precomputation of duration probabilities 
for i=1:model.nbStates
	model.Pd(i,:) = gaussPDF(log([1:nbD]), model.Mu_Pd(:,i), model.Sigma_Pd(:,:,i)); 
	%The rescaling formula below can be used to guarantee that the cumulated sum is one (to avoid the numerical issues)
	model.Pd(i,:) = model.Pd(i,:) ./ sum(model.Pd(i,:));
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


%% AR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbSamples
	x = kron(ones(1,nbHist), s(n).x(:,1)); %Matrix containing history for the last nbHist iterations
	for t=1:nbData
		r(n).Data(:,t) = x(:,1); %Log data
		for i=1:model.nbStates
			htmp(i) = gaussPDF(r(n).Data(:,t), model.Mu(out,i), model.Sigma(out,out,i));
		end
		htmp = htmp ./ sum(htmp);
% 		htmp = h(:,t); 

		xtmp = zeros(nbVar,1);
		for i=1:model.nbStates
% 			xtmp = xtmp + htmp(i) .* model.A(:,:,i) * x(:); %Update data history matrix (without offset)
			xtmp = xtmp + htmp(i) .* model.A(:,:,i) * [x(:); ones(1,1)]; %Update data history matrix (with offset)
		end
		x = [xtmp, x(:,1:end-1)];  
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,600],'color',[1 1 1]); 
clrmap = lines(model.nbStates);
%Spatial plot of the data
subplot(3,2,[1,3]); axis off; hold on; 
for n=1:nbSamples
	plot(s(n).Data(1,:), s(n).Data(2,:), '-', 'color', [.3 .3 .3]);
	plot(r(n).Data(1,:), r(n).Data(2,:), '-', 'color', [.8 0 0]);
end
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), clrmap(i,:), .6);
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


% print('-dpng','graphs/demo_AR_HSMM01.png'); 
pause;
close all;

% % Saving .txt files
% model.varnames{1} = 'x1';
% model.varnames{2} = 'x2';
% HSMMtoText(model, 'test2');

