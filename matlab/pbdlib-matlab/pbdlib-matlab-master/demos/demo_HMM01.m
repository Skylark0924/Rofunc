function demo_HMM01
% Hidden Markov model with single Gaussian as emission distribution. 
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
model.nbStates = 5; %Number of hidden states in the HMM
nbData = 50; %Length of each trajectory
nbSamples = 5; %Number of demonstrations
% model.params_diagRegFact = 1E-8;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
%nbSamples = length(demos);
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).nbData = size(s(n).Data,2);
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kmeans(Data, model);
% model = init_GMM_kbins(Data, model, nbSamples);

%Random initialization
model.Mu = rand(2,model.nbStates);
model.Sigma = repmat(eye(2)*100,[1,1,model.nbStates]);
model.Trans = rand(model.nbStates,model.nbStates);
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
model.StatesPriors = rand(model.nbStates,1);
model.StatesPriors = model.StatesPriors / sum(model.StatesPriors);

% %Uniform initialization
% model.Trans = ones(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = ones(model.nbStates,1);
% model.StatesPriors = model.StatesPriors / sum(model.StatesPriors);

%Left-right model initialization
model.Trans = zeros(model.nbStates);
for i=1:model.nbStates-1
	model.Trans(i,i) = 1 - (model.nbStates / nbData);
	model.Trans(i,i+1) = model.nbStates / nbData;
end
model.Trans(model.nbStates,model.nbStates) = 1.0;
model.StatesPriors = zeros(model.nbStates,1);
model.StatesPriors(1) = 1;

% %Adding tiny values can change the structure (if required) of a left-right model
% model.StatesPriors = model.StatesPriors + 1E-10;
% model.Trans = model.Trans + 1E-10;

%Parameters refinement with EM
[model,~,LL] = EM_HMM(s, model);

%Display of initial state distribution and transition probabilities
model.StatesPriors
model.Trans

%HMM likelihood computation based on ALPHA
Lmean = 0;
for n=1:nbSamples
	for i=1:model.nbStates
		s(n).B(i,:) = gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
	end
	%Forward variable ALPHA
	s(n).ALPHA(:,1) = model.StatesPriors .* s(n).B(:,1);
	for t=2:s(n).nbData
		s(n).ALPHA(:,t) = (s(n).ALPHA(:,t-1)'*model.Trans)' .* s(n).B(:,t); 
	end
	Lmean = Lmean + sum(s(n).ALPHA(:,end));
end
Lmean = Lmean / nbSamples

% %HMM likelihood computation based on ALPHA-BETA
% Lmean = 0;
% t = 20; %Any time step within the trajectory
% for n=1:nbSamples
% 	for i=1:model.nbStates
% 		s(n).B(i,:) = gaussPDF(s(n).Data, model.Mu(:,i), model.Sigma(:,:,i));
% 	end
% 	%Backward variable BETA 
% 	s(n).BETA(:,s(n).nbData) = ones(model.nbStates,1); %Rescaling
% 	for t=s(n).nbData-1:-1:1
% 		s(n).BETA(:,t) = model.Trans * (s(n).BETA(:,t+1) .* s(n).B(:,t+1));
% 	end
% 	Lmean = Lmean + sum(s(n).ALPHA(:,t).*s(n).BETA(:,t));
% end
% Lmean = Lmean / nbSamples


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,600],'color',[1 1 1]);
clrmap = lines(model.nbStates);
%Plot spatial data
subplot(1,2,1); axis off; hold on; 
for n=1:nbSamples
	plot(s(n).Data(1,:), s(n).Data(2,:), '-', 'linewidth', 1, 'color', [.3 .3 .3]);
end
for i=1:model.nbStates
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:), .6);
end
axis equal;
%Plot transition information
subplot(1,2,2); axis off; hold on; 
plotHMM(model.Trans, model.StatesPriors);
axis([-1 1 -1 1]*1.9); 

% figure; hold on;
% plot(LL)
% xlabel('iteration'); ylabel('log-likelihood');

%print('-dpng','graphs/demo_HMM01.png');
pause;
close all;
