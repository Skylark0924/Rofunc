function demo_GMM_augmSigma01
% Gaussian mixture model (GMM) parameters estimation with zero means and augmented covariances.
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
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
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 2; %Number of variables [x1,x2]
nbData = 200; %Length of each trajectory
nbSamples = 5; %Number of demonstrations


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
%nbSamples = length(demos);
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; 
end


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kmeans(Data, model);
%model = init_GMM_kbins(Data, model, nbSamples);

%Initialisation of model2
model2 = model;
model2.nbVar = model.nbVar+1;
model2.Mu = zeros(model2.nbVar, model2.nbStates);
%Random initialization
model2.Sigma = zeros(model2.nbVar, model2.nbVar, model2.nbStates);
for i=1:model.nbStates
	MuTmp = randn(model.nbVar,1);
	model2.Sigma(:,:,i) = [eye(model.nbVar)+MuTmp*MuTmp', MuTmp; MuTmp', 1];
end
%Initialization based on model previously trained
% model2.Sigma = zeros(model2.nbVar, model2.nbVar, model2.nbStates);
% for i=1:model.nbStates
% 	model2.Sigma(:,:,i) = [model.Sigma(:,:,i)+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1];
% end
model2.params_updateComp = [1,0,1]; %No update of Mu

%Parameters estimation
model0 = EM_GMM(Data, model);
model2 = EM_GMM([Data; ones(1,nbData*nbSamples)], model2);
for i=1:model.nbStates
	%model2.Sigma(:,:,i) = model2.Sigma(:,:,i) / model2.Sigma(end,end,i);
	model.Mu(:,i) = model2.Sigma(1:end-1,end,i);
	model.Sigma(:,:,i) = model2.Sigma(1:end-1,1:end-1,i) - model.Mu(:,i)*model.Mu(:,i)';
end

%Likelihoods
for i=1:model.nbStates
	L(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
	L2(i,:) = model2.Priors(i) * gaussPDF([Data; ones(1,nbData*nbSamples)], zeros(model.nbVar+1,1), model2.Sigma(:,:,i));
end
L = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
L2 = L2 ./ repmat(sum(L2,1)+realmin, model.nbStates, 1);
%norm(L-L2)
%return


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,700,500]); hold on; axis off;
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(model0.Mu, model0.Sigma, [.7 .7 .7],.5);
plotGMM(model.Mu, model.Sigma, [.8 0 0],.5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

figure('position',[720,10,700,500]); hold on; 
for i=1:model.nbStates
	plot(L(i,1:nbData),'-','linewidth',2,'color',[.5 .5 .5]);
	plot(L2(i,1:nbData),':','linewidth',2,'color',[0 0 0]);
end
xlabel('t'); ylabel('L');

%print('-dpng','graphs/demo_GMM_augmSigma01.png');
pause;
close all;
