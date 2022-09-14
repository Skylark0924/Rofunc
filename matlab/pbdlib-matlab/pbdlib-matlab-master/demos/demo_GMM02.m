function demo_GMM02
% GMM with different covariance structures.
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
nbData = 100; %Length of each trajectory
nbSamples = 5; %Number of demonstrations

%Parameters of the EM algorithm
nbMinSteps = 50; %Minimum number of iterations allowed
nbMaxSteps = 200; %Maximum number of iterations allowed
maxDiffLL = 1E-5; %Likelihood increase threshold to stop the algorithm
diagRegularizationFactor = 1E-2; %Regularization term is optional


% %% Load  data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('data/faithful.mat');
% Data = faithful';
% nbData = size(Data,2); 

%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/C.mat');
Data=[];
a=pi/3;
R = [cos(a) sin(a); -sin(a) cos(a)];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data, R*s(n).Data]; 
end

%Initialization
%model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased([repmat(1:nbData,1,nbSamples); Data], model);
model.Mu = model.Mu(2:end,:);
model.Sigma = model.Sigma(2:end,2:end,:);
nbData = nbData * nbSamples;


%% EM with isotropic covariances 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for nbIter=1:nbMaxSteps
	fprintf('.');
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		%Update Mu
		model.Mu(:,i) = Data * GAMMA2(i,:)';
		%Update Sigma
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = diag(diag(DataTmp * diag(GAMMA2(i,:)) * DataTmp')) + eye(size(Data,1)) * diagRegularizationFactor;
	end
	model.Sigma = repmat(eye(model.nbVar),[1 1 model.nbStates]) * mean(mean(mean(model.Sigma)));
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			break;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
m1 = model;


%% EM with diagonal covariances 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for nbIter=1:nbMaxSteps
	fprintf('.');
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		%Update Mu
		model.Mu(:,i) = Data * GAMMA2(i,:)';
		%Update Sigma
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = diag(diag(DataTmp * diag(GAMMA2(i,:)) * DataTmp')) + eye(size(Data,1)) * diagRegularizationFactor;
	end
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			break;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
m2 = model;


%% EM for full covariance matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m3,~,LL] = EM_GMM(Data, model);


%% EM with tied covariances 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for nbIter=1:nbMaxSteps
	fprintf('.');
	%E-step
	[L, GAMMA] = computeGamma(Data, model); %See 'computeGamma' function below
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / nbData;
		%Update Mu
		model.Mu(:,i) = Data * GAMMA2(i,:)';
		%Update Sigma
		DataTmp = Data - repmat(model.Mu(:,i),1,nbData);
		model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp' + eye(size(Data,1)) * diagRegularizationFactor;
	end
	model.Sigma = repmat(mean(model.Sigma,3), [1 1 model.nbStates]);
	%Compute average log-likelihood
	LL(nbIter) = sum(log(sum(L,1))) / nbData;
	%Stop the algorithm if EM converged (small change of LL)
	if nbIter>nbMinSteps
		if LL(nbIter)-LL(nbIter-1)<maxDiffLL || nbIter==nbMaxSteps-1
			disp(['EM converged after ' num2str(nbIter) ' iterations.']);
			break;
		end
	end
end
disp(['The maximum number of ' num2str(nbMaxSteps) ' EM iterations has been reached.']);
m4 = model;


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 16 5],'position',[10,10,1300,400]); 
%Isotropic covariance
subplot(1,4,1); hold on; axis off; title('Isotropic','fontsize',14);
plotGMM(m1.Mu, m1.Sigma, [.8 0 0], .5);
plot(Data(1,:),Data(2,:),'.','markersize',12,'color',[.5 .5 .5]);
axis([min(Data(1,:))-1E0 max(Data(1,:))+1E0 min(Data(2,:))-1E0 max(Data(2,:))+1E0]); axis equal; 
%Diagonal covariance
subplot(1,4,2); hold on; axis off; title('Diagonal','fontsize',14);
plotGMM(m2.Mu, m2.Sigma, [.8 0 0], .5);
plot(Data(1,:),Data(2,:),'.','markersize',12,'color',[.5 .5 .5]);
axis([min(Data(1,:))-1E0 max(Data(1,:))+1E0 min(Data(2,:))-1E0 max(Data(2,:))+1E0]); axis equal; 
%Full covariance
subplot(1,4,3); hold on; axis off; title('Full','fontsize',14);
plotGMM(m3.Mu, m3.Sigma, [.8 0 0], .5);
plot(Data(1,:),Data(2,:),'.','markersize',12,'color',[.5 .5 .5]);
axis([min(Data(1,:))-1E0 max(Data(1,:))+1E0 min(Data(2,:))-1E0 max(Data(2,:))+1E0]); axis equal; 
%Tied covariance
subplot(1,4,4); hold on; axis off; title('Tied','fontsize',14);
plotGMM(m4.Mu, m4.Sigma, [.8 0 0], .5);
plot(Data(1,:),Data(2,:),'.','markersize',12,'color',[.5 .5 .5]);
axis([min(Data(1,:))-1E0 max(Data(1,:))+1E0 min(Data(2,:))-1E0 max(Data(2,:))+1E0]); axis equal;

%print('-dpng','-r300','graphs/demo_GMM02.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L, GAMMA] = computeGamma(Data, model)
L = zeros(model.nbStates,size(Data,2));
for i=1:model.nbStates
	L(i,:) = model.Priors(i) * gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
end
GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
end
