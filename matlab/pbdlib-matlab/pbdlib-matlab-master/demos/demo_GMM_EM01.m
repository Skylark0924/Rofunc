function demo_GMM_EM01
% Illustration of the problem of local optima in EM for GMM parameters estimation.
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
model.nbStates = 2; %Number of states in the GMM
model.nbVar = 2; %Number of variables [x1,x2]

%% Load  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/faithful.mat');
Data = faithful';
nbData = size(Data,2); 

%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model.Priors = ones(1,model.nbStates)/model.nbStates;

%Good initialization point
% model.Mu = [2 4.1; 90 50];
% model.Sigma = repmat(diag([.2 50]),[1 1 model.nbStates]); %eye(model.nbVar)

%Bad initialization point
model.Mu = [Data(:,26) Data(:,47)];
model.Sigma(:,:,1) = diag([.2 50]);
model.Sigma(:,:,2) = diag([5E-2 2.5]);

%[model,~,LL] = EM_GMM(Data, model);

figure('position',[10,10,700,500]); hold on; %axis off;
plot(Data(1,:),Data(2,:),'.','markersize',12,'color',[.5 .5 .5]);
xlabel('$x_1$','interpreter','latex','fontsize',14);
ylabel('$x_2$','interpreter','latex','fontsize',14);


%% EM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Parameters of the EM algorithm
nbMinSteps = 50; %Minimum number of iterations allowed
nbMaxSteps = 200; %Maximum number of iterations allowed
maxDiffLL = 1E-5; %Likelihood increase threshold to stop the algorithm

diagRegularizationFactor = 1E-2; %Regularization term is optional
hp=[];

for nbIter=1:nbMaxSteps
	fprintf('.');
	delete(hp)
	hp = plotGMM(model.Mu, model.Sigma, [.8 0 0],.2);
	drawnow;
	if mod(nbIter,5)==1
		print('-dpng', ['graphs/demo_GMM_EM' num2str(((nbIter-1)/5)+1,'%.2d') 'b.png']);
	end

	pause(0.01);
	if nbIter==1
		%pause
	end
	
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


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for t=1:nbData
% 	h=plot(Data(1,t),Data(2,t),'.','markersize',18,'color',[1 .5 .5]);
% 	t
% 	pause
% 	delete(h)
% end
% plotGMM(model.Mu, model.Sigma, [.8 0 0],.2);
%plot(model.Mu(1,:),model.Mu(2,:),'o','color',[.8 0 0]);
%axis equal; 
%set(gca,'Xtick',[]); set(gca,'Ytick',[]);

figure('position',[710,10,400,500]); hold on;
plot(LL,'k.');
axis([1 length(LL) -5 -4]);

%print('-dpng','graphs/demo_GMM_EM01.png');
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
