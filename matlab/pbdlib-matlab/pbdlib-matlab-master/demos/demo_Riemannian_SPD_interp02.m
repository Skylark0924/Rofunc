function demo_Riemannian_SPD_interp02
% Covariance interpolation on Riemannian manifold from a GMM with augmented covariances,
% and comparison with weighted product interpolation and Wasserstein interpolation
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Sylvain Calinon, https://calinon.ch/
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 6; %Number of states in the GMM
model.nbVar = 2; %Number of variables [x1,x2]
nbData = 20; %Length of each trajectory


% %% Generate random Gaussians
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model.Mu = rand(model.nbVar, model.nbStates) * 1E1;
% for i=1:model.nbStates
% 	xtmp = randn(model.nbVar,5) * 1E0;
% 	model.Sigma(:,:,i) = xtmp*xtmp' + eye(model.nbVar) * 1E-4;
% end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbSamples = 5; %Number of demonstrations
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data, s(n).Data]; 
end
%Learning 
model = init_GMM_kbins(Data, model, nbSamples);
%model = EM_GMM(Data, model);
for i=1:model.nbStates
	model.Sigma(:,:,i) = model.Sigma(:,:,i) + eye(model.nbVar) * 1E0;
end


%% Transformation to Gaussians with augmented covariances centered on zero 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
modelAugm = model;
modelAugm.nbVar = model.nbVar + 1;
modelAugm.Mu = zeros(modelAugm.nbVar, modelAugm.nbStates);
modelAugm.Sigma = zeros(modelAugm.nbVar, modelAugm.nbVar, modelAugm.nbStates);
for i=1:model.nbStates
	modelAugm.Sigma(:,:,i) = [model.Sigma(:,:,i)+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1];
% 	modelAugm.Sigma(:,:,i) = [eye(2)*1E1+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1];
end


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = linspace(0,1,nbData); %standard interpolation
% w = linspace(-.5,1.5,nbData); %extrapolation (exaggeration)

% wi = [linspace(1,0,nbData); linspace(0,1,nbData)]; %standard interpolation
% % wi = [linspace(1.5,-.5,nbData); linspace(-.5,1.5,nbData)]; %extrapolation (exaggeration)
% S = model2.Sigma(:,:,1);

SigmaAugm = zeros(modelAugm.nbVar, modelAugm.nbVar, nbData*(model.nbStates-1));
for i=2:modelAugm.nbStates
	for t=1:nbData		
% 		%Interpolation between more than 2 covariances can be computed in an iterative form
% 		nbIter = 10; %Number of iterations for the convergence of Riemannian estimate
% 		for n=1:nbIter
% 			W = zeros(model2.nbVar);
% 			for j=1:model.nbStates
% 				W = W + wi(j,t) * logmap(model2.Sigma(:,:,j), S);
% 			end
% 			S = expmap(W,S);
% 		end
% 		Sigma2(:,:,(i-2)*nbData+t) = S;
		
		%Interpolation between two covariances can be computed in closed form
		SigmaAugm(:,:,(i-2)*nbData+t) = expmap(w(t)*logmap(modelAugm.Sigma(:,:,i), modelAugm.Sigma(:,:,i-1)), modelAugm.Sigma(:,:,i-1));
	end
end

Mu = zeros(model.nbVar, nbData*(model.nbStates-1));
Sigma = zeros(model.nbVar, model.nbVar, nbData*(model.nbStates-1));
for t=1:nbData*(model.nbStates-1)
	beta = SigmaAugm(end,end,t);
	Mu(:,t) = SigmaAugm(end,1:end-1,t) ./ beta;
	Sigma(:,:,t) = SigmaAugm(1:end-1,1:end-1,t) - beta * Mu(:,t)*Mu(:,t)';
end


%% Weighted product interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];

% %Euclidean interpolation
% Mu2 = zeros(model.nbVar, nbData);
% Sigma2 = zeros(model.nbVar, model.nbVar, nbData);
% for i=2:model.nbStates
% 	for t=1:nbData
% 		for j=1:model.nbStates
% 			Mu2(:,t) = Mu2(:,t) + w(j,t) * model.Mu(:,j);
% 			Sigma2(:,:,t) = Sigma2(:,:,t) + w(j,t) * model.Sigma(:,:,j);
% 		end
% 	end
% end

% %Interpolation with products
% Mu2 = zeros(model.nbVar, nbData*(model.nbStates-1));
% Sigma2 = zeros(model.nbVar, model.nbVar, nbData*(model.nbStates-1));
% for i=2:model.nbStates
% 	for t=1:nbData
% 		tt = (i-2)*nbData + t;
% 		for j=1:2
% 			Mu2(:,tt) = Mu2(:,tt) + w(j,t) * model.Mu(:,i-2+j);
% 			Sigma2(:,:,tt) = Sigma2(:,:,tt) + w(j,t) * (model.Sigma(:,:,i-2+j) + model.Mu(:,i-2+j) * model.Mu(:,i-2+j)');
% 		end
% 		Sigma2(:,:,tt) = Sigma2(:,:,tt) - Mu2(:,tt) * Mu2(:,tt)';
% 	end
% end

%Interpolation with products
Mu2 = zeros(model.nbVar, nbData*(model.nbStates-1));
Sigma2 = zeros(model.nbVar, model.nbVar, nbData*(model.nbStates-1));
for i=2:model.nbStates
	for t=1:nbData
		tt = (i-2)*nbData + t;
		for j=1:2
			Sigma2(:,:,tt) = Sigma2(:,:,tt) + w(j,t) * inv(model.Sigma(:,:,i-2+j)); 
			Mu2(:,tt) = Mu2(:,tt) + w(j,t) * (model.Sigma(:,:,i-2+j) \ model.Mu(:,i-2+j));
		end
		Sigma2(:,:,tt) = inv(Sigma2(:,:,tt));
		Mu2(:,tt) = Sigma2(:,:,tt) * Mu2(:,tt);
	end
end


%% Wasserstein interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SigmaAugm = zeros(modelAugm.nbVar, modelAugm.nbVar, nbData*(model.nbStates-1));
for i=2:modelAugm.nbStates
	for t=1:nbData	
		%See e.g. eq.(2) of "Optimal Transport Mixing of Gaussian Texture Models"
		S1 = modelAugm.Sigma(:,:,i-1);
		S2 = modelAugm.Sigma(:,:,i);
		T = S2^.5 * (S2^.5 * S1 * S2^.5)^-.5 * S2^.5;
		S = w(1,t) .* eye(modelAugm.nbVar) + w(2,t) .* T;
		SigmaAugm(:,:,(i-2)*nbData+t) = S * S1 * S;
	end
end

Mu3 = zeros(model.nbVar, nbData*(model.nbStates-1));
Sigma3 = zeros(model.nbVar, model.nbVar, nbData*(model.nbStates-1));
for t=1:nbData*(model.nbStates-1)
	beta = SigmaAugm(end,end,t);
	Mu3(:,t) = SigmaAugm(end,1:end-1,t) ./ beta;
	Sigma3(:,:,t) = SigmaAugm(1:end-1,1:end-1,t) - beta * Mu3(:,t)*Mu3(:,t)';
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,950]); 

subplot(1,3,1); hold on; axis off; title('Geodesic interpolation');
plot(Mu(1,:), Mu(2,:), '-','linewidth',2,'color',[.8 0 0]);
plot(Mu(1,:), Mu(2,:), '.','markersize',15,'color',[.8 0 0]);
plotGMM(model.Mu, model.Sigma, [.5 .5 .5], .5);
plotGMM(Mu, Sigma, [.8 0 0], .02);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

subplot(1,3,2); hold on; axis off; title('Weighted product interpolation');
plot(Mu2(1,:), Mu2(2,:), '-','linewidth',2,'color',[0 .8 0]);
plot(Mu2(1,:), Mu2(2,:), '.','markersize',15,'color',[0 .8 0]);
plotGMM(model.Mu, model.Sigma, [.5 .5 .5], .5);
plotGMM(Mu2, Sigma2, [0 .8 0], .02);
% plotGMM(Mu2(:,nbData/2:nbData:end), Sigma2(:,:,nbData/2:nbData:end), [0 0 .8], .1);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

subplot(1,3,3); hold on; axis off; title('Wasserstein interpolation');
plot(Mu3(1,:), Mu3(2,:), '-','linewidth',2,'color',[0 0 .8]);
plot(Mu3(1,:), Mu3(2,:), '.','markersize',15,'color',[0 0 .8]);
plotGMM(model.Mu, model.Sigma, [.5 .5 .5], .5);
plotGMM(Mu3, Sigma3, [0 0 .8], .02);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%print('-dpng','graphs/demo_Riemannian_SPD_interp02.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
	X = S^.5 * expm(S^-.5 * U * S^-.5) * S^.5;
end

function U = logmap(X,S)
	N = size(X,3);
	for n = 1:N
	% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
	% 	U(:,:,n) = S * logm(S\X(:,:,n));
		[v,d] = eig(S\X(:,:,n));
		U(:,:,n) = S * v*diag(log(diag(d)))*v^-1;
	end
end