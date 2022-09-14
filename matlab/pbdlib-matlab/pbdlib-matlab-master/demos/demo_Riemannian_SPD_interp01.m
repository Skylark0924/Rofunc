function demo_Riemannian_SPD_interp01
% Covariance interpolation on Riemannian manifold (comparison with linear interpolation, 
% with Euclidean interpolation on Cholesky decomposition, and with Wasserstein interpolation)
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
model.nbVar = 2; %Number of variables
model.nbStates = 2; %Number of states
nbData = 20; %Number of interpolations
% nbIter = 5; %Number of iteration for the Gauss Newton algorithm


%% Gaussians parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Mu(:,1) = [0; 0];
model.Mu(:,2) = [1; 0];

d1 = [.02; .3];
model.Sigma(:,:,1) = d1*d1' + eye(model.nbVar)*1E-4;

d2 = [.4; .1];
model.Sigma(:,:,2) = d2*d2' + eye(model.nbVar)*1E-2;

% %[R,~] = qr(randn(model.nbVar));
% [R,~] = qr([1,2;3,4]);
% model.Sigma(:,:,2) = R * model.Sigma(:,:,1) * R';


%% Linear interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];
ml.Mu = interp1([0,1], model.Mu', w(2,:))';
ml.Sigma = zeros(model.nbVar,model.nbVar,nbData);
for t=1:nbData
	for i=1:model.nbStates
		ml.Sigma(:,:,t) = ml.Sigma(:,:,t) + w(i,t) * model.Sigma(:,:,i);
	end
	%Analysis
	[ml.V(:,:,t), ml.D(:,:,t)] = eigs(ml.Sigma(:,:,t));
	ml.Sdet(t) = det(ml.Sigma(:,:,t)); 
	ml.Stra(t) = trace(ml.Sigma(:,:,t));
end


%% Cholesky interpolation (optionally, with logm and expm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:model.nbStates
	Q(:,:,i) = chol(model.Sigma(:,:,i),'lower');
% 	Q(:,:,i) = logm(chol(model.Sigma(:,:,i),'lower'));
end
mc.Mu = interp1([0,1], model.Mu', w(2,:))';
mc.Sigma = zeros(model.nbVar,model.nbVar,nbData);
mc.Q = zeros(model.nbVar,model.nbVar,nbData);
for t=1:nbData
	for i=1:model.nbStates
		mc.Q(:,:,t) = mc.Q(:,:,t) + w(i,t) * Q(:,:,i);
	end
	mc.Sigma(:,:,t) = mc.Q(:,:,t) * mc.Q(:,:,t)';
% 	mc.Sigma(:,:,t) = expm(mc.Q(:,:,t)) * expm(mc.Q(:,:,t))';
	%Analysis
	[mc.V(:,:,t), mc.D(:,:,t)] = eigs(mc.Sigma(:,:,t));
	mc.Sdet(t) = det(mc.Sigma(:,:,t)); 
	mc.Stra(t) = trace(mc.Sigma(:,:,t));
end


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mg.Mu = interp1([0,1], model.Mu', w(2,:))';
mg.Sigma = zeros(model.nbVar,model.nbVar,nbData);
% S = model.Sigma(:,:,1);
for t=1:nbData
	
% 	%Interpolation between more than 2 covariances can be computed in an iterative form
% 	for n=1:nbIter
% 		W = zeros(model.nbVar);
% 		for i=1:model.nbStates
% 			W = W + w(i,t) * logmap(model.Sigma(:,:,i), S);
% 		end
% 		S = expmap(W,S);
% 	end
% 	Sigma(:,:,t) = S;

	%Interpolation between two covariances can be computed in closed form
	mg.Sigma(:,:,t) = expmap(w(2,t)*logmap(model.Sigma(:,:,2), model.Sigma(:,:,1)), model.Sigma(:,:,1));
	
	%Analysis
	[mg.V(:,:,t), mg.D(:,:,t)] = eigs(mg.Sigma(:,:,t));
	mg.Sdet(t) = det(mg.Sigma(:,:,t));
	mg.Stra(t) = trace(mg.Sigma(:,:,t));
end


%% Wasserstein interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mw.Mu = interp1([0,1], model.Mu', w(2,:))';
mw.Sigma = zeros(model.nbVar,model.nbVar,nbData);
for t=1:nbData
	
	%See e.g. eq.(2) of "Optimal Transport Mixing of Gaussian Texture Models"
	S1 = model.Sigma(:,:,1);
	S2 = model.Sigma(:,:,2);
	T = S2^.5 * (S2^.5 * S1 * S2^.5)^-.5 * S2^.5;
	S = w(1,t) .* eye(model.nbVar) + w(2,t) .* T;
	mw.Sigma(:,:,t) = S * S1 * S;
	
	%Analysis
	[mw.V(:,:,t), mw.D(:,:,t)] = eigs(mw.Sigma(:,:,t));
	mw.Sdet(t) = det(mw.Sigma(:,:,t));
	mw.Stra(t) = trace(mw.Sigma(:,:,t));
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2400,1000]); 

%Plot determinant for linear interpolation
subplot(2,4,1); hold on; title('Linear interpolation');
plot(w(2,:),ml.Sdet,'-','color',[.8 0 0]);
%plot(w(2,:),ml.Stra,':','color',[.8 0 0]);
axis([0, 1, min(ml.Sdet)-.001, max(ml.Sdet)+.001]);
xlabel('t'); ylabel('det(S)');

%Plot determinant for Cholesky interpolation
subplot(2,4,2); hold on; title('Cholesky interpolation'); %(with logm and expm)
plot(w(2,:),mc.Sdet,'-','color',[0 .8 0]);
%plot(w(2,:),mc.Stra,':','color',[0 .8 0]);
axis([0, 1, min(mc.Sdet)-.001, max(mc.Sdet)+.001]);
xlabel('t'); ylabel('det(S)');

%Plot determinant for geodesic interpolation
subplot(2,4,3); hold on; title('Geodesic interpolation');
plot(w(2,:),mg.Sdet,'-','color',[0 0 .8]);
%plot(w(2,:),mg.Stra,':','color',[0 0 .8]);
axis([0, 1, min(mg.Sdet)-.001, max(mg.Sdet)+.001]);
xlabel('t'); ylabel('det(S)');

%Plot determinant for Wasserstein interpolation
subplot(2,4,4); hold on; title('Wasserstein interpolation');
plot(w(2,:),mw.Sdet,'-','color',[.8 .8 0]);
%plot(w(2,:),mw.Stra,':','color',[.8 .8 0]);
axis([0, 1, min(mw.Sdet)-.001, max(mw.Sdet)+.001]);
xlabel('t'); ylabel('det(S)');

%Plot linear interpolation
subplot(2,4,5); hold on; axis off; 
plotGMM(model.Mu, model.Sigma, [0 0 0]);
plotGMM(ml.Mu, ml.Sigma, [.8 0 0], .1);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%Plot Cholesky interpolation
subplot(2,4,6); hold on; axis off; 
plotGMM(model.Mu, model.Sigma, [0 0 0]);
plotGMM(mc.Mu, mc.Sigma, [0 .8 0], .1);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%Plot geodesic interpolation
subplot(2,4,7); hold on; axis off; 
plotGMM(model.Mu, model.Sigma, [0 0 0]);
plotGMM(mg.Mu, mg.Sigma, [0 0 .8], .1);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%Plot Wasserstein interpolation
subplot(2,4,8); hold on; axis off; 
plotGMM(model.Mu, model.Sigma, [0 0 0]);
plotGMM(mw.Mu, mw.Sigma, [.8 .8 0], .1);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%print('-dpng','graphs/demo_Riemannian_SPD_interp01.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = expmap(W,S)
% 	S = S^.5 * expm(S^-.5 * W * S^-.5) * S^.5;
% 	S = S * expm(S\W);
	[V,D] = eig(S\W);
	S = S * V * diag(exp(diag(D))) * V^-1;
end

function U = logmap(X,S)
	N = size(X,3);
	for n = 1:N
	% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
	% 	U(:,:,n) = S * logm(S\X(:,:,n));
		[V,D] = eig(S\X(:,:,n));
		U(:,:,n) = S * V * diag(log(diag(D))) * V^-1;

	% 	[U,D,V] = svd(S\X(:,:,n));
	% 	U(:,:,n) = S * U * diag(log(diag(D))) * V';
	end
end