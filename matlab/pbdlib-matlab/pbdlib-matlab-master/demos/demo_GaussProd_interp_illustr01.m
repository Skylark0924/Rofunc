function demo_GaussProd_interp_illustr01
% Smooth transition between hierarchy constraints by relying on SPD geodesics
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
nbVar = 2; %Number of variables
nbData = 6; % Number of interpolation steps


%% Set GMM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Transition 1
m(1).Mu = [2.5 2.5; 0 0];
d = [5; -3];
m(1).Sigma(:,:,1) = d*d' + eye(2)*1E-4;
m(1).Sigma(:,:,2) = eye(2)*2E-1;

%Transition 2
m(2).Mu = [-.5 -.5; 0 0];
d = [5; 2];
m(2).Sigma(:,:,1) = eye(2)*2E-1;
m(2).Sigma(:,:,2) = d*d' + eye(2)*1E-4;


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];
for n=1:2
	mg(n).Mu = interp1([0,1], m(n).Mu', w(2,:))';
	mg(n).Sigma = zeros(nbVar, nbVar, nbData);
	for t=1:nbData
		%Interpolation between two covariances can be computed in closed form
		mg(n).Sigma(:,:,t) = expmap(w(2,t) * logmap(m(n).Sigma(:,:,2), m(n).Sigma(:,:,1)), m(n).Sigma(:,:,1));
	end
end


%% Product of Gaussians
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:nbData
	SigmaTmp = zeros(nbVar);
	MuTmp = zeros(nbVar,1);
	for n=1:2
		SigmaTmp = SigmaTmp + inv(mg(n).Sigma(:,:,t));
		MuTmp = MuTmp + mg(n).Sigma(:,:,t) \ mg(n).Mu(:,t);
	end
	SigmaP(:,:,t) = inv(SigmaTmp);
	MuP(:,t) = SigmaP(:,:,t) * MuTmp;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,400,800]);
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
set(gca,'LooseInset',[0,0,0,0]);
colormap(repmat(linspace(1,.2,64),3,1)');

rng = [-1.5 3.5 -1 2];
nbp = 10;
[X,Y] = meshgrid(linspace(rng(1),rng(2),nbp), linspace(rng(3),rng(4),nbp));
p = [reshape(X,1,nbp^2); reshape(Y,1,nbp^2)];

for t=1:nbData
	subaxis(nbData,1,t,'Spacing',0.01); hold on; axis off;
	c = zeros(nbp^2,1);
	%for i=1:nbStates
	%	dtmp = (p - repmat(model(n).Mu(:,i),1,nbp^2))';
	%	c = c + sum((dtmp/model(n).Sigma(:,:,i)).*dtmp, 2);
	%end
	dtmp = (p - repmat(MuP(:,t),1,nbp^2))';
	c = c + sum((dtmp/SigmaP(:,:,t)).*dtmp, 2);
	pcolor(X,Y,reshape(c,nbp,nbp)); 
	shading interp;
	plotGMM(mg(1).Mu(:,t), mg(1).Sigma(:,:,t), [0 .8 0], .4);
	plotGMM(mg(2).Mu(:,t), mg(2).Sigma(:,:,t), [0 .8 0], .4);
	plotGMM(MuP(:,t), SigmaP(:,:,t), [.8 0 0], .7);
	
	plot([rng(1) rng(2) rng(2) rng(1) rng(1)], [rng(3) rng(3) rng(4) rng(4) rng(3)], 'k-','linewidth',3);
	axis equal; axis(rng); 
end
% print('-dpng','graphs/GaussProd_interp_illustr01.png');


% %% Anim
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,500,300]); hold on; axis off;
% set(0,'DefaultAxesLooseInset',[0,0,0,0]);
% set(gca,'LooseInset',[.05,.05,.05,.05]);
% colormap(repmat(linspace(1,.2,64),3,1)');
% rng = [-1.5 3.5 -1 2];
% nbp = 10;
% [X,Y] = meshgrid(linspace(rng(1),rng(2),nbp), linspace(rng(3),rng(4),nbp));
% p = [reshape(X,1,nbp^2); reshape(Y,1,nbp^2)];
% lst = [1:nbData, nbData:-1:1];
% for f=52:length(lst)
% 	t = lst(f);
% 	%subplot(1,size(nList,2),n); hold on; box on;
% 	c = zeros(nbp^2,1);
% 	%for i=1:nbStates
% 	%	dtmp = (p - repmat(model(n).Mu(:,i),1,nbp^2))';
% 	%	c = c + sum((dtmp/model(n).Sigma(:,:,i)).*dtmp, 2);
% 	%end
% 	dtmp = (p - repmat(MuP(:,t),1,nbp^2))';
% 	c = c + sum((dtmp/SigmaP(:,:,t)).*dtmp, 2);
% 	h = pcolor(X,Y,reshape(c,nbp,nbp)); 
% 	shading interp;
% 	h = [h plotGMM(mg(1).Mu(:,t), mg(1).Sigma(:,:,t), [0 .8 0], .4)];
% 	h = [h plotGMM(mg(2).Mu(:,t), mg(2).Sigma(:,:,t), [0 .8 0], .4)];
% 	h = [h plotGMM(MuP(:,t), SigmaP(:,:,t), [.8 0 0], .7)];
% 	h = [h plot([rng(1) rng(2) rng(2) rng(1) rng(1)], [rng(3) rng(3) rng(4) rng(4) rng(3)], 'k-','linewidth',3)];
% 	axis equal; axis(rng); 
% 	pause(.1);
% 	print('-dpng',['graphs/animGaussProd/' num2str(f,'%.3d') '.png']);
% 	pause(.1);
% 	delete(h);
% end

pause;
close all;
end

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function S = expmap(W,S)
	[V,D] = eig(S\W);
	S = S * V * diag(exp(diag(D))) * V^-1;
end

function U = logmap(X,S)
	N = size(X,3);
	for n = 1:N
		[V,D] = eig(S\X(:,:,n));
		U(:,:,n) = S * V * diag(log(diag(D))) * V^-1;
	end
end