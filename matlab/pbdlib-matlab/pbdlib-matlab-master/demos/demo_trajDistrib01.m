function demo_trajDistrib01
% Stochastic sampling with Gaussian trajectory distribution 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
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
nbVar = 2; %Number of variables [x1,x2]
nbData = 100; %Length of each trajectory
nbSamples = 3; %Number of demonstrations
nbRepros = 50; %Number of reproductions 
nbEigs = 5; %Number of principal eigencomponents to keep


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data(:,n) = reshape(s(n).Data, nbVar*nbData, 1); 
end


%% Compute normal trajectory distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 1;
model.Mu = mean(Data,2); 
model.Sigma = cov(Data');
%Keep only a few principal eigencomponents
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	[d,id] = sort(diag(D),'descend');
	model.D(:,:,i) = diag(d(1:nbEigs));
	model.V(:,:,i) = V(:,id(1:nbEigs));
	model.Sigma(:,:,i) = model.V(:,:,i) * model.D(:,:,i) * model.V(:,:,i)' + eye(nbVar*nbData)*0E-4; 
end


%% Stochastic sampling from normal trajectory distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data2 = model.V * model.D.^.5 * randn(nbEigs,nbRepros) + repmat(model.Mu,1,nbRepros);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1200,1000]); hold on; axis off;
for n=1:nbSamples
	plot(Data(1:2:end,n),Data(2:2:end,n),'-','linewidth',1,'color',[.5 .5 .5]);
end
for n=1:nbRepros
	plot(Data2(1:2:end,n),Data2(2:2:end,n),'-','linewidth',1,'color',[0 .7 0]);
end
plot(model.Mu(1:2:end),model.Mu(2:2:end),'-','linewidth',2,'color',[.8 0 0]);
for t=1:nbData
	id = (t-1)*2+1:t*2; 
	plotGMM(model.Mu(id), model.Sigma(id,id), [.8 0 0], .1);
end
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

figure('position',[1220,10,1000,1000]); hold on; axis off;
pcolor(abs(model.Sigma)); 
colormap(flipud(gray));
axis square; axis([1 nbData*nbVar 1 nbData*nbVar]); axis ij; shading flat;

%print('-dpng','graphs/demo_trajDistrib01.png');
pause;
close all;