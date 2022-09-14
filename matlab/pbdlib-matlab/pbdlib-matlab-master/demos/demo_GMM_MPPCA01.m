function demo_GMM_MPPCA01
% Mixture of probabilistic principal component analyzers (MPPCA) encoding.
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
model.nbStates = 4; %Number of states in the GMM
model.nbVar = 4; %Number of variables [x1,x2,x3,x4]
model.nbFA = 1; %Dimension of the subspace (number of principal components)
nbData = 200; %Length of each trajectory
nbSamples = 5; %Number of demonstrations


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/C.mat'); %Load x1,x2 variables
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
end
demos=[];
load('data/2Dletters/D.mat'); %Load x3,x4 variables
Data=[];
for n=1:nbSamples
	s(n).Data = [s(n).Data; spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData))]; %Resampling
	Data = [Data s(n).Data]; 
end


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kmeans(Data, model);
model0 = EM_GMM(Data, model); %for comparison
model = EM_MPPCA(Data, model);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,500]); 
for i=1:2
	subplot(1,2,i); hold on; box on; 
	plot(Data((i-1)*2+1,:),Data(i*2,:),'.','markersize',8,'color',[.7 .7 .7]);
	plotGMM(model0.Mu((i-1)*2+1:i*2,:), model0.Sigma((i-1)*2+1:i*2,(i-1)*2+1:i*2,:), [.8 .8 .8], .5);
	plotGMM(model.Mu((i-1)*2+1:i*2,:), model.Sigma((i-1)*2+1:i*2,(i-1)*2+1:i*2,:), [.8 0 0], .5);
	axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
	xlabel(['x_' num2str((i-1)*2+1)]); ylabel(['x_' num2str(i*2)]);
end

%print('-dpng','graphs/demo_GMM_MPPCA01.png');
pause;
close all;