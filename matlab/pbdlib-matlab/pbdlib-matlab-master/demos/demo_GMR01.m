function demo_GMR01
% Gaussian mixture regression (GMR) 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
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
model.nbVar = 3; %Number of variables [t,x1,x2]
model.dt = 0.001; %Time step duration
nbData = 200; %Length of each trajectory
nbSamples = 5; %Number of demonstrations


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
DataIn(1,:) = [1:nbData] * model.dt;
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data [DataIn; s(n).Data]]; 
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model); %Initialization of model parameters with k-means
model = init_GMM_timeBased(Data, model); %Initialization of model parameters with equal bins
model = EM_GMM(Data, model); %Model parameters fitting with expectation-maximization algorithm
[DataOut, SigmaOut] = GMR(model, DataIn, 1, 2:model.nbVar); %Gaussian mixture regression


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,1000]); 

%Plot GMM
subplot(2,2,1); hold on; axis off; title('GMM');
plot(Data(2,:),Data(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(model.Mu(2:model.nbVar,:), model.Sigma(2:model.nbVar,2:model.nbVar,:), [.8 0 0], .5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%Plot GMR
subplot(2,2,2); hold on; axis off; title('GMR');
plot(Data(2,:),Data(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(DataOut, SigmaOut, [0 .8 0], .03);
%plotGMM(model.Mu(2:model.nbVar,:), model.Sigma(2:model.nbVar,2:model.nbVar,:), [.8 0 0], .5);
plot(DataOut(1,:),DataOut(2,:),'-','linewidth',2,'color',[0 .4 0]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%Timeline plots
for m=1:2
	subplot(2,2,2+m); hold on; 
	for n=1:nbSamples
		plot(Data(1,(n-1)*nbData+1:n*nbData), Data(m+1,(n-1)*nbData+1:n*nbData), '-','markersize',8,'color',[.2 .2 .2]);
	end
	patch([DataIn(1,:), DataIn(1,end:-1:1)], [DataOut(m,:)+squeeze(SigmaOut(m,m,:).^.5)', DataOut(m,end:-1:1)-squeeze(SigmaOut(m,m,end:-1:1).^.5)'], [.2 .9 .2],'edgecolor','none','facealpha',.5);
	plotGMM(model.Mu([1,1+m],:), model.Sigma([1,1+m],[1,1+m],:), [.8 0 0], .5);
	plot(DataIn(1,:), DataOut(m,:), '-','lineWidth',3,'color',[0 .3 0]);
	set(gca,'xtick',[],'ytick',[]);
	xlabel('t','fontsize',16); ylabel(['x_' num2str(m)],'fontsize',16);
	axis([DataIn(:,1), DataIn(:,end), min(DataOut(m,:))-4E0, max(DataOut(m,:))+4E0]);
end

%print('-dpng','graphs/demo_GMR01.png');
pause;
close all;