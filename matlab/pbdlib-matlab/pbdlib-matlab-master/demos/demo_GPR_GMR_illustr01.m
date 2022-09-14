function demo_GPR_GMR_illustr01
% Illustration of the different notions of variance for GPR and GMR.
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
model.nbStates = 4; %Number of states in the GMM
model.nbVar = 3; %Number of variables [t,x1,x2]
model.dt = 0.005; %Time step duration
model.params_diagRegFact = 1E-4;

nbVar = 3; %Dimension of datapoint (t,x1,x2)
nbData = 100; %Number of datapoints
nbDataRepro = 100; %Number of datapoints for reproduction
nbSamples = 8; %Number of demonstrations

p(1)=1E0; p(2)=1E-3; p(3)=1E-2; %GPR parameters 


%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/pathsWithVariations01.mat');
for n=1:nbSamples	
	s0(n).Data = Data(:,(n-1)*nbData+1:n*nbData);
end

%s = s0;

%Dynamic time warping
wMax = 3; %Warping time window 
s(1).Data = s0(1).Data;
for n=2:nbSamples
	[s(1).Data, s(n).Data, s(n-1).wPath] = DTW(s(1).Data, s0(n).Data, wMax);
	%Realign previous trajectories
	pp = s(n-1).wPath(1,:);
	for m=2:n-1
		DataTmp = s(m).Data(:,pp);
		s(m).Data = spline(1:size(DataTmp,2), DataTmp, linspace(1,size(DataTmp,2),nbData)); %Resampling
	end
end

%Simulate missing data
Data=[];
for n=1:nbSamples	
	%s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	%tt = [1:nbData/2,3*nbData/4:nbData];  %Simulate missing data
	tt = [1:45, 80:nbData];
	tt2 = [1:45, 60:nbData-20];
		
	%tt = 1:nbData;
	s(n).Data = [tt2*model.dt; s(n).Data(:,tt)];
	Data = [Data s(n).Data]; 
end
%Recenter data
Data(2:end,:) = Data(2:end,:) - repmat(mean(Data(2:end,:),2),1,size(Data,2));


%% Learning and reproduction with GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xIn = Data(1,:);
xOut = Data(2:end,:);
xInHat = linspace(xIn(:,1), xIn(:,end), nbDataRepro);
%model = init_GMM_timeBased(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);
model = EM_GMM(Data, model);
[DataOut, r(1).SigmaOut] = GMR(model, xInHat, 1, 2:model.nbVar); %see Eq. (17)-(19)
r(1).Data = [xInHat; DataOut];
%[DataOut, SigmaOut] = GMR(model, model.Mu(1,:), 1, 2:model.nbVar); %see Eq. (17)-(19)


%% Reproduction with GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%GPR precomputation
M = pdist2(xIn', xIn');
K = p(1) * exp(-p(2)^-1 * M.^2);
invK = pinv(K + p(3) * eye(size(K))); 

Md = pdist2(xInHat', xIn');
Kd = p(1) * exp(-p(2)^-1 * Md.^2);

r(2).Data = [xInHat; (Kd * invK * xOut')']; 
%Covariance computation
Mdd = pdist2(xInHat',xInHat');
Kdd = p(1) * exp(-p(2)^-1 * Mdd.^2);
S = Kdd - Kd * invK * Kd';
r(2).SigmaOut = zeros(nbVar-1,nbVar-1,nbDataRepro);
for t=1:nbDataRepro
	r(2).SigmaOut(:,:,t) = eye(nbVar-1) * S(t,t) * 1E1; 
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 2300 1200]);
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
set(gca,'LooseInset',[0,0,0,0]);
	
%Plots 1D GMR
for m=2:nbVar
	limAxes = [xIn(:,1), xIn(:,end), min(Data(m,:))-1E0, max(Data(m,:))+2E0];
	%subplot(nbVar-1,4,(m-2)*4+1); hold on;
	subaxis(nbVar-1,4,(m-2)*4+1,'Spacing',1E-2); hold on; 
	
	plot(Data(1,:), Data(m,:), '.','markersize',12,'color',[.6 .6 .6]);
	
	patch([r(1).Data(1,:), r(1).Data(1,end:-1:1)], ...
		[r(1).Data(m,:)+squeeze(r(1).SigmaOut(m-1,m-1,:).^.5)', r(1).Data(m,end:-1:1)-squeeze(r(1).SigmaOut(m-1,m-1,end:-1:1).^.5)'], ...
		[.2 .9 .2],'edgecolor','none','facealpha',.5);
	plotGMM(model.Mu([1,m],:), model.Sigma([1,m],[1,m],:), [0 .4 0], .3);
	plot(r(1).Data(1,:), r(1).Data(m,:), '-','lineWidth',2,'color',[0 .7 0]);

	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	ylabel(['$x_' num2str(m-1) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
	if m==2
		title('GMR','fontsize',18);
	end
end

%Plots 1D GPR
for m=2:nbVar
	limAxes = [xIn(:,1), xIn(:,end), min(Data(m,:))-1E0, max(Data(m,:))+2E0];
	%subplot(nbVar-1,4,(m-2)*4+2); hold on;
	subaxis(nbVar-1,4,(m-2)*4+2,'Spacing',1E-2); hold on; 
	
	plot(Data(1,:), Data(m,:), '.','markersize',12,'color',[.6 .6 .6]);
	
	patch([r(2).Data(1,:), r(2).Data(1,end:-1:1)], ...
		[r(2).Data(m,:)+squeeze(r(2).SigmaOut(m-1,m-1,:).^.5)', r(2).Data(m,end:-1:1)-squeeze(r(2).SigmaOut(m-1,m-1,end:-1:1).^.5)'], ...
		[1 .2 .2],'edgecolor','none','facealpha',.5);
	plot(r(2).Data(1,:), r(2).Data(m,:),'-','linewidth',2,'color',[.8 0 0]);

	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	%ylabel(['$x_' num2str(m) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
	if m==2
		title('GPR','fontsize',18);
	end
end

%Plot 2D
subplot(nbVar-1,4,[3,4,7,8]); hold on;
%subaxis(nbVar-1,4,[3,4,7,8],'Spacing',0); hold on; 

plot(Data(2,:), Data(3,:), '.','markersize',12,'color',[.6 .6 .6]); 

plotGMM(r(2).Data(2:3,:), r(2).SigmaOut, [1 .2 .2], .1);
plotGMM(r(1).Data(2:3,:), r(1).SigmaOut, [.2 1 .2], .1);
plotGMM(model.Mu(2:model.nbVar,:), model.Sigma(2:model.nbVar,2:model.nbVar,:), [0 .4 0], .3);

plot(r(1).Data(2,:), r(1).Data(3,:), '-','lineWidth',2,'color',[0 .7 0]);
plot(r(2).Data(2,:), r(2).Data(3,:), '-','lineWidth',2,'color',[.8 0 0]);

set(gca,'xtick',[],'ytick',[]); axis equal;
xlabel(['$x_1$'],'interpreter','latex','fontsize',18);
ylabel(['$x_2$'],'interpreter','latex','fontsize',18);

%print('-dpng','graphs/demo_GPR_GMR_illustr01.png');
pause;
close all;