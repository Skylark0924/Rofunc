function demo_affineTransform01
% Affine transformations of raw data as pre-processing step to train a task-parameterized model. 
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVar = 2; %Dimension of the datapoints in the dataset (here: x1,x2)
model.dt = 0.01; %Time step
nbData = 200; %Number of datapoints in a trajectory


%% Load 3rd order tensor data (for [x] encoding)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load 3rd order tensor data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=2 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=TM the number of datapoints in a trajectory (T=200)
% multiplied by the number of demonstrations (M=5).
load('data/Data01.mat');


%% Task parameters and observed data for [t,x] encoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model2.nbVar = 3; %Dimension of the datapoints in the dataset (here: t,x1,x2)
%Create 3rd order tensor data and task parameters for [t,x]
Data2 = zeros(model2.nbVar, model.nbFrames, nbSamples*nbData);
for n=1:nbSamples
	%size(s(n).Data)
	for m=1:model.nbFrames
		s2(n).p(m).b = [0; s(n).p(m).b];
		s2(n).p(m).A = eye(model2.nbVar);
		s2(n).p(m).A(2:end,2:end) = s(n).p(m).A;
		Data2(:,m,(n-1)*nbData+1:n*nbData) = s2(n).p(m).A \ (s(n).Data0 - repmat(s2(n).p(m).b, 1, nbData));
	end
end


%% Task parameters and observed data for [x,dx] encoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model3.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model3.nbDeriv = 2; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model3.nbVar = model3.nbVarPos * model3.nbDeriv; %Dimension of state vector
%Create transformation matrix to compute derivatives
D = (diag(ones(1,nbData-1),-1)-eye(nbData)) / model.dt;
D(end,end) = 0;
%Create 3rd order tensor data and task parameters for [x,dx]
Data3 = zeros(model3.nbVar, model.nbFrames, nbSamples*nbData);
for n=1:nbSamples
	s3(n).Data = zeros(model3.nbVar, model.nbFrames, nbData);
	s3(n).Data0 = s(n).Data0(2:end,:); %Remove time
	DataTmp = s3(n).Data0;
	for k=1:model3.nbDeriv-1
		DataTmp = [DataTmp; s3(n).Data0*D^k]; %Compute derivatives
	end
	for m=1:model.nbFrames
		s3(n).p(m).b = [s(n).p(m).b; zeros((model3.nbDeriv-1)*model3.nbVarPos,1)];
		s3(n).p(m).A = kron(eye(model3.nbDeriv), s(n).p(m).A);
		s3(n).Data(:,m,:) = s3(n).p(m).A \ (DataTmp - repmat(s3(n).p(m).b, 1, nbData));
		Data3(:,m,(n-1)*nbData+1:n*nbData) = s3(n).Data(:,m,:);
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,700]);
xx = round(linspace(1,64,nbSamples));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [[.9,.5,.9];[.5,.9,.5]];

%DEMOS1
subplot(3,model.nbFrames+1,1); hold on; box on; title('Demonstrations [x]');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plot([s(n).p(m).b(1) s(n).p(m).b(1)+s(n).p(m).A(1,2)], [s(n).p(m).b(2) s(n).p(m).b(2)+s(n).p(m).A(2,2)], '-','linewidth',6,'color',colPegs(m,:));
		plot(s(n).p(m).b(1), s(n).p(m).b(2),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
	%Plot trajectories
	plot(s(n).Data0(2,1), s(n).Data0(3,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(2,:), s(n).Data0(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%DEMOS2
subplot(3,model.nbFrames+1,4); hold on; box on; title('Demonstrations [t,x]');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plot([s2(n).p(m).b(2) s2(n).p(m).b(2)+s2(n).p(m).A(2,3)], [s2(n).p(m).b(3) s2(n).p(m).b(3)+s2(n).p(m).A(3,3)], '-','linewidth',6,'color',colPegs(m,:));
		plot(s2(n).p(m).b(2), s2(n).p(m).b(3),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
	%Plot trajectories
	plot(s(n).Data0(2,1), s(n).Data0(3,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(2,:), s(n).Data0(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%DEMOS3
subplot(3,model.nbFrames+1,7); hold on; box on; title('Demonstrations [x,dx]');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plot([s3(n).p(m).b(1) s3(n).p(m).b(1)+s3(n).p(m).A(1,2)], [s3(n).p(m).b(2) s3(n).p(m).b(2)+s3(n).p(m).A(2,2)], '-','linewidth',6,'color',colPegs(m,:));
		plot(s3(n).p(m).b(1), s3(n).p(m).b(2),'.','markersize',30,'color',colPegs(m,:)-[.05,.05,.05]);
	end
	%Plot trajectories
	plot(s(n).Data0(2,1), s(n).Data0(3,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(2,:), s(n).Data0(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%FRAMES
for m=1:model.nbFrames
	subplot(3,model.nbFrames+1,1+m); hold on; grid on; box on; title(['Frame ' num2str(m) ' [x]']);
	for n=1:nbSamples
		plot(squeeze(Data(1,m,(n-1)*s(1).nbData+1)), ...
			squeeze(Data(2,m,(n-1)*s(1).nbData+1)), '.','markersize',15,'color',clrmap(n,:));
		plot(squeeze(Data(1,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), ...
			squeeze(Data(2,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), '-','linewidth',1.5,'color',clrmap(n,:));
	end
	axis square; set(gca,'xtick',[0],'ytick',[0]);
end

%FRAMES2
for m=1:model.nbFrames
	subplot(3,model.nbFrames+1,4+m); hold on; grid on; box on; title(['Frame ' num2str(m) ' [t,x]']);
	for n=1:nbSamples
		plot(squeeze(Data2(2,m,(n-1)*s(1).nbData+1)), ...
			squeeze(Data2(3,m,(n-1)*s(1).nbData+1)), '.','markersize',15,'color',clrmap(n,:));
		plot(squeeze(Data2(2,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), ...
			squeeze(Data2(3,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), '-','linewidth',1.5,'color',clrmap(n,:));
	end
	axis square; set(gca,'xtick',[0],'ytick',[0]);
end

%FRAMES3
for m=1:model.nbFrames
	subplot(3,model.nbFrames+1,7+m); hold on; grid on; box on; title(['Frame ' num2str(m) ' [x,dx]']);
	for n=1:nbSamples
		plot(squeeze(Data3(1,m,(n-1)*s(1).nbData+1)), ...
			squeeze(Data3(2,m,(n-1)*s(1).nbData+1)), '.','markersize',15,'color',clrmap(n,:));
		plot(squeeze(Data3(1,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), ...
			squeeze(Data3(2,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), '-','linewidth',1.5,'color',clrmap(n,:));
	end
	axis square; set(gca,'xtick',[0],'ytick',[0]);
end

%print('-dpng','graphs/demo_affineTransform01.png');
%pause;
%close all;
