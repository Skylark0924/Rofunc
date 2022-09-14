function demo_TPproMP01
% Task-parameterized probabilistic movement primitives (TP-ProMP)
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
model.nbStates = 10; %Number of Gaussians in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVar = 3; %Dimension of the datapoints in the dataset (here: t,x1,x2)
nbData = 200; %Number of datapoints in a trajectory
nbRepros = 4; %Number of reproductions with new situations randomly generated


%% Load 3rd order tensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load 3rd order tensor data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=3 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=200x4 the number of datapoints in a trajectory (200)
% multiplied by the number of demonstrations (nbSamples=5).
load('data/Data02.mat');


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_tensorGMM_timeBased(Data, model); 
%model = EM_tensorGMM(Data, model); 
model.Sigma(1,:,:,:) = model.Sigma(1,:,:,:) * 3;
model.Sigma(:,1,:,:) = model.Sigma(:,1,:,:) * 3;

%Compute basis functions activation based on time
DataIn = s(1).Data(1,:);
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = gaussPDF(DataIn, model.Mu(1,1,i), model.Sigma(1,1,1,i));
end
H = H ./ repmat(sum(H,1),model.nbStates,1);
GAMMA = repmat(H, 1, nbSamples);
GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
Psi = [];
for t=1:nbData
	Psi = [Psi; kron(H(:,t)',eye(model.nbVar))];
end
	
mBlock.nbStates = 1;
mBlock.nbFrames = model.nbFrames;
mBlock.nbVar = model.nbVar*model.nbStates;
for m=1:model.nbFrames	
	for n=1:nbSamples
		s(n).p(m).model.nbStates = model.nbStates;
		s(n).p(m).model = init_GMM_timeBased(squeeze(Data(:,m,(n-1)*nbData+1:n*nbData)), s(n).p(m).model);
		w(:,n) = reshape(s(n).p(m).model.Mu, model.nbStates*model.nbVar, 1);
	end
	Mu_w = mean(w,2);
	Sigma_w = cov(w') + eye(model.nbVar*model.nbStates) * 1E-2;
	%mBlock.Mu(:,m,1) = Psi * Mu_w;
	%mBlock.Sigma(:,:,m,1) = Psi * Sigma_w * Psi' + eye(model.nbVar*nbD) * 1E1;
	mBlock.Mu(:,m,1) = Mu_w;
	mBlock.Sigma(:,:,m,1) = Sigma_w;
	%Compute block version of task parameters 
	for n=1:nbSamples
		s(n).pBlock(m).A = kron(eye(model.nbStates), s(n).p(m).A);
		s(n).pBlock(m).b = kron(ones(model.nbStates,1), s(n).p(m).b);
	end
end

%For plot 
for i=1:model.nbStates
	id = (i-1)*model.nbVar+1:i*model.nbVar;
	model.Mu2(:,:,i) = mBlock.Mu(id,:);
	model.Sigma2(:,:,:,i) = mBlock.Sigma(id,id,:);
end

	
%% Reproduction for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reconstruct GMM for each demonstration
for n=1:nbSamples
	[s(n).MuBlock, s(n).SigmaBlock] = productTPGMM0(mBlock, s(n).pBlock);
	s(n).Mu = reshape(s(n).MuBlock, model.nbVar, model.nbStates);
	for i=1:model.nbStates
		id = (i-1)*model.nbVar+1:i*model.nbVar;
		s(n).Sigma(:,:,i) = s(n).SigmaBlock(id,id);
	end
	s(n).MuTraj = reshape(s(n).Mu*H, model.nbVar*nbData, 1);
	s(n).SigmaTraj = Psi * s(n).SigmaBlock * Psi' + eye(model.nbVar*nbData) * 0;
end


%% Reproduction for new task parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('New reproductions...');
for n=1:nbRepros
	for m=1:model.nbFrames
		%Random generation of new task parameters
		id=ceil(rand(2,1)*nbSamples);
		w=rand(2); w=w/sum(w);
		rnew(n).p(m).b = s(id(1)).p(m).b * w(1) + s(id(2)).p(m).b * w(2);
		rnew(n).p(m).A = s(id(1)).p(m).A * w(1) + s(id(2)).p(m).A * w(2);
		%Compute block version of task parameters 
		rnew(n).pBlock(m).A = kron(eye(model.nbStates), rnew(n).p(m).A);
		rnew(n).pBlock(m).b = kron(ones(model.nbStates,1), rnew(n).p(m).b);
	end
	[rnew(n).MuBlock, rnew(n).SigmaBlock] = productTPGMM0(mBlock, rnew(n).pBlock);
	rnew(n).Mu = reshape(rnew(n).MuBlock, model.nbVar, model.nbStates);
	for i=1:model.nbStates
		id = (i-1)*model.nbVar+1:i*model.nbVar;
		rnew(n).Sigma(:,:,i) = rnew(n).SigmaBlock(id,id);
	end
	rnew(n).MuTraj = reshape(rnew(n).Mu*H, model.nbVar*nbData, 1);
	rnew(n).SigmaTraj = Psi * rnew(n).SigmaBlock * Psi' + eye(model.nbVar*nbData) * 0;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,500]);
xx = round(linspace(1,64,nbSamples));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];

%DEMO
subplot(1,model.nbFrames+3,1); hold on; box on; title('Demonstrations');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
	%Plot demonstrations
	plot(s(n).Data(2,1), s(n).Data(3,1),'.','markersize',15,'color',clrmap(n,:));
	plot(s(n).Data(2,:), s(n).Data(3,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%REPRO
subplot(1,model.nbFrames+3,2); hold on; box on; title('Reproductions');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
	%Plot Gaussians
	plotGMM(s(n).Mu(2:3,:), s(n).Sigma(2:3,2:3,:), [.5 .5 .5], .3);
	%Plot reproductions
	plot(s(n).MuTraj(2:model.nbVar:end), s(n).MuTraj(3:model.nbVar:end), '-','lineWidth',2,'color',[0 0 0]);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%NEW REPRO
subplot(1,model.nbFrames+3,3); hold on; box on; title('New reproductions');
for n=1:nbRepros
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(rnew(n).p(m), colPegs(m,:));
	end
	%Plot trajectories
	plot(rnew(n).MuTraj(2), rnew(n).MuTraj(3), '.','markersize',12,'color',clrmap(n,:));
	plot(rnew(n).MuTraj(2:model.nbVar:end), rnew(n).MuTraj(3:model.nbVar:end), '-','lineWidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%MODEL
p0.A = eye(3);
p0.b = zeros(3,1);
for m=1:model.nbFrames
	subplot(1,model.nbFrames+3,3+m); hold on; grid on; box on; title(['Frame ' num2str(m)]);
	for n=1:nbSamples
		plot(squeeze(Data(2,m,(n-1)*s(1).nbData+1)), squeeze(Data(3,m,(n-1)*s(1).nbData+1)), '.','markersize',15,'color',clrmap(n,:));
		plot(squeeze(Data(2,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), squeeze(Data(3,m,(n-1)*s(1).nbData+1:n*s(1).nbData)), '-','linewidth',1.5,'color',clrmap(n,:));
	end
	plotGMM(squeeze(model.Mu2(2:end,m,:)), squeeze(model.Sigma2(2:end,2:end,m,:)), [.5 .5 .5], .3);
	plotPegs(p0, colPegs(m,:));
	axis equal; axis([-4.5 4.5 -1 8]); set(gca,'xtick',[0],'ytick',[0]);
end

% print('-dpng','graphs/demo_TPproMP01.png');
pause;
close all;
end

%Function to plot pegs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotPegs(p, colPegs, fa)
	if ~exist('colPegs')
		colPegs = [0.2863    0.0392    0.2392; 0.9137    0.4980    0.0078];
	end
	if ~exist('fa')
		fa = .6;
	end
	pegMesh = [-4 -3.5; -4 10; -1.5 10; -1.5 -1; 1.5 -1; 1.5 10; 4 10; 4 -3.5; -4 -3.5]' *1E-1;
	for m=1:length(p)
		dispMesh = p(m).A(2:3,2:3) * pegMesh + repmat(p(m).b(2:3),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end