function demo_TPtrajGMM01
% Task-parameterized model with trajectory-GMM encoding (GMM with dynamic features).
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
model.nbStates = 3; %Number of Gaussians in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 3; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 0.01; %Time step
nbData = 200; %Number of datapoints in a trajectory
nbRepros = 5; %Number of reproductions with new situations randomly generated


%% Load 3rd order tensor data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Load 3rd order tensor data...');
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension DC x P x N, with D=2 the dimension of a
% datapoint, C=2 the number of derivatives (incl. position), P=2 the number of candidate frames, and N=TM 
% the number of datapoints in a trajectory (T=200) multiplied by the number of demonstrations (M=5).
load('data/DataWithDeriv02.mat');

% %Convert position data to position+velocity data
% load('data/Data01.mat');
% %Create transformation matrix to compute derivatives
% D = (diag(ones(1,nbData-1),-1)-eye(nbData)) / model.dt;
% D(end,end) = 0;
% %Create 3rd order tensor data and task parameters
% Data = zeros(model.nbVar, model.nbFrames, nbSamples*nbData);
% for n=1:nbSamples
% 	s(n).Data = zeros(model.nbVar,model.nbFrames,nbData);
% 	s(n).Data0 = s(n).Data0(2:end,:); %Remove time
% 	DataTmp = s(n).Data0;
% 	for k=1:model.nbDeriv-1
% 		DataTmp = [DataTmp; s(n).Data0*D^k]; %Compute derivatives
% 	end
% 	for m=1:model.nbFrames
% 		s(n).p(m).b = [s(n).p(m).b; zeros((model.nbDeriv-1)*model.nbVarPos,1)];
% 		s(n).p(m).A = kron(eye(model.nbDeriv), s(n).p(m).A);
% 		s(n).Data(:,m,:) = s(n).p(m).A \ (DataTmp - repmat(s(n).p(m).b, 1, nbData));
% 		Data(:,m,(n-1)*nbData+1:n*nbData) = s(n).Data(:,m,:);
% 	end
% end
% %Save new dataset including derivatives
% save('data/DataWithDeriv02.mat', 'Data','s','nbSamples');

%Construct PHI operator (big sparse matrix)
[PHI, PHI1] = constructPHI(model, nbData, nbSamples); 


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of TP-GMM with EM...');
% model = init_tensorGMM_timeBased(Data, model); %Initialization
% model = init_tensorGMM_kmeans(Data, model); %Initialization

%Initialization based on position data
model0 = init_tensorGMM_kmeans(Data(1:model.nbVarPos,:,:), model);
[~,~,GAMMA2] = EM_tensorGMM(Data(1:model.nbVarPos,:,:), model0);
model.Priors = model0.Priors;
for i=1:model.nbStates
	for m=1:model.nbFrames
		DataTmp = squeeze(Data(:,m,:));
		model.Mu(:,m,i) = DataTmp * GAMMA2(i,:)';
		DataTmp = DataTmp - repmat(model.Mu(:,m,i),1,nbData*nbSamples);
		model.Sigma(:,:,m,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp';
	end
end

model = EM_tensorGMM(Data, model);


%% Reproduction for the task parameters used to train the model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproductions...');
for n=1:nbSamples
	%Products of linearly transformed Gaussians
	for i=1:model.nbStates
		SigmaTmp = zeros(model.nbVar);
		MuTmp = zeros(model.nbVar,1);
		for m=1:model.nbFrames
			MuP = s(n).p(m).A * model.Mu(:,m,i) + s(n).p(m).b;
			SigmaP = s(n).p(m).A * model.Sigma(:,:,m,i) * s(n).p(m).A';
			SigmaTmp = SigmaTmp + inv(SigmaP);
			MuTmp = MuTmp + SigmaP\MuP;
		end
		r(n).Sigma(:,:,i) = inv(SigmaTmp);
		r(n).Mu(:,i) = r(n).Sigma(:,:,i) * MuTmp;
	end
	%Create single Gaussian N(MuQ,SigmaQ) based on state sequence q, see Eq. (27)
	[~,r(n).q] = max(model.Pix(:,(n-1)*nbData+1:n*nbData),[],1); %works also for nbStates=1
	r(n).q
	r(n).MuQ = reshape(r(n).Mu(:,r(n).q), model.nbVarPos*model.nbDeriv*nbData, 1);
	r(n).SigmaQ = zeros(model.nbVarPos*model.nbDeriv*nbData);
	for t=1:nbData
		id1 = (t-1)*model.nbVarPos*model.nbDeriv+1:t*model.nbVarPos*model.nbDeriv;
		r(n).SigmaQ(id1,id1) = r(n).Sigma(:,:,r(n).q(t));
	end
	%Retrieval of data with trajectory GMM, see Eq. (30)
	PHIinvSigmaQ = PHI1'/r(n).SigmaQ;
	Rq = PHIinvSigmaQ * PHI1;
	rq = PHIinvSigmaQ * r(n).MuQ;
	r(n).Data = reshape(Rq\rq, model.nbVarPos, nbData); %Reshape data for plotting
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
	end
	%GMM products
	for i=1:model.nbStates
		SigmaTmp = zeros(model.nbVar);
		MuTmp = zeros(model.nbVar,1);
		for m=1:model.nbFrames
			MuP = rnew(n).p(m).A * model.Mu(:,m,i) + rnew(n).p(m).b;
			SigmaP = rnew(n).p(m).A * model.Sigma(:,:,m,i) * rnew(n).p(m).A';
			SigmaTmp = SigmaTmp + inv(SigmaP);
			MuTmp = MuTmp + SigmaP\MuP;
		end
		rnew(n).Sigma(:,:,i) = inv(SigmaTmp);
		rnew(n).Mu(:,i) = rnew(n).Sigma(:,:,i) * MuTmp;
	end
	%Create single Gaussian N(MuQ,SigmaQ) based on state sequence q, see Eq. (27)
	[~,rnew(n).q] = max(model.Pix(:,1:nbData),[],1); %works also for nbStates=1
	rnew(n).MuQ = reshape(rnew(n).Mu(:,rnew(n).q), model.nbVarPos*model.nbDeriv*nbData, 1);
	rnew(n).SigmaQ = zeros(model.nbVarPos*model.nbDeriv*nbData);
	for t=1:nbData
		id1 = (t-1)*model.nbVarPos*model.nbDeriv+1:t*model.nbVarPos*model.nbDeriv;
		rnew(n).SigmaQ(id1,id1) = rnew(n).Sigma(:,:,rnew(n).q(t));
	end
	%Retrieval of data with trajectory GMM, see Eq. (30)
	PHIinvSigmaQ = PHI1'/rnew(n).SigmaQ;
	Rq = PHIinvSigmaQ * PHI1;
	rq = PHIinvSigmaQ * rnew(n).MuQ;
	rnew(n).Data = reshape(Rq\rq, model.nbVarPos, nbData); %Reshape data for plotting
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,900]);
xx = round(linspace(1,64,nbSamples));
clrmap = colormap('jet');
clrmap = min(clrmap(xx,:),.95);
limAxes = [-1.2 0.8 -1.1 0.9];
colPegs = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078];

%DEMOS
subplot(1,3,1); hold on; box on; title('Demonstrations');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
	%Plot trajectories
	plot(s(n).Data0(1,1), s(n).Data0(2,1),'.','markersize',12,'color',clrmap(n,:));
	plot(s(n).Data0(1,:), s(n).Data0(2,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%REPROS
subplot(1,3,2); hold on; box on; title('Reproductions');
for n=1:nbSamples
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(s(n).p(m), colPegs(m,:));
	end
end
for n=1:nbSamples
	%Plot trajectories
	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',12,'color',clrmap(n,:));
	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',1.5,'color',clrmap(n,:));
end
for n=1:nbSamples
	%Plot Gaussians
	plotGMM(r(n).Mu(1:2,:,1), r(n).Sigma(1:2,1:2,:,1), [.5 .5 .5], .4);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%NEW REPROS
subplot(1,3,3); hold on; box on; title('New reproductions');
for n=1:nbRepros
	%Plot frames
	for m=1:model.nbFrames
		plotPegs(rnew(n).p(m), colPegs(m,:));
	end
end
for n=1:nbRepros
	%Plot trajectories
	plot(rnew(n).Data(1,1), rnew(n).Data(2,1),'.','markersize',12,'color',[.2 .2 .2]);
	plot(rnew(n).Data(1,:), rnew(n).Data(2,:),'-','linewidth',1.5,'color',[.2 .2 .2]);
end
for n=1:nbRepros
	%Plot Gaussians
	plotGMM(rnew(n).Mu(1:2,:,1), rnew(n).Sigma(1:2,1:2,:,1), [.5 .5 .5], .4);
end
axis(limAxes); axis square; set(gca,'xtick',[],'ytick',[]);

%print('-dpng','graphs/demo_TPtrajGMM01.png');
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
		dispMesh = p(m).A(1:2,1:2) * pegMesh + repmat(p(m).b(1:2),1,size(pegMesh,2));
		h(m) = patch(dispMesh(1,:),dispMesh(2,:),colPegs(m,:),'linewidth',1,'edgecolor','none','facealpha',fa);
	end
end