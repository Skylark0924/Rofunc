function demo_GMR_probTraj01
% Probabilistic trajectory generation with GMR obtained from normally distributed GMM centers
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
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
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 3; %Number of variables 
model.nbVarOut = 2; %Number of variables
model.dt = 1E-2; %Time step duration
nbData = 100; %Length of each trajectory
nbSamples = 10; %Number of demonstrations
nbRepros = 1;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data [[1:nbData]*model.dt; s(n).Data]]; 
end


%% Learning and reproduction (GMM, GMR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);	

% %Removal of last Gaussian (for debug purpose)
% model.Mu = model.Mu(:,1:end-1);
% model.Sigma = model.Sigma(:,:,1:end-1);
% model.nbStates = model.nbStates - 1;

in = 1;
out = 2:model.nbVar;
DataIn = Data(1,1:nbData);

MuTmp = zeros(model.nbVarOut, model.nbStates);
expData = zeros(model.nbVarOut, nbData);
expSigma = zeros(model.nbVarOut, model.nbVarOut, nbData);
SigmaCond = zeros(model.nbVarOut, model.nbVarOut, model.nbStates);
for i = 1:model.nbStates
	SigmaCond(:,:,i) = model.Sigma(out,out,i) - model.Sigma(out,in,i) / model.Sigma(in,in,i) * model.Sigma(in,out,i);
end

H = zeros(model.nbStates, nbData);
Hs = zeros(model.nbStates, nbData);
expSigmaBlk = [];
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(DataIn(:,t), model.Mu(in,i), model.Sigma(in,in,i));
	end
	Hs(:,t) = H(:,t) / sum(H(:,t)+realmin);
	%Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(out,i) + model.Sigma(out,in,i)/model.Sigma(in,in,i) * (DataIn(:,t) - model.Mu(in,i));
	end
	expData(:,t) = MuTmp * Hs(:,t);	
	%Compute conditional covariances
	for i=1:model.nbStates
% 		MuCtr = MuTmp - repmat(mean(MuTmp,2),1,model.nbStates);
% 		expSigma(:,:,t) = MuCtr * diag(Hs(:,t)) * MuCtr';

% 		expSigma(:,:,t) = expSigma(:,:,t) + Hs(i,t) * SigmaCond(:,:,i);
		expSigma(:,:,t) = expSigma(:,:,t) + Hs(i,t) * (SigmaCond(:,:,i) + MuTmp(:,i)*MuTmp(:,i)');
	end
	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t)*expData(:,t)' + eye(model.nbVarOut) * model.params_diagRegFact;
	expSigmaBlk = [expSigmaBlk; sqrtm(expSigma(:,:,t))];
end


%% GMR process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The centers of the GMM are given by (Mu_k(in), Normal(Mu_k(out), Sigma_k(out|in))).
% The mean of the trajectory distribution is the mean obtained by the GMR.
% The covariance of the trajectory distribution is Sum_k(H_k*H_k'*Sigma_k(out|in)).
MuTraj = expData(:);
% Ht = kron(Hs', ones(model.nbVarOut,1));
% SigmaTraj0 = (Ht.^.5 * Ht.^.5') .* (expSigmaBlk * expSigmaBlk');

% SigmaTraj0 = Ht * Ht';
SigmaTraj0 = expSigmaBlk * expSigmaBlk';
rank(SigmaTraj0)

H = exp(-1E2 .* pdist2(DataIn',DataIn').^2);
H = kron(H', eye(model.nbVarOut));
SigmaTraj = SigmaTraj0 .* H;
% SigmaTraj = 1 .* H';
rank(SigmaTraj)

	
% %% Stochastic trajectory generation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [V,D] = eig(SigmaTraj);
% for m = 1:nbRepros-1
% 	traj(m).Mu = MuTraj + real(V) * real(D).^.5 * randn(size(MuTraj));
% end


%% Trajectory reconstruction from partial data with Gaussian conditioning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in0 = [1,50]; %The number of points should not exceed the number of states in the GMM (i.e., rank(SigmaTraj)/nbVar)
out0 = [2:59,51:nbData];
	
offset = (rand(model.nbVarOut,length(in0))-.5) .* 1E1;
inTraj = [];
for i=1:length(in0)
	inTraj = [inTraj (in0(i)-1)*model.nbVarOut+[1:model.nbVarOut]];
end
outTraj = [];
for i=1:length(out0)
	outTraj = [outTraj (out0(i)-1)*model.nbVarOut+[1:model.nbVarOut]];
end
for m=nbRepros
	traj(m).Mu = zeros(size(MuTraj));
	traj(m).Sigma = zeros(size(SigmaTraj));
	traj(m).Mu(inTraj) = reshape(Data(2:model.nbVar,in0)+offset, model.nbVarOut*length(in0), 1);	
	traj(m).Mu(outTraj) = MuTraj(outTraj) + SigmaTraj(outTraj,inTraj)/SigmaTraj(inTraj,inTraj) * (traj(m).Mu(inTraj)-MuTraj(inTraj));
	traj(m).Sigma(outTraj,outTraj) = SigmaTraj(outTraj,outTraj) - SigmaTraj(outTraj,inTraj) / SigmaTraj(inTraj,inTraj) * SigmaTraj(inTraj,outTraj);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 20 10],'position',[10 10 2200 800],'color',[1 1 1]);
subplot(1,3,1); hold on; axis off;
%Plot data
for nn=1:nbSamples
	plot(Data(2,(nn-1)*nbData+1:nn*nbData), Data(3,(nn-1)*nbData+1:nn*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
end
%Plot GMR distribution
plotGMM(expData, expSigma, [0 .8 0],.1);
plot(expData(1,:), expData(2,:), '-','lineWidth',1,'color',[0 0 .8]);
%Plot prior distribution
for t=1:nbData
	plotGMM(MuTraj((t-1)*2+[1,2]), SigmaTraj((t-1)*2+[1,2],(t-1)*2+[1,2]), [0 0 .8],.1);
end
plot(MuTraj(1:2:end), MuTraj(2:2:end), '-','lineWidth',1,'color',[0 0 .8]);
%Plot posterior distribution
for t=1:1:nbData
	plotGMM(traj(nbRepros).Mu((t-1)*2+[1,2]), traj(nbRepros).Sigma((t-1)*2+[1,2],(t-1)*2+[1,2]), [.2 .2 .2],.1);
end
plot(traj(nbRepros).Mu(1:2:end), traj(nbRepros).Mu(2:2:end), '-','lineWidth',1,'color',[0 0 0]);
plot(traj(nbRepros).Mu(inTraj(1:2:end)), traj(nbRepros).Mu(inTraj(2:2:end)), '.','markersize',18,'color',[0 0 0]);
% for m=1:nbRepros-1
% 	plot(traj(m).Mu(1:2:end), traj(m).Mu(2:2:end), '-','lineWidth',1,'color',[0 0 0]);
% end
plotGMM(model.Mu(2:3,:), model.Sigma(2:3,2:3,:), [.8 0 0], .3);
axis equal; axis([min(Data(2,:))-8 max(Data(2,:))+8 min(Data(3,:))-8 max(Data(3,:))+8]);

%Plot covariance of trajectory distribution
subplot(1,3,2); hold on; box on; set(gca,'linewidth',2); title('SigmaTraj0');
colormap(gca, flipud(gray));
pcolor(abs(SigmaTraj0));
set(gca,'xtick',[1,nbData*model.nbVarOut],'ytick',[1,nbData*model.nbVarOut]);
axis square; axis([1 nbData*model.nbVarOut 1 nbData*model.nbVarOut]); shading flat;

%Plot covariance of trajectory distribution
subplot(1,3,3); hold on; box on; set(gca,'linewidth',2); title('SigmaTraj');
colormap(gca, flipud(gray));
pcolor(abs(SigmaTraj));
set(gca,'xtick',[1,nbData*model.nbVarOut],'ytick',[1,nbData*model.nbVarOut]);
axis square; axis([1 nbData*model.nbVarOut 1 nbData*model.nbVarOut]); shading flat;


%Timeline plots
figure('PaperPosition',[0 0 20 10],'position',[10 850 1600 500],'color',[1 1 1]);
for m=1:model.nbVarOut
	subplot(1,2,m); hold on; 
	for n=1:nbSamples
		plot(Data(1,(n-1)*nbData+1:n*nbData), Data(m+1,(n-1)*nbData+1:n*nbData), '-','linewidth',1,'color',[.7 .7 .7]);
	end
	id = m:2:nbData*model.nbVarOut;
	h(1) = patch([DataIn(1,:), DataIn(1,end:-1:1)], [expData(m,:)+squeeze(expSigma(m,m,:).^.5)', expData(m,end:-1:1)-squeeze(expSigma(m,m,end:-1:1).^.5)'], [0 .8 0],'edgecolor','none','facealpha',.2);
	h(2) = patch([DataIn(1,:), DataIn(1,end:-1:1)], [MuTraj(id)'-diag(SigmaTraj(id,id).^.5)', MuTraj(fliplr(id))'+diag(SigmaTraj(fliplr(id),fliplr(id)).^.5)'], [0 0 .8],'edgecolor','none','facealpha',.2);
	
	plot(DataIn(1,:), MuTraj(m:2:end)', '-','lineWidth',1,'color',[0 0 .8]);
	plotGMM(model.Mu([1,1+m],:), model.Sigma([1,1+m],[1,1+m],:), [.8 0 0], .5);
	
	%Plot conditional distribution
	h(3) = patch([DataIn(1,:), DataIn(1,end:-1:1)], [traj(nbRepros).Mu(id)'-diag(traj(nbRepros).Sigma(id,id).^.5)', traj(nbRepros).Mu(fliplr(id))'+diag(traj(nbRepros).Sigma(fliplr(id),fliplr(id)).^.5)'], [.6 .6 .6],'edgecolor','none','facealpha',.5);
	plot(DataIn(1,:), traj(nbRepros).Mu(m:2:end)', '-','lineWidth',1,'color',[0 0 0]);
	plot(DataIn(1,in0), traj(nbRepros).Mu(inTraj(m:2:end)), '.','markersize',18,'color',[0 0 0]);
	set(gca,'xtick',[],'ytick',[]); xlabel('t'); ylabel(['x_' num2str(m)]);
	legend(h,{'GMR distrib.','GPR prior distrib.','GPR posterior distrib.'},'fontsize',16);
end
% std = diag(Kss)'.^.5 .* .4;
% patch([r(1).Data(1,:), r(1).Data(1,end:-1:1)], [std, -fliplr(std)], [.8 .8 .8],'edgecolor','none');

pause;
close all;