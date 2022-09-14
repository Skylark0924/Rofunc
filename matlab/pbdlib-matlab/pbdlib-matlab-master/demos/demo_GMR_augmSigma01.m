function demo_GMR_augmSigma01
% GMR with Gaussians reparameterized to have zero means and augmented covariances 
% (showing that it gives the same result as standard GMR)
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
model.dt = 1E-2; %Time step duration
nbData = 50; %Length of each trajectory
nbSamples = 5; %Number of demonstrations


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data [[1:nbData]*model.dt; s(n).Data]]; 
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased(Data, model);
model = EM_GMM(Data, model);

%Initialisation of model2 (Gaussians with augmented covariances centered on zero)
model2 = model;
model2.nbVar = model.nbVar+1;
model2.Mu = zeros(model2.nbVar, model2.nbStates);
model2.Sigma = zeros(model2.nbVar, model2.nbVar, model2.nbStates);
for i=1:model.nbStates
	model2.Sigma(:,:,i) = [model.Sigma(:,:,i)+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1];
% 	model2.Sigma(:,:,i) = [model.Sigma(:,:,i)+model.Mu(:,i)*model.Mu(:,i)', model.Mu(:,i); model.Mu(:,i)', 1] .* (det(model.Sigma(:,:,i)).^(-1./(model.nbVar+1)));
end

[DataOut, SigmaOut] = GMR(model, [1:nbData]*model.dt, 1, 2:model.nbVar); %see Eq. (17)-(19)
%[DataOut2, SigmaOut2] = GMR(model2, [[1:nbData]*model.dt'; ones(1,nbData)], [1,model.nbVar+1], 2:model.nbVar); %see Eq. (17)-(19)


%GMR with augmented covariance model
in = [1,model.nbVar+1]; 
out = 2:model.nbVar;
DataIn = [[1:nbData]*model.dt; ones(1,nbData)];

nbVarOut = length(out);
MuTmp = zeros(nbVarOut,model.nbStates);
DataOut2 = zeros(nbVarOut,nbData);
SigmaOut2 = zeros(nbVarOut,nbVarOut,nbData);
for t=1:nbData
	%Compute activation weight
	for i=1:model2.nbStates
		H(i,t) = model2.Priors(i) * gaussPDF(DataIn(:,t), model2.Mu(in,i), model2.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	%Compute conditional means
	for i=1:model2.nbStates
		MuTmp(:,i) = model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * DataIn(:,t); %-> we have y=Ax instead of y=Ax+b
		DataOut2(:,t) = DataOut2(:,t) + H(i,t) * MuTmp(:,i);
	end
	%Compute conditional covariances
	for i=1:model2.nbStates
		Sout(:,:,i) = model2.Sigma(out,out,i) - model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * model2.Sigma(in,out,i);
		SigmaOut2(:,:,t) = SigmaOut2(:,:,t) + H(i,t) * (Sout(:,:,i) + MuTmp(:,i)*MuTmp(:,i)');
		%SigmaOut2(:,:,t) = SigmaOut2(:,:,t) + H(i,t) * Sout(:,:,i);
	end
	SigmaOut2(:,:,t) = SigmaOut2(:,:,t) - DataOut2(:,t)*DataOut2(:,t)' + eye(nbVarOut)*1E-6; 
end


%GMR with augmented covariance model (test)
in = 1; 
out = 2:model.nbVar+1;
DataIn = [1:nbData]*model.dt;

nbVarOut = length(out);
MuTmp = zeros(nbVarOut,model.nbStates);
DataOut3 = zeros(nbVarOut,nbData);
SigmaOut3 = zeros(nbVarOut,nbVarOut,nbData);
Sout = zeros(nbVarOut,nbVarOut,model.nbStates);
for t=1:nbData
% 	%Compute activation weight
% 	for i=1:model2.nbStates
% 		H(i,t) = model2.Priors(i) * gaussPDF(DataIn(:,t), model2.Mu(in,i), model2.Sigma(in,in,i));
% 	end
% 	H(:,t) = H(:,t) / sum(H(:,t)+realmin);

% 	%Compute conditional means
% 	for i=1:model2.nbStates
% 		MuTmp(:,i) = model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * DataIn(:,t); %-> we have y=Ax instead of y=Ax+b
% 		DataOut3(:,t) = DataOut3(:,t) + H(i,t) * MuTmp(:,i);
% 	end

	%Compute conditional covariances
	for i=1:model2.nbStates
		Sout(:,:,i) = model2.Sigma(out,out,i); % - model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * model2.Sigma(in,out,i);
		%SigmaOut3(:,:,t) = SigmaOut3(:,:,t) + H(i,t) * (Sout(:,:,i) + MuTmp(:,i)*MuTmp(:,i)');
		SigmaOut3(:,:,t) = SigmaOut3(:,:,t) + H(i,t) * Sout(:,:,i);
	end
	%SigmaOut3(:,:,t) = SigmaOut3(:,:,t) - DataOut3(:,t)*DataOut3(:,t)' + eye(nbVarOut)*1E-6; 
end

SigmaOut3 = SigmaOut3 ./ repmat(SigmaOut3(end,end,:)+realmin, [nbVarOut,nbVarOut,1]); %Post-rescaling (bad!)
DataOut4 = squeeze(SigmaOut3(1:end-1,end,:));
SigmaOut4 = zeros(nbVarOut-1, nbVarOut-1, nbData*model.nbStates);
for t=1:nbData
	SigmaOut4(:,:,t) = SigmaOut3(1:end-1,1:end-1,t) - DataOut4(:,t)*DataOut4(:,t)';
end


% %GMR with augmented covariance model (non-working test)
% in = 1; 
% out = 2:model.nbVar+1;
% 
% in2 = [1,model.nbVar+1]; 
% %out2 = 2:model.nbVar;
% 
% nbVarOut = length(out);
% %DataIn = [1:nbData]*model.dt;
% DataIn = [[1:nbData]*model.dt; ones(1,nbData)];
% 
% MuTmp = zeros(nbVarOut,model.nbStates);
% DataOut2 = zeros(nbVarOut,nbData);
% SigmaOut2 = zeros(nbVarOut,nbVarOut,nbData);
% for t=1:nbData
% 	%Compute activation weight
% 	for i=1:model2.nbStates
% 		H(i,t) = model2.Priors(i) * gaussPDF(DataIn(:,t), model2.Mu(in2,i), model2.Sigma(in2,in2,i));
% 	end
% 	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
% % 	%Compute conditional means
% % 	for i=1:model2.nbStates
% % 		MuTmp(:,i) = model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * DataIn(:,t); %We have y=Ax instead of y=Ax+b
% % 		%[t*model.dt; 1]
% % 		%model2.Sigma(out,in,i)/model2.Sigma(in,in,i) 
% % 		%DataIn(:,t)
% % 		%pause
% % 		DataOut2(:,t) = DataOut2(:,t) + H(i,t) * MuTmp(:,i);
% % 	end
% 	%Compute conditional covariances
% 	for i=1:model2.nbStates
% 		Sout(:,:,i) = model2.Sigma(out,out,i) - model2.Sigma(out,in,i)/model2.Sigma(in,in,i) * model2.Sigma(in,out,i);
% 		SigmaOut2(:,:,t) = SigmaOut2(:,:,t) + H(i,t) * (Sout(:,:,i) + MuTmp(:,i)*MuTmp(:,i)');
% 		%SigmaOut2(:,:,t) = SigmaOut2(:,:,t) + H(i,t) * Sout(:,:,i);
% 	end
% 	SigmaOut2(:,:,t) = SigmaOut2(:,:,t) - DataOut2(:,t)*DataOut2(:,t)' + eye(nbVarOut)*1E-26; 
% end
% %DataOut2 = DataOut2(1:end-1,:) ./ repmat(DataOut2(end,:),model.nbVar-1,1);
% %SigmaOut2 = SigmaOut2 ./ repmat(SigmaOut2(end,end,:),[model.nbVar,model.nbVar,1]);
% DataOut2 = squeeze(SigmaOut2(end,1:end-1,:));
% %DataOut2 = squeeze(SigmaOut2(1:end-1,end,:));
% SigmaOut2 = SigmaOut2(1:end-1,1:end-1,:);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,500]); 
%Plot GMM
subplot(1,2,1); hold on; axis off; title('GMM');
plot(Data(2,:),Data(3,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(model.Mu(2:model.nbVar,:), model.Sigma(2:model.nbVar,2:model.nbVar,:), [.8 0 0], .5);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
%Plot GMR
subplot(1,2,2); hold on; axis off; title('GMR');
plot(Data(2,:),Data(3,:),'.','markersize',8,'color',[.5 .5 .5]);
%plot(DataOut(1,:),DataOut(2,:),'-','linewidth',2,'color',[0 .4 0]);
%plot(DataOut2(1,:),DataOut2(2,:),'o','color',[0 0 .4]);
plotGMM(DataOut, SigmaOut, [0 .8 0], .03);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%pause
%plotGMM(DataOut2, SigmaOut2, [0 0 .8], .03);

pause;
%plotGMM(DataOut3(1:2,:), SigmaOut3(1:2,1:2,:), [.8 0 0], .03);
plotGMM(DataOut4, SigmaOut4, [0 0 .8], .03);

figure; hold on;
clrmap = lines(model.nbStates);
for i=1:model2.nbStates
	plot(H(i,:),'-','linewidth',2,'color',clrmap(i,:));
end
axis([1 nbData 0 1.05]);
xlabel('t'); ylabel('h_i');

%print('-dpng','graphs/demo_GMR_augmSigma01.png');
pause;
close all;