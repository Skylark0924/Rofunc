function demo_TPGMM_bimanualReaching01
% Time-invariant task-parameterized GMM applied to a bimanual reaching task.
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
nbData = 100;
for h=1:2
	hand(h).model.nbStates = 2; %Number of Gaussians in the GMM
	hand(h).model.nbFrames = 2; %Number of candidate frames of reference
	hand(h).model.nbVarIn = 2; %Input dimension (position of object)
	hand(h).model.nbVarOut = 2; %Output dimension (position of the hand)
	hand(h).model.nbVar = hand(h).model.nbVarIn + hand(h).model.nbVarOut;
end


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posObj = [mvnrnd([0.2;-0.2],eye(2)*0.1^2,nbData/2)', mvnrnd([0.2;0.2],eye(2)*0.1^2,nbData/2)'];
hand(1).Data0 = [posObj(:,1:nbData/2)+randn(2,nbData/2)*1E-2, repmat([0.2;-0.2],1,nbData/2)+randn(2,nbData/2)*1E-2];
hand(2).Data0 = [repmat([0.2;0.2],1,nbData/2)+randn(2,nbData/2)*1E-2, posObj(:,nbData/2+1:end)+randn(2,nbData/2)*1E-2];

%Set task parameters
for h=1:2
	for t=1:nbData
		hand(h).model.p(1,t).b = [zeros(hand(h).model.nbVarIn,1); posObj(:,t)]; %param1 (object)
		hand(h).model.p(2,t).b = zeros(hand(h).model.nbVar,1); %param2 (robot)
		for m=1:hand(h).model.nbFrames
			hand(h).model.p(m,t).A = eye(hand(h).model.nbVar);
			hand(h).model.p(m,t).invA = eye(hand(h).model.nbVar); %Precomputation of inverse
		end
	end
end

%Observation of data from the perspective of the frames
for h=1:2
	for m=1:hand(h).model.nbFrames
		for t=1:nbData
			hand(h).Data(:,m,t) = hand(h).model.p(m,t).invA * ([posObj(:,t); hand(h).Data0(:,t)] - hand(h).model.p(m,t).b); 
			% -> hand(h).Data(1:2,m,t) = posObj(:,t);
		end
	end
end


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of TP-GMM with EM');
for h=1:2
	hand(h).model = init_tensorGMM_kmeans(hand(h).Data, hand(h).model); 
	hand(h).model = EM_tensorGMM(hand(h).Data, hand(h).model);
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Reproduction...\n');
%Generate object trajectory
aTmp = linspace(0,4*pi,nbData);
rPosObj = [ones(1,nbData).*aTmp*(0.02/4*pi)+sin(aTmp)*0.05+0.1; cos(aTmp)*0.2]; 

for t=1:nbData
	for h=1:2
		%Frame1 (object)
		pTmp(1).A = eye(4);
		pTmp(1).b = [zeros(2,1); rPosObj(:,t)];
		%Frame2 (robot)
		pTmp(2).A = eye(4);
		pTmp(2).b = zeros(4,1);
		
		%GMR with linearly transformed Gaussians followed by Gaussian product
		SigmaTmp = zeros(2);
		MuTmp = zeros(2,1);
		for m=1:hand(h).model.nbFrames
			mtmp.nbStates = hand(h).model.nbStates;
			mtmp.Priors = hand(h).model.Priors;
			for i=1:mtmp.nbStates
				mtmp.Mu(:,i) = pTmp(m).A * squeeze(hand(h).model.Mu(:,m,i)) + pTmp(m).b;
				mtmp.Sigma(:,:,i) = pTmp(m).A * squeeze(hand(h).model.Sigma(:,:,m,i)) * pTmp(m).A';
			end
			[MuOut, SigmaOut] = GMR(mtmp, rPosObj(:,t), 1:2, 3:4);
			SigmaTmp = SigmaTmp + inv(SigmaOut);
			MuTmp = MuTmp + SigmaOut\MuOut;
		end
		hand(h).rData(:,t) = SigmaTmp \ MuTmp;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,50,2200,800]);
clrmap = lines(2);
ttl={'Frame 1 (object)','Frame 2 (robot)'};

%DEMOS
subplot(1,hand(1).model.nbFrames+2,1); hold on; box on; title('Demonstrations');
for h=1:2
	plot(hand(h).Data0(1,:), hand(h).Data0(2,:), '.','markersize',15,'color',clrmap(h,:));
end
plot(posObj(1,:), posObj(2,:), 'ko');
legend('left hand','right hand','object');
axis equal; set(gca,'xtick',[],'ytick',[]);

%FRAMES
for m=1:hand(1).model.nbFrames
	subplot(1,hand(1).model.nbFrames+2,1+m); hold on; grid on; box on; title(ttl{m});
	for h=1:2
		plot(squeeze(hand(h).Data(3,m,:)), squeeze(hand(h).Data(4,m,:)), '.','markersize',15,'color',clrmap(h,:));
		%plotGMM(squeeze(hand(h).model.Mu(1:2,m,:)), squeeze(hand(h).model.Sigma(1:2,1:2,m,:)), [.5 1 .5], .4);
		plotGMM(squeeze(hand(h).model.Mu(3:4,m,:)), squeeze(hand(h).model.Sigma(3:4,3:4,m,:)), [.5 .5 .5], .4);
	end
	axis equal; set(gca,'xtick',[0],'ytick',[0]);
end

% %REPRO
% subplot(1,hand(1).model.nbFrames+2,hand(1).model.nbFrames+2); hold on; grid on; box on; title('Reproduction');
% for h=1:2
% 	plot(hand(h).rData(1,:), hand(h).rData(2,:), '.','linewidth',1.5,'color',clrmap(h,:));
% end
% plot(rPosObj(1,:), rPosObj(2,:), 'ko');
% axis equal; set(gca,'xtick',[0],'ytick',[0]);

%REPRO ANIM
subplot(1,hand(1).model.nbFrames+2,hand(1).model.nbFrames+2); hold on; grid on; box on; title('Reproduction');
axis equal; axis([min(rPosObj(1,:)) max(rPosObj(1,:)) min(rPosObj(2,:)) max(rPosObj(2,:))]); set(gca,'xtick',[0],'ytick',[0]);
for t=1:nbData
	for h=1:2
		plot(hand(h).rData(1,t), hand(h).rData(2,t), '.','markersize',15,'color',clrmap(h,:));
	end
	plot(rPosObj(1,t), rPosObj(2,t), 'ko');
	pause(0.05);
end

%print('-dpng','graphs/demo_TPGMM_bimanualReaching01.png');
pause;
close all;