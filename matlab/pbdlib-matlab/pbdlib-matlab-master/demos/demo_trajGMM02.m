function demo_trajGMM02
% Trajectory synthesis with a GMM with dynamic features (trajectory GMM), where the GMM is learned from trajectory examples.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 6; %Number of components in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E0; %Time step (without rescaling, large values such as 1 has the advantage of creating clusers based on position information)
nbSamples = 5; %Number of demonstrations
nbData = 200; %Number of datapoints in a trajectory
nbRepros = 0; %Number of stochastic reproductions

[PHI,PHI1] = constructPHI(model,nbData,nbSamples); %Construct PHI operator (big sparse matrix)


%% Load handwriting movements
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data, s(n).Data]; 
end

%Re-arrange data in vector form
x = Data(:) * 1E2; %Scale data to avoid numerical computation problem
zeta = PHI * x; %zeta is for example [x1(1), x2(1), x1d(1), x2d(1), x1(2), x2(2), x1d(2), x2d(2), ...]
Data = reshape(zeta, model.nbVarPos*model.nbDeriv, nbData*nbSamples); %Include derivatives in Data


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data,model,nbSamples);
[model, GAMMA2] = EM_GMM(Data, model);

%Compute basis functions activation
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = model.Priors(i) * gaussPDF(Data(:,1:nbData), model.Mu(:,i), model.Sigma(:,:,i));
end
H = H ./ repmat(sum(H,1),model.nbStates,1);

%Precomputation of inverse and eigencomponents (optional)
for i=1:model.nbStates
	[model.V(:,:,i), model.D(:,:,i)] = eigs(model.Sigma(:,:,i));
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:1 %nbSamples
	%Compute best path for the n-th demonstration
	[~,r(n).q] = max(GAMMA2(:,(n-1)*nbData+1:n*nbData),[],1); %works also for nbStates=1
	
	%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q
	MuQ = reshape(model.Mu(:,r(n).q), model.nbVar*nbData, 1); 
	
	Stmp = model.Sigma(:,:,r(n).q);
	SigmaQ = kron(speye(nbData), ones(model.nbVar));
	SigmaQ(logical(SigmaQ)) = Stmp(:);

% 	SigmaQ = zeros(model.nbVar*nbData);
% 	for t=1:nbData
% 		id = (t-1)*model.nbVar+1:t*model.nbVar;
% 		SigmaQ(id,id) = model.Sigma(:,:,r(n).q(t));
% 	end

	%Least squares computation method 1 (using lscov Matlab function)
	%%%%%%%%%%%%%%%%%%%
	[xhat,~,~,S] = lscov(PHI1, MuQ, SigmaQ, 'chol'); %Retrieval of data with weighted least squares solution
	r(n).Data = reshape(xhat, model.nbVarPos, nbData); %Reshape data for plotting
	
% 	%Least squares computation method 2 (most readable but not optimized)
% 	%%%%%%%%%%%%%%%%%%%
% 	PHIinvSigmaQ = PHI1' / SigmaQ;
% 	Rq = PHIinvSigmaQ * PHI1;
% 	rq = PHIinvSigmaQ * MuQ;
% 	xhat = Rq \ rq; %Can also be computed with c = lscov(Rq, rq)
% 	r(n).Data = reshape(xhat, model.nbVarPos, nbData); %Reshape data for plotting
% 	%Covariance Matrix computation of ordinary least squares estimate
% 	mse =  (MuQ'/SigmaQ*MuQ - rq'/Rq*rq) ./ ((model.nbVar-model.nbVarPos)*nbData);
% 	S = inv(Rq) * mse; 

% 	%Least squares computation method 3 (computation using Cholesky and QR decompositions, inspired by lscov code)
% 	%%%%%%%%%%%%%%%%%%%
% 	T = chol(SigmaQ)'; %SigmaQ=T*T'
% 	TA = T \ PHI1;
% 	TMuQ = T \ MuQ;
% 	[Q, R, perm] = qr(TA,0); %PHI1(:,perm)=Q*R
% 	z = Q' * TMuQ;
% 	xhat = zeros(nbData*model.nbVarPos,1);
% 	xhat(perm,:) = R \ z; %xhat=(TA'*TA)\(TA'*TMuQ)
% 	r(n).Data = reshape(xhat, model.nbVarPos, nbData); %Reshape data for plotting
% 	%Covariance Matrix computation of ordinary least squares estimate
% 	err = TMuQ - Q*z;
% 	mse = err'*err ./ (model.nbVar*nbData - model.nbVarPos*nbData);
% 	Rinv = R \ eye(model.nbVarPos*nbData);
% 	S(perm,perm) = Rinv*Rinv' .* mse; 
	
	%Rebuild covariance by reshaping S
	for t=1:nbData
		id = (t-1)*model.nbVarPos+1:t*model.nbVarPos;
		r(n).expSigma(:,:,t) = S(id,id) * nbData;
	end
end %nbSamples


%% Plot timeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 500 500]);
for m=1:model.nbVarPos
	limAxes = [1, nbData, min(Data(m,:))-1E0, max(Data(m,:))+1E0];
	subplot(model.nbVarPos,1,m); hold on;
	for n=1:1 %nbSamples
		msh=[]; x0=[];
		for t=1:nbData-1
			if size(msh,2)==0
				msh(:,1) = [t; model.Mu(m,r(n).q(t))];
			end
			if t==nbData-1 || r(n).q(t+1)~=r(n).q(t)
				msh(:,2) = [t+1; model.Mu(m,r(n).q(t))];
				sTmp = model.Sigma(m,m,r(n).q(t))^.5;
				msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
				patch(msh2(1,:), msh2(2,:), [.85 .85 .85],'edgecolor',[.7 .7 .7]);
				plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
				plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.7 .7 .7]);
				x0 = [x0 msh];
				msh=[];
			end
		end
		msh = [1:nbData, nbData:-1:1; r(n).Data(m,:)-squeeze(r(n).expSigma(m,m,:).^.5)'*1, fliplr(r(n).Data(m,:)+squeeze(r(n).expSigma(m,m,:).^.5)'*1)];
		patch(msh(1,:), msh(2,:), ones(1,size(msh,2)), [1 .4 .4],'edgecolor',[1 .2 .2],'edgealpha',.8,'facealpha',.5);
	end
	for n=1:nbSamples
		plot(1:nbData, Data(m,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]);
	end
	for n=1:1
		%plot(1:nbData, model.Mu(m,r(n).q), '-','lineWidth',3.5,'color',[.8 0 0]);
		plot(1:nbData, r(n).Data(m,:), '-','lineWidth',3,'color',[.8 0 0]);
	end
	for n=1:nbRepros
		plot(1:nbData, rnew(n).Data(m,:), '-','lineWidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	ylabel(['$x_' num2str(m) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
end
%print('-dpng','graphs/demo_trajGMM02a.png');


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if model.nbVarPos>1
	figure('position',[520 10 500 500]); hold on;
% 	plotGMM(model.Mu([1,2],:), model.Sigma([1,2],[1,2],:), [.5 .5 .5],.8);
% 	for n=1:1 %nbSamples
% 		plotGMM(r(n).Data([1,2],:), r(n).expSigma([1,2],[1,2],:), [1 .2 .2],.1);
% 	end
	for n=1:nbSamples
		plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]); 
	end
	for n=1:1
		%plot(model.Mu(1,r(n).q), model.Mu(2,r(n).q), '-','lineWidth',3.5,'color',[.8 0 0]);
		plot(r(n).Data(1,:), r(n).Data(2,:), '-','lineWidth',3,'color',[.8 0 0]);
	end
	for n=1:nbRepros
		plot(rnew(n).Data(1,:), rnew(n).Data(2,:), '-','lineWidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	set(gca,'xtick',[],'ytick',[]); axis equal; axis square;
	xlabel(['$x_1$'],'interpreter','latex','fontsize',18);
	ylabel(['$x_2$'],'interpreter','latex','fontsize',18);
end
%print('-dpng','-r300','graphs/demo_trajGMM02b.png');


%% Plot covariance structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[1030 10 500 500]); hold on; axis off;
colormap(flipud(gray));
pcolor(abs(S)); 
axis square; axis([1 nbData*model.nbVarPos 1 nbData*model.nbVarPos]); axis ij; shading flat;

% %Visualize PHI matrix
% figure('PaperPosition',[0 0 4 8],'position',[10 10 400 650],'name','PHI1'); 
% axes('Position',[0.01 0.01 .98 .98]); hold on; set(gca,'linewidth',2); 
% colormap(flipud(gray));
% pcolor([abs(PHI1) zeros(size(PHI1,1),1); zeros(1,size(PHI1,2)+1)]); %dummy values for correct display
% shading flat; axis ij; axis equal tight;
% set(gca,'xtick',[],'ytick',[]);

pause;
close all;