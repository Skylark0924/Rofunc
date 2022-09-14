function demo_OC_LQT02
% Batch computation of linear quadratic tracking problem, by tracking position and velocity references.
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
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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
% nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbNewRepros = 10; %Number of stochastically generated reproductions
nbData = 200; %Number of datapoints

model.nbStates = 1; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 1; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E-2; %Time step duration
model.rfactor = 1E-3; %model.dt^model.nbDeriv;	%Control cost in LQR

%Control cost matrix
R = speye(model.nbVarPos) * model.rfactor;
R = kron(speye(nbData-1),R);


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); 
% Bc1d = [zeros(model.nbDeriv-1,1); 1];
% % A = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% % B = kron(Bc1d*model.dt, eye(model.nbVarPos));
% 
% Ac = kron(Ac1d, eye(model.nbVarPos));
% Bc = kron(Bc1d, eye(model.nbVarPos));
% A = eye(model.nbVar) + Ac * model.dt;
% B = Bc * model.dt;

%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, speye(model.nbVarPos)); %Discrete nD
B = kron(B1d, speye(model.nbVarPos)); %Discrete nD

% %Conversion with control toolbox
% Ac1d = diag(ones(model.nbDeriv-1,1),1); %Continuous 1D
% Bc1d = [zeros(model.nbDeriv-1,1); 1]; %Continuous 1D
% Cc1d = [1, zeros(1,model.nbDeriv-1)]; %Continuous 1D
% sysd = c2d(ss(Ac1d,Bc1d,Cc1d,0), model.dt);
% A = kron(sysd.a, eye(model.nbVarPos)); %Discrete nD
% B = kron(sysd.b, eye(model.nbVarPos)); %Discrete nD

%Build Sx and Su transfer matrices
Su = sparse(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1), speye(model.nbVar));
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end

% %Alternative computation of Su by separating A and B
% Su = sparse(model.nbVar*nbData, model.nbVar*(nbData-1));
% M = eye(model.nbVar);
% for n=2:nbData
% 	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
% 	id2 = 1:(n-1)*model.nbVar;
% 	Su(id1,id2) = M;
% 	M = [A*M(:,1:model.nbVar), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
% end
% Su = Su * kron(eye(nbData-1),B);


% %% Load handwriting data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('data/2Dletters/B.mat');
% Data=[];
% for n=1:nbSamples
% 	s(n).Data=[];
% 	for m=1:model.nbDeriv
% 		if m==1
% 			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
% 		else
% 			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
% 		end
% 		s(n).Data = [s(n).Data; dTmp];
% 	end
% 	Data = [Data s(n).Data]; 
% end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('Parameters estimation...');
% %model = init_GMM_kmeans(Data, model);
% model = init_GMM_kbins(Data, model, nbSamples);

% %Initialization based on position data
% model0 = init_GMM_kmeans(Data(1:model.nbVarPos,:), model);
% [~,GAMMA2] = EM_GMM(Data(1:model.nbVarPos,:), model0);
% model.Priors = model0.Priors;
% for i=1:model.nbStates
% 	model.Mu(:,i) = Data * GAMMA2(i,:)';
% 	DataTmp = Data - repmat(model.Mu(:,i),1,nbData*nbSamples);
% 	model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp';
% end

% %Refinement of parameters
% [model, H] = EM_GMM(Data, model);


%Compute activation functions
MuRBF(1,:) = linspace(1, nbData, model.nbStates);
SigmaRBF = 1E2; 
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = gaussPDF(1:nbData, MuRBF(:,i), SigmaRBF);
end
H = H ./ repmat(sum(H,1),model.nbStates,1);

%Debug
model.Mu = [(rand(model.nbVarPos,model.nbStates)-0.5).*2E1; zeros(model.nbVar-model.nbVarPos,model.nbStates)];
model.Sigma = repmat(blkdiag(eye(model.nbVarPos).*1E0,eye(model.nbVar-model.nbVarPos).*1E2), [1,1,model.nbStates]);


%Precomputation of inverse and eigencomponents (optional)
for i=1:model.nbStates
	[model.V(:,:,i), model.D(:,:,i)] = eigs(model.Sigma(:,:,i));
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
	[V,D]= eig(model.invSigma(:,:,i));
	model.U(:,:,i) = V * D.^.5;
% 	model.U(:,:,i) = sqrtm(model.invSigma(:,:,i));
end

%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q, see Eq. (27)
MuQ = reshape(model.Mu(:,qList), model.nbVar*nbData, 1);

% %Version 1 (fastest)
% S = model.invSigma(:,:,qList);
% Q = kron(eye(nbData), ones(model.nbVar)); %speye
% Q(logical(Q)) = S(:);
% % rank(Q)

% %Version 2
% Q = zeros(model.nbVar*nbData);
% for t=1:nbData
% 	id = (t-1)*model.nbVar+1:t*model.nbVar;
% 	Q(id,id) = model.invSigma(:,:,qList(t));
% end	

% %Version 3
% Q = kron(ones(nbData,1), reshape(model.invSigma(:,:,qList), model.nbVar, model.nbVar*nbData)) .* kron(speye(nbData), ones(model.nbVar));

%Version 4 (kernel approach forming a sparse trajectory distribution)
A = [];
for i=1:nbData
	A = [A; model.U(:,:,qList(i))];
end
H = exp(-1E-2 .* pdist2([1:nbData]', [1:nbData]').^2);
H = kron(H', eye(model.nbVar));
Q = (A*A') .* H;
% rank(Q)


% %Create single Gaussian N(MuQ,SigmaQ) based on h	
% h = H(:,1:nbData);
% h = h ./ repmat(sum(h,1),model.nbStates,1);
% MuQ = zeros(model.nbVar*nbData,1);
% Q = zeros(model.nbVar*nbData);
% for t=1:nbData
% 	id = (t-1)*model.nbVar+1:t*model.nbVar;
% 	for i=1:model.nbStates
% 		MuQ(id) = MuQ(id) + model.Mu(:,i) * h(i,t);
% 		Q(id,id) = SigmaQ(id,id) + model.invSigma(:,:,i) * h(i,t);
% 	end
% end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Set matrices to compute the damped weighted least squares estimate, see Eq. (37)
SuInvSigmaQ = Su' * Q;
Rq = SuInvSigmaQ * Su + R;
%M = (SuInvSigmaQ * Su + R) \ SuInvSigmaQ; 
%M = inv(Su' / SigmaQ * Su + R) * Su' /SigmaQ;

%Reproductions
for n=1:nbRepros
	x0 = [randn(model.nbVarPos,1)*0E0; zeros(model.nbVar-model.nbVarPos,1)];
	%X = Data(:,1); 
 	rq = SuInvSigmaQ * (MuQ-Sx*x0);
 	u = Rq \ rq; %Can also be computed with u = lscov(Rq, rq);
	%[u,~,~,S] = lscov(Rq, rq);
	%u = M * (MuQ-Sx*X);
	r(n).x = reshape(Sx*x0+Su*u, model.nbVar, nbData); %Reshape data for plotting
	r(n).u = reshape(u, model.nbVarPos, nbData-1); %Reshape data for plotting
	
% 	%Simulate plant
% 	r(n).u = reshape(u, model.nbVarPos, nbData-1);
% 	for t=1:nbData-1
% 		%id = (t-1)*model.nbVar+1:t*model.nbVar;
% 		%id2 = (t-1)*model.nbVarPos+1:t*model.nbVarPos;
% 		%r(n).u(:,t) = M(id2,id) * (MuQ(id) - X);
% 		X = A * X + B * r(n).u(:,t);
% 		r(n).x(:,t) = X;
% 	end
end

% mse =  abs(MuQ'*Q*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
uSigma = inv(Rq); % * mse; % + eye((nbData-1)*model.nbVarU) * 1E-10;
xSigma = Su * uSigma * Su';


%% Stochastic sampling by exploiting the GMM representation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbEigs = 2; %Number of principal eigencomponents to keep
x0 = zeros(model.nbVar,1); %Data(:,1);
SuInvSigmaQ = Su' * Q;
Rq = SuInvSigmaQ * Su + R;
for n=1:nbNewRepros
	N = randn(nbEigs,model.nbStates) * 1E-1; %Noise on all components
	%N = [randn(nbEigs,1), zeros(nbEigs,model.nbStates-1)] * 2E0; %Noise on first component
	%N = [zeros(nbEigs,model.nbStates-3), randn(nbEigs,1), zeros(nbEigs,2)] * 1E0; %Noise on second component
	for i=1:model.nbStates
		MuTmp(:,i) = model.Mu(:,i) + model.V(:,1:nbEigs,i) * model.D(1:nbEigs,1:nbEigs,i).^.5 * N(:,i);
	end
	MuQ2 = reshape(MuTmp(:,qList), model.nbVar*nbData, 1); 
	rq = SuInvSigmaQ * (MuQ2-Sx*x0);
	u = Rq \ rq;
	rnew(n).x = reshape(Sx*x0+Su*u, model.nbVar, nbData); %Reshape data for plotting
	rnew(n).u = reshape(u, model.nbVarPos, nbData-1); %Reshape data for plotting
end


% %% Stochastic sampling by exploiting distribution on x
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %nbEigs = 3; %Number of principal eigencomponents to keep
% X = Data(:,1);
% SuInvSigmaQ = Su' * Q;
% Rq = SuInvSigmaQ * Su + R;
% rq = SuInvSigmaQ * (MuQ-Sx*X);
% u = Rq \ rq;
% Mu = Sx * X + Su * u;
% mse = 1; %abs(MuQ'/SigmaQ*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
% Sigma = Su*(Rq\Su') * mse + eye(nbData*model.nbVar) * 0E-10;
% [V,D] = eigs(Sigma);
% %[V,D] = eig(Sigma);
% for n=1:nbNewRepros
% 	%xtmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1);
% 	xtmp = real(Mu + V * D.^.5 * randn(size(D,1),1) * 8.9);
% 	rnew(n).x = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
% end


% %% Stochastic sampling by exploiting distribution on u
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbEigs = 3; %Number of principal eigencomponents to keep
% X = Data(:,1);
% SuInvSigmaQ = Su' * Q;
% Rq = SuInvSigmaQ * Su + R;
% rq = SuInvSigmaQ * (MuQ-Sx*X);
% Mu = Rq \ rq;
% mse =  abs(MuQ'*Q*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
% Sigma = inv(Rq) * mse + eye((nbData-1)*model.nbVarPos) * 1E-10;
% %[V,D] = eigs(Sigma);
% [V,D] = eig(Sigma);
% for n=1:nbNewRepros
% 	%utmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1);
% 	utmp = real(Mu + V * D.^.5 * randn(size(D,1),1) * 0.2);
% 	xtmp = Sx * X + Su * utmp;
% 	rnew(n).x = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
% end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 600 600],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [.8 .8 .8]);
% for n=1:nbSamples
% 	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
% end
% %Plot uncertainty
% for t=1:nbData
% 	plotGMM(r(1).x(1:2,t), xSigma(model.nbVar*(t-1)+[1,2],model.nbVar*(t-1)+[1,2]), [.8 0 0], .1);
% end	
%Plot reproduction samples
for n=1:nbNewRepros
	plot(rnew(n).x(1,:), rnew(n).x(2,:), '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
end
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[.8 0 0]);
end
axis equal; 

% figure('position',[10 700 600 600],'color',[1 1 1],'name','x1-dx1 plot'); hold on; axis off;
% plotGMM(model.Mu([1,3],:), model.Sigma([1,3],[1,3],:), [.8 .8 .8]);
% % for n=1:nbSamples
% % 	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(3,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.7 .7 .7]);
% % end	
% % %Plot uncertainty
% % for t=1:nbData
% % 	plotGMM(r(1).x([1,3],t), xSigma(model.nbVar*(t-1)+[1,3],model.nbVar*(t-1)+[1,3]), [.8 0 0], .1);
% % end
% %Plot reproduction samples
% for n=1:nbNewRepros
% 	plot(rnew(n).x(1,:), rnew(n).x(3,:), '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
% end
% for n=1:nbRepros
% 	plot(r(n).x(1,:), r(n).x(3,:), '-','linewidth',2,'color',[.8 0 0]);
% end
% plot([min(r(1).x(1,:)),max(r(1).x(1,:))], [0,0], 'k:');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'};
figure('position',[620 10 1000 850],'color',[1 1 1]); 
for j=1:model.nbVar
	subplot(model.nbVar+model.nbVarPos+1,1,j); hold on;
	limAxes = [1, nbData, min(r(1).x(j,:))-8E0, max(r(1).x(j,:))+8E0];
	msh=[]; x0=[];
	for t=1:nbData-1
		if size(msh,2)==0
			msh(:,1) = [t; model.Mu(j,qList(t))];
		end
		if t==nbData-1 || qList(t+1)~=qList(t)
			i = qList(t);
			msh(:,2) = [t+1; model.Mu(j,i)];
			sTmp = model.Sigma(j,j,qList(t))^.5;
			msh2 = [msh(:,1)+[0;sTmp], msh(:,2)+[0;sTmp], msh(:,2)-[0;sTmp], msh(:,1)-[0;sTmp], msh(:,1)+[0;sTmp]];
			patch(msh2(1,:), msh2(2,:), [.7 .7 .7],'edgecolor',[.6 .6 .6],'facealpha', .4, 'edgealpha', .4);
			plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.6 .6 .6]);
			if msh(1,1)>1
				plot([msh(1,1) msh(1,1)], limAxes(3:4), ':','linewidth',1,'color',[.5 .5 .5]);
			end
			x0 = [x0 msh];
			msh=[];
		end
	end
% 	for n=1:nbSamples
% 		plot(Data(j,(n-1)*nbData+1:n*nbData), '-','linewidth',.5,'color',[.6 .6 .6]);
% 	end

% 	%Plot uncertainty
% 	id = j:model.nbVar:model.nbVar*nbData;
% 	S = diag(xSigma(id,id));
% 	patch([1:nbData nbData:-1:1], [r(1).x(j,:)-S(:)' fliplr(r(1).x(j,:)+S(:)')], [.8 0 0],'edgecolor',[.6 0 0],'facealpha', .2, 'edgealpha', .2);
	%Plot reproduction samples
	for n=1:nbNewRepros
		plot(rnew(n).x(j,:), '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	for n=1:nbRepros
		plot(r(n).x(j,:), '-','linewidth',2,'color',[.8 0 0]);
	end
	if j<7
		ylabel(labList{j},'fontsize',14,'interpreter','latex');
	end
	axis(limAxes);
end

%Speed profile
if model.nbDeriv>1
	subplot(model.nbVar+model.nbVarPos+1,1,model.nbVar+1); hold on;
% 	for n=1:nbSamples
% 		sp = sqrt(Data(3,(n-1)*nbData+1:n*nbData).^2 + Data(4,(n-1)*nbData+1:n*nbData).^2);
% 		plot(sp, '-','linewidth',.5,'color',[.6 .6 .6]);
% 	end
	for n=1:nbNewRepros
		sp = sqrt(rnew(n).x(3,:).^2 + rnew(n).x(4,:).^2);
		plot(sp, '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	for n=1:nbRepros
		sp = sqrt(r(n).x(3,:).^2 + r(n).x(4,:).^2);
		plot(sp, '-','linewidth',2,'color',[.8 0 0]);
	end
	ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
end

%Control profile
for j=1:model.nbVarPos
	subplot(model.nbVar+model.nbVarPos+1,1,model.nbVar+1+j); hold on;
% 	%Plot uncertainty
% 	id = j:model.nbVarPos:model.nbVarPos*(nbData-1);
% 	S = diag(uSigma(id,id));
% 	patch([1:nbData-1 nbData-1:-1:1], [r(1).u(j,:)-S(:)' fliplr(r(1).u(j,:)+S(:)')], [.8 0 0],'edgecolor',[.6 0 0],'facealpha', .2, 'edgealpha', .2);
	%Plot reproduction samples
	for n=1:nbNewRepros
		plot(rnew(n).u(j,:), '-','linewidth',1,'color',max(min([0 .7+randn(1)*1E-1 0],1),0));
	end
	for n=1:nbRepros
		plot(r(n).u(j,:), '-','linewidth',2,'color',[.8 0 0]);
	end
	ylabel(['$u_' num2str(j) '$'],'fontsize',14,'interpreter','latex');
end
xlabel('$t$','fontsize',14,'interpreter','latex');

%print('-dpng','graphs/demo_LQT02.png');


%% Plot covariance of control trajectory distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[1630 10 500 1000],'color',[1 1 1],'name','Covariances'); 
subplot(2,1,1); hold on; box on; set(gca,'linewidth',2); title('Q','fontsize',14);
colormap(gca, flipud(gray));
pcolor(abs(Q));
set(gca,'xtick',[1,size(uSigma,1)],'ytick',[1,size(uSigma,1)]);
axis square; axis([1 size(uSigma,1) 1 size(uSigma,1)]); shading flat;

subplot(2,1,2); hold on; box on; set(gca,'linewidth',2); title('S^x','fontsize',14);
colormap(gca, flipud(gray));
pcolor(abs(xSigma));
set(gca,'xtick',[1,size(xSigma,1)],'ytick',[1,size(xSigma,1)]);
axis square; axis([1 size(xSigma,1) 1 size(xSigma,1)]); shading flat;

% %Visualize Su matrix
% figure('PaperPosition',[0 0 4 8],'position',[10 10 400 650],'name','Su'); 
% axes('Position',[0.01 0.01 .98 .98]); hold on; set(gca,'linewidth',2); 
% colormap(flipud(gray));
% pcolor([abs(Su) zeros(size(Su,1),1); zeros(1,size(Su,2)+1)]); %dummy values for correct display
% shading flat; axis ij; axis equal tight;
% set(gca,'xtick',[],'ytick',[]);

pause;
close all;