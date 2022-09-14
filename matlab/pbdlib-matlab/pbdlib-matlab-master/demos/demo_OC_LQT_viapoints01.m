function demo_OC_LQT_viapoints01
% Batch LQT with viapoints and a double integrator system, and an encoding of position and velocity.
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions in new situations
nbNewRepros = 0; %Number of stochastically generated reproductions

model.nbStates = 3; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector

model.dt = 1E-2; %Time step duration
model.rfactor = 1E-5;	%Control cost in LQR
% model.params_diagRegFact = 1E-8; %GMM regularization factor

nbData = model.nbStates * 30; %Number of datapoints

R = speye((nbData-1)*model.nbVarPos) * model.rfactor; %Control cost matrix


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Integration with Euler method 
% Ac1d = diag(ones(model.nbDeriv-1,1),1); 
% Bc1d = [zeros(model.nbDeriv-1,1); 1];
% A = kron(eye(model.nbDeriv)+Ac1d*model.dt, eye(model.nbVarPos)); 
% B = kron(Bc1d*model.dt, eye(model.nbVarPos));

%Integration with higher order Taylor series expansion
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD

%Build Sx and Su transfer matrices 
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


% %% Generate keypoints from handwriting data
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('data/2Dletters/M.mat');
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
% 
% %Refinement of parameters
% [model, H] = EM_GMM(Data, model);

% %Compute function activation
% MuRBF(1,:) = linspace(1, nbData, model.nbStates);
% SigmaRBF = 1E2; 
% H = zeros(model.nbStates,nbData);
% for i=1:model.nbStates
% 	H(i,:) = gaussPDF(1:nbData, MuRBF(:,i), SigmaRBF);
% end
% H = H ./ repmat(sum(H,1),model.nbStates,1);
% %Set list of states according to first demonstration (alternatively, an HSMM can be used)
% [~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

qList = kron(1:model.nbStates, ones(1, nbData/model.nbStates));

%Debug
% model.Mu = [10; 10; zeros(model.nbVarPos,1)];
% model.Mu = [(rand(model.nbVarPos,model.nbStates)-0.5).*2E1; zeros(model.nbVar-model.nbVarPos,model.nbStates)];
model.Mu = [[5 -10 -7; 10 12 -10]; zeros(model.nbVar-model.nbVarPos,model.nbStates)];

model.Sigma = repmat(blkdiag(eye(model.nbVarPos).*1E-2, eye(model.nbVar-model.nbVarPos).*1E2), [1,1,model.nbStates]);
model.Sigma(1:model.nbVarPos,1:model.nbVarPos,2) = eye(model.nbVarPos).*1E0;

%Precomputation of inverses
for i=1:model.nbStates
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q
MuQ = zeros(model.nbVar*nbData, 1); 
Q = zeros(model.nbVar*nbData);
qCurr = qList(1);
xDes = []; 
for t=1:nbData
	if qCurr~=qList(t) || t==nbData
		id = (t-1)*model.nbVar+1:t*model.nbVar;
		MuQ(id) = model.Mu(:,qCurr); 
		Q(id,id) = model.invSigma(:,:,qCurr);
		%List of viapoints with time and error information (for plots)
		xDes = [xDes, [t; model.Mu(:,qCurr); diag(model.Sigma(:,:,qCurr)).^.5 .* 1E1]];	
		qCurr = qList(t);
	end
end	


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reproductions
for n=1:nbRepros
	x0 = zeros(model.nbVar,1); %Data(:,1) + [randn(model.nbVarPos,1)*0E0; zeros(model.nbVarPos,1)];
 	u = (Su' * Q * Su + R) \ Su' * Q * (MuQ-Sx*x0); %Can also be computed with u = lscov(Rq, rq);
	r(n).x = reshape(Sx*x0+Su*u, model.nbVar, nbData); %Reshape data for plotting
	r(n).u = reshape(u, model.nbVarPos, nbData-1); %Reshape data for plotting
end

% mse =  abs(MuQ'*Q*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
uSigma = inv(Su' * Q * Su + R); % * mse; % + eye((nbData-1)*model.nbVarU) * 1E-10;
xSigma = Su * uSigma * Su';
xQ = Su * (Su' * Q * Su + R) * Su';
SuQSu = Su' * Q * Su;


% %% Stochastic sampling by exploiting distribution on x
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %nbEigs = 3; %Number of principal eigencomponents to keep
% x0 = x0 + 10;
% SuInvSigmaQ = Su' * Q;
% Rq = SuInvSigmaQ * Su + R;
% rq = SuInvSigmaQ * (MuQ-Sx*x0);
% u = Rq \ rq;
% Mu = Sx * x0 + Su * u;
% mse = 1; %abs(MuQ'/SigmaQ*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
% Sigma = Su*(Rq\Su') * mse + eye(nbData*model.nbVar) * 0E-10;
% [V,D] = eigs(Sigma);
% %[V,D] = eig(Sigma);
% for n=1:nbNewRepros
% 	%xtmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1);
% 	xtmp = real(Mu + V * D.^.5 * randn(size(D,1),1) * 1E0);
% 	rnew(n).x = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
% end


% %% Stochastic sampling by exploiting distribution on u
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %nbEigs = 3; %Number of principal eigencomponents to keep
% x0 = x0 + 10;
% SuInvSigmaQ = Su' * Q;
% Rq = SuInvSigmaQ * Su + R;
% rq = SuInvSigmaQ * (MuQ-Sx*x0);
% Mu = Rq \ rq;
% mse =  abs(MuQ'*Q*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
% Sigma = inv(Rq) * mse + eye((nbData-1)*model.nbVarPos) * 1E-10;
% %[V,D] = eigs(Sigma);
% [V,D] = eig(Sigma);
% for n=1:nbNewRepros
% 	%utmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1);
% 	utmp = real(Mu + V * D.^.5 * randn(size(D,1),1) * 1E0);
% 	xtmp = Sx * x0 + Su * utmp;
% 	rnew(n).x = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
% end


%% Stochastic sampling by exploiting nullspace structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x0 = x0 + 10;
% u = (Su' * Q * Su + R) \ Su' * Q * (MuQ-Sx*x0);
pinvSu = (Su' * Q * Su + R) \ Su' * Q;
N = eye(size(Su,2)) - pinvSu * Su;

% Q2 = kron(eye(nbData), diag([0; 1; 0; 0]));
% MuQ2 = kron(ones(nbData,1), [0; -3; 0; 0]);

Q2 = kron(eye(nbData), diag([0; 0; 1E1; 1E1]));
tl = linspace(0,40*pi,nbData);
v = [zeros(2,nbData); cos(tl); sin(tl)] .*5E1;
MuQ2 = v(:);

for n=1:nbNewRepros
% 	utmp = u + N * (rand(model.nbVarPos*(nbData-1),1)-.5).*1E3;
% 	utmp = u + N * kron(ones(nbData-1,1), [0; 1E3]);
	utmp = (Su'*Q*Su + N*Su'*Q2*Su*N' + R) \  (Su'*Q*(MuQ-Sx*x0) + N*Su'*Q2*(MuQ2-Sx*x0));

	xtmp = Sx * x0 + Su * utmp;
	rnew(n).x = reshape(xtmp, model.nbVar, nbData); %Reshape data for plotting
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%x1-x2 
figure('position',[10 10 500 500],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
%Plot uncertainty
for t=1:nbData
	plotGMM(r(1).x(1:2,t), xSigma(model.nbVar*(t-1)+[1,2],model.nbVar*(t-1)+[1,2]).*1E1, [.2 .2 .2], .1);
end	
%Plot reproduction samples
for n=1:nbNewRepros
	colTmp = [.6 .6 .6] + (rand(1)-.5).*.4;
	plot(rnew(n).x(1,:), rnew(n).x(2,:), '-','linewidth',1,'color',colTmp);
	plot(rnew(n).x(1,1), rnew(n).x(2,1), '.','markersize',15,'color',colTmp);
end
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[0 0 0]);
	plot(r(n).x(1,1), r(n).x(2,1), '.','markersize',15,'color',[0 0 0]);
end
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:).*1E1, [.8 0 0], .3);
plot(model.Mu(1,:), model.Mu(2,:), '.','markersize',15,'color',[.8 0 0]);
axis equal;
% print('-dpng','graphs/batchLQR_x_dx01.png');

% %x1-dx1
% if model.nbDeriv>1
% figure('position',[10 700 600 600],'color',[1 1 1],'name','x1-dx1 plot'); hold on; axis off;
% %Plot uncertainty
% for t=1:nbData
% 	plotGMM(r(1).x([1,3],t), xSigma(model.nbVar*(t-1)+[1,3],model.nbVar*(t-1)+[1,3]).*1E0, [.2 .2 .2], .02);
% end
% for n=1:nbNewRepros
% 	colTmp = [.6 .6 .6] + (rand(1)-.5).*.4;
% 	plot(rnew(n).x(1,:), rnew(n).x(3,:), '-','linewidth',1,'color',colTmp);
% 	plot(rnew(n).x(1,1), rnew(n).x(3,1), '.','markersize',15,'color',colTmp);
% end
% for n=1:nbRepros
% 	plot(r(n).x(1,:), r(n).x(3,:), '-','linewidth',2,'color',[0 0 0]);
% 	plot(r(n).x(1,1), r(n).x(3,1), '.','markersize',15,'color',[0 0 0]);
% end
% plot([min(r(1).x(1,:)),max(r(1).x(1,:))], [0,0], 'k:');
% % plotGMM(model.Mu([1,3],:), model.Sigma([1,3],[1,3],:), [.8 0 0], .3);
% plot(model.Mu(1,:), model.Mu(3,:), '.','markersize',15,'color',[.8 0 0]);
% end %model.nbDeriv

%u1-u2
figure('position',[260 560 250 250],'color',[1 1 1],'name','u1-u2 plot'); hold on; axis off;
%Plot uncertainty
for t=1:nbData-1
	plotGMM(r(1).u(1:2,t), uSigma(model.nbVarPos*(t-1)+[1,2],model.nbVarPos*(t-1)+[1,2]).*1E0, [.2 .2 .2], .02);
end
for n=1:nbRepros
	plot(r(n).u(1,:), r(n).u(2,:), '-','linewidth',1,'color',[0 0 0]);
	plot(r(n).u(1,1), r(n).u(2,1), '.','markersize',20,'color',[0 0 0]);
end

% print('-dpng','graphs/batchLQR_x_dx02.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'};
figure('position',[520 10 600 800],'color',[1 1 1]); 
for j=1:model.nbVar
	subplot(model.nbVar+model.nbVarPos+1,1,j); hold on;
	if j>model.nbVarPos
		plot([1,nbData],[0,0],':','color',[.5 .5 .5]);
	end
	%Plot uncertainty
	id = j:model.nbVar:model.nbVar*nbData;
	S = diag(xSigma(id,id)).*1E1;
	patch([1:nbData nbData:-1:1], [r(1).x(j,:)-S(:)' fliplr(r(1).x(j,:)+S(:)')], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
	%Plot reproduction samples
	for n=1:nbNewRepros
		plot(rnew(n).x(j,:), '-','linewidth',1,'color',[.6 .6 .6]+(rand(1)-.5).*.4);
	end
	for n=1:nbRepros
		plot(r(n).x(j,:), '-','linewidth',1,'color',[0 0 0]);
	end
	for t=1:size(xDes,2)
		errorbar(xDes(1,t), xDes(1+j,t), xDes(1+model.nbVar+j,t), 'color',[.8 0 0]);
		plot(xDes(1,t), xDes(1+j,t), '.','markersize',15,'color',[.8 0 0]);
	end
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
	xlabel('$t$','fontsize',14,'interpreter','latex');
	set(gca,'xtick',[],'ytick',[]);
	axis tight;
end

%Speed profile
if model.nbDeriv>1
	subplot(model.nbVar+model.nbVarPos+1,1,model.nbVar+1); hold on;
% 	for n=1:nbSamples
% 		sp = sqrt(Data(3,(n-1)*nbData+1:n*nbData).^2 + Data(4,(n-1)*nbData+1:n*nbData).^2);
% 		plot(sp, '-','linewidth',.5,'color',[.6 .6 .6]);
% 	end
	for n=1:nbRepros
		sp = sqrt(r(n).x(3,:).^2 + r(n).x(4,:).^2);
		plot(sp, '-','linewidth',1,'color',[.6 .6 .6]+(rand(1)-.5).*.4);
	end
	plot(xDes(1,:), zeros(size(xDes,2),1), '.','markersize',15,'color',[.8 0 0]);
	ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
end

%Control profile
for j=1:model.nbVarPos
	subplot(model.nbVar+model.nbVarPos+1,1,model.nbVar+1+j); hold on;
	%Plot uncertainty
	id = j:model.nbVarPos:model.nbVarPos*(nbData-1);
	S = diag(uSigma(id,id)).*1E-3;
	patch([1:nbData-1 nbData-1:-1:1], [r(1).u(j,:)-S(:)' fliplr(r(1).u(j,:)+S(:)')], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
	%Plot reproduction samples
	for n=1:nbRepros
		plot(r(n).u(j,:), '-','linewidth',1,'color',[0 0 0]);
	end
	ylabel(['$u_' num2str(j) '$'],'fontsize',14,'interpreter','latex');
end
xlabel('$t$','fontsize',14,'interpreter','latex');

% print('-dpng','graphs/batchLQR_x_dx03.png');


% %% Plot covariance of control trajectory distribution
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[1630 10 1000 1300],'color',[1 1 1],'name','Covariances'); 
% subplot(4,2,1); hold on; box on; set(gca,'linewidth',2); title('${({S^u}^T Q S^u + R)}^{-1}$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(uSigma));
% set(gca,'xtick',[1,size(uSigma,1)],'ytick',[1,size(uSigma,1)]);
% axis square; axis([1 size(uSigma,1) 1 size(uSigma,1)]); shading flat;
% subplot(4,2,2); hold on; box on; set(gca,'linewidth',2); title('${S^u}^T Q S^u + R$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(SuQSu+R));
% set(gca,'xtick',[1,size(uSigma,1)],'ytick',[1,size(uSigma,1)]);
% axis square; axis([1 size(uSigma,1) 1 size(uSigma,1)]); shading flat;
% 
% subplot(4,2,3); hold on; box on; set(gca,'linewidth',2); title('$\Sigma^x$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(xSigma));
% set(gca,'xtick',[1,size(xSigma,1)],'ytick',[1,size(xSigma,1)]);
% axis square; axis([1 size(xSigma,1) 1 size(xSigma,1)]); shading flat;
% subplot(4,2,4); hold on; box on; set(gca,'linewidth',2); title('$Q^x$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(xQ));
% set(gca,'xtick',[1,size(xSigma,1)],'ytick',[1,size(xSigma,1)]);
% axis square; axis([1 size(xSigma,1) 1 size(xSigma,1)]); shading flat;
% 
% subplot(4,2,5); hold on; box on; set(gca,'linewidth',2); title('${({S^u}^T Q S^u)}^{-1}$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(inv(SuQSu)));
% set(gca,'xtick',[1,size(SuQSu,1)],'ytick',[1,size(SuQSu,1)]);
% axis square; axis([1 size(SuQSu,1) 1 size(SuQSu,1)]); shading flat;
% subplot(4,2,6); hold on; box on; set(gca,'linewidth',2); title('${S^u}^T Q S^u$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(SuQSu));
% set(gca,'xtick',[1,size(SuQSu,1)],'ytick',[1,size(SuQSu,1)]);
% axis square; axis([1 size(SuQSu,1) 1 size(SuQSu,1)]); shading flat;
% 
% subplot(4,2,7); hold on; box on; set(gca,'linewidth',2); title('$Q$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(Q));
% set(gca,'xtick',[1,size(Q,1)],'ytick',[1,size(Q,1)]);
% axis square; axis([1 size(Q,1) 1 size(Q,1)]); shading flat;
% subplot(4,2,8); hold on; box on; set(gca,'linewidth',2); title('$R$','fontsize',14,'interpreter','latex');
% colormap(gca, flipud(gray));
% pcolor(abs(R));
% set(gca,'xtick',[1,size(R,1)],'ytick',[1,size(R,1)]);
% axis square; axis([1 size(R,1) 1 size(R,1)]); shading flat;
% % print('-dpng','graphs/batchLQR_cov01.png');


% %% Plot for slides
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 600 600],'color',[1 1 1]); hold on; axis off;
% %Plot uncertainty
% for t=1:nbData
% 	plotGMM(r(1).x(1:2,t), xSigma(model.nbVar*(t-1)+[1,2],model.nbVar*(t-1)+[1,2]).*1E1, [.2 .2 .2], .1);
% end	
% %Plot reproduction samples
% for n=1:nbRepros
% 	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[0 0 0]);
% 	plot(r(n).x(1,1), r(n).x(2,1), '.','markersize',15,'color',[0 0 0]);
% end
% plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:).*1E1, [.8 0 0], .3);
% plot(model.Mu(1,:), model.Mu(2,:), '.','markersize',15,'color',[.8 0 0]);
% axis equal;
% print('-dpng','graphs/batchLQR_x01.png');
% 
% figure('position',[620 10 600 600],'color',[1 1 1]); 
% for j=1:model.nbVarPos
% 	subplot(model.nbVarPos,1,j); hold on;
% 	if j>model.nbVarPos
% 		plot([1,nbData],[0,0],':','color',[.5 .5 .5]);
% 	end
% 	%Plot uncertainty
% 	id = j:model.nbVar:model.nbVar*nbData;
% 	S = diag(xSigma(id,id)).*1E1;
% 	patch([1:nbData nbData:-1:1], [r(1).x(j,:)-S(:)' fliplr(r(1).x(j,:)+S(:)')], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
% 	%Plot reproduction samples
% 	for n=1:nbRepros
% 		plot(r(n).x(j,:), '-','linewidth',1,'color',[0 0 0]);
% 	end
% 	for t=1:size(xDes,2)
% 		errorbar(xDes(1,t), xDes(1+j,t), xDes(1+model.nbVar+j,t), 'color',[.8 0 0]);
% 		plot(xDes(1,t), xDes(1+j,t), '.','markersize',15,'color',[.8 0 0]);
% 	end
% 	ylabel(['$x_' num2str(j) '$'],'fontsize',22,'interpreter','latex');
% 	xlabel('$t$','fontsize',22,'interpreter','latex');
% 	set(gca,'xtick',[],'ytick',[]);
% 	axis tight;
% end
% print('-dpng','graphs/batchLQR_tx01.png');
% 
% figure('position',[1220 10 600 600],'color',[1 1 1]); 
% for j=1:model.nbVarPos
% 	subplot(model.nbVarPos,1,j); hold on;
% 	%Plot uncertainty
% 	id = j:model.nbVarPos:model.nbVarPos*(nbData-1);
% 	S = diag(uSigma(id,id)).*1E-3;
% 	patch([1:nbData-1 nbData-1:-1:1], [r(1).u(j,:)-S(:)' fliplr(r(1).u(j,:)+S(:)')], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
% 	%Plot reproduction samples
% 	for n=1:nbRepros
% 		plot(r(n).u(j,:), '-','linewidth',1,'color',[0 0 0]);
% 	end
% 	ylabel(['$u_' num2str(j) '$'],'fontsize',22,'interpreter','latex');
% 	xlabel('$t$','fontsize',22,'interpreter','latex');
% 	set(gca,'xtick',[],'ytick',[]);
% 	axis tight;
% end
% print('-dpng','graphs/batchLQR_tu01.png');
%
% figure; hold on; box on;
% colormap(gca, flipud(gray));
% xlim = [1 size(Q,1); 1 size(Q,2)];
% pcolor(abs(Q));
% set(gca,'xtick',[1,size(Q,1)],'ytick',[1,size(Q,2)],'xticklabel',{'1','DCT'},'yticklabel',{'1','DCT'},'fontsize',20);
% plot([xlim(1,1),xlim(1,1:2),xlim(1,2:-1:1),xlim(1,1)], [xlim(2,1:2),xlim(2,2:-1:1),xlim(2,1),xlim(2,1)],'-','color',[0,0,0]);
% axis square; axis([1 size(Q,1) 1 size(Q,2)]); shading flat;
% print('-dpng','graphs/batchLQR_Q01.png');
% 
% figure; hold on; box on;
% colormap(gca, flipud(gray));
% xlim = [1 size(Su,2); 1 size(Su,2)];
% pcolor(abs(SuQSu));
% set(gca,'xtick',[1,size(Su,2)],'ytick',[1,size(Su,2)],'xticklabel',{'1','DT'},'yticklabel',{'1','DT'},'fontsize',20);
% plot([xlim(1,1),xlim(1,1:2),xlim(1,2:-1:1),xlim(1,1)], [xlim(2,1:2),xlim(2,2:-1:1),xlim(2,1),xlim(2,1)],'-','color',[0,0,0]);
% axis square; axis([1 size(Su,2) 1 size(Su,2)]); shading flat;
% print('-dpng','graphs/batchLQR_SuQSu01.png');
% 
% figure; hold on; box on;
% colormap(gca, flipud(gray));
% xlim = [1 size(Su,1); 1 size(Su,2)];
% pcolor(abs(Su'));
% set(gca,'xtick',[1,size(Su,1)],'ytick',[1,size(Su,2)],'xticklabel',{'1','DCT'},'yticklabel',{'1','DT'},'fontsize',20);
% plot([xlim(1,1),xlim(1,1:2),xlim(1,2:-1:1),xlim(1,1)], [xlim(2,1:2),xlim(2,2:-1:1),xlim(2,1),xlim(2,1)],'-','color',[0,0,0]);
% axis equal; axis([1 size(Su,1) 1 size(Su,2)]); shading flat;
% print('-dpng','graphs/batchLQR_Su01.png');

pause;
close all;
