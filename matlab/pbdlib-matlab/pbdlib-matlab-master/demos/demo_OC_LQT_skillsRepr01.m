function demo_OC_LQT_skillsRepr01
% Representation of skills combined in parallel and in series through a batch LQT formulation.
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
nbData = 120; %Number of datapoints in a trajectory
nbRepros = 5; %Number of reproductions in new situations

nbFrames = 3; %Number of candidate frames of reference
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-3;	%Control cost in LQR


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Integration with higher order Taylor series expansion
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(nbVarPos)); %Discrete nD
B = kron(B1d, eye(nbVarPos)); %Discrete nD

%Construct Su and Sx matrices (transfer matrices in batch LQR)
Su = zeros(nbVar*nbData, nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(nbVar)); 
M = B;
for n=2:nbData
	%Build Sx matrix
	id1 = (n-1)*nbVar+1:nbData*nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	%Build Su matrix
	id1 = (n-1)*nbVar+1:n*nbVar; 
	id2 = 1:(n-1)*nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:nbVarPos), M];
end

%Control cost matrix in LQR
R = eye(nbVarPos) * rfactor;
R = kron(eye(nbData-1), R);


%% Definition of Task A1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mu = [[-.7;0;0;0], [.7;.2;0;0], [1.5;-.8;0;0]];
% v = [1; 1];
% Sigma(:,:,1) = blkdiag(v*v' + eye(nbVarPos).*3E-3, eye(nbVarPos).*1E0);
% v = [1; -.5];
% Sigma(:,:,2) = blkdiag(v*v' + eye(nbVarPos).*3E-3, eye(nbVarPos).*1E0);
% Sigma(:,:,3) = blkdiag(eye(nbVarPos) .* 3E-3, eye(nbVarPos).*1E0);
% %Build a reference trajectory for each frame
% p(1).Q = blkdiag(zeros((nbData/2-1)*nbVar), inv(Sigma(:,:,1)), zeros((nbData/2)*nbVar));  
% p(2).Q = blkdiag(zeros((nbData/2-1)*nbVar), inv(Sigma(:,:,2)), zeros((nbData/2)*nbVar));  
% p(3).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,3)));  


%% Definition of Task A2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = [[-.7;0;0;0], [.7;.2;0;0], [1.7;.6;0;0]];
v = [1; 1];
Sigma(:,:,1) = blkdiag(v*v' + eye(nbVarPos).*3E-3, eye(nbVarPos).*1E0);
v = [1; -.5];
Sigma(:,:,2) = blkdiag(v*v' + eye(nbVarPos).*3E-3, eye(nbVarPos).*1E0);
% v = [-.4; -1];
% Sigma(:,:,3) = blkdiag(v*v' + eye(nbVarPos).*3E-3, eye(nbVarPos).*1E0);
Sigma(:,:,3) = blkdiag(eye(nbVarPos) .* 3E-3, eye(nbVarPos).*1E0);
%Build a reference trajectory for each frame
% p(1).Q = blkdiag(kron(eye(nbData/3),inv(Sigma(:,:,1))), zeros((nbData/3)*nbVar), zeros((nbData/3)*nbVar));  
% p(2).Q = blkdiag(zeros((nbData/3)*nbVar),  kron(eye(nbData/3),inv(Sigma(:,:,2))), zeros((nbData/3)*nbVar));  
% p(3).Q = blkdiag(zeros((nbData/3)*nbVar), zeros((nbData/3)*nbVar), kron(eye(nbData/3),inv(Sigma(:,:,3))));  

% p(1).Q = blkdiag(zeros((nbData/2-1)*nbVar), inv(Sigma(:,:,1)), zeros((nbData/2)*nbVar));  
% p(2).Q = blkdiag(zeros((nbData/2-1)*nbVar), inv(Sigma(:,:,2)), zeros((nbData/2)*nbVar));  
% p(3).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,3)));  

p(1).Q = blkdiag(zeros((nbData/3-1)*nbVar), inv(Sigma(:,:,1)), zeros((2*nbData/3)*nbVar));  
p(2).Q = blkdiag(zeros((2*nbData/3-1)*nbVar), inv(Sigma(:,:,2)), zeros((nbData/3)*nbVar));  
p(3).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,3)));  


% %% Definition of Task B
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mu = [[-1.6;-.4;0;0], [.7;.35;0;0], [1.5;-.8;0;0]];
% Sigma = repmat(blkdiag(eye(nbVarPos) .* 1E-2, eye(nbVarPos).*1E2), [1,1,nbFrames]);
% Sigma(1:nbVarPos,1:nbVarPos,2) = eye(nbVarPos) .* 5E-1;
% %Build a reference trajectory for each frame
% p(1).Q = blkdiag(zeros((nbData/3-1)*nbVar), inv(Sigma(:,:,1)), zeros((2*nbData/3)*nbVar));  
% p(2).Q = blkdiag(zeros((2*nbData/3-1)*nbVar), inv(Sigma(:,:,2)), zeros((nbData/3)*nbVar));  
% p(3).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,3)));  


% %% Definition of Task C
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mu = [[-1.2;-.3;0;0], [1.0;.3;0;0], [0;-.8;0;0]];
% Sigma = repmat(blkdiag(eye(nbVarPos) .* 1E-1, eye(nbVarPos).*1E4), [1,1,nbFrames]);
% Sigma(1:nbVarPos,1:nbVarPos,3) = diag([1E18,1E-3]);
% %Build a reference trajectory for each frame
% p(1).Q = blkdiag(zeros((nbData/2-1)*nbVar), inv(Sigma(:,:,1)), zeros((nbData/2)*nbVar));  
% p(2).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,2)));  
% p(3).Q = kron(eye(nbData), inv(Sigma(:,:,3)));  


%% Batch LQT computed as a product of Gaussian controllers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = zeros(nbVar*nbData);
for m=1:nbFrames
	p(m).MuQ = kron(ones(nbData,1), Mu(:,m));  
% 	p(m).Q = kron(eye(nbData), inv(Sigma(:,:,m)));  
% 	p(m).Q = blkdiag(zeros((nbData-1)*nbVar), inv(Sigma(:,:,m)));  
	Q = Q + p(m).Q;
end
for n=1:nbRepros
	Rq = Su' * Q * Su + R;
	x0 = [[-.4;-.6] + randn(nbVarPos,1)*2E-1; zeros(nbVarPos,1)];
 	rq = zeros(nbVar*nbData,1);
	for m=1:nbFrames
		rq = rq + p(m).Q * (p(m).MuQ - Sx*x0);
	end
	rq = Su' * rq; 
 	u = Rq \ rq; 
	r(n).Data = reshape(Sx*x0+Su*u, nbVar, nbData);
end


%% Plots Task A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(nbFrames);
colTmp = .5 - [.5; .5; .5] * rand(1,nbRepros);
figure('PaperPosition',[0 0 10 6],'position',[10,10,1000,600]); hold on; axis off;
for m=1:nbFrames
	plotGMM(Mu(1:2,m), Sigma(1:2,1:2,m), clrmap(m,:),.4);
end
for n=1:nbRepros
	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',2,'color',colTmp(:,n));
	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',15,'color',colTmp(:,n));
end
axis equal; axis tight; %axis([-1.8,1.8,-1.1,1.1]); 
% print('-dpng','graphs/OCrepr_taskA1_03.png');

%Timeline plot
figure; 
for i=1:nbVarPos
	subplot(2,1,i); hold on;
	for m=1:2
		errorbar(nbData/2, Mu(i,m), Sigma(i,i,m)^.5, 'linewidth',2,'color',clrmap(m,:));
		plot(nbData/2, Mu(i,m), '.','markersize',20,'color',clrmap(m,:));
	end
	errorbar(nbData, Mu(i,3), Sigma(i,i,3)^.5, 'linewidth',2,'color',clrmap(3,:));
	
	%patch([1 nbData nbData 1 1], [-Sigma(i,i,3)^.5 -Sigma(i,i,3)^.5 Sigma(i,i,3)^.5 Sigma(i,i,3)^.5 -Sigma(i,i,3)^.5]+Mu(i,3), clrmap(3,:),'linewidth',2,'edgecolor',clrmap(3,:),'facealpha',.4);
	
	plot(nbData, Mu(i,3), '.','markersize',20,'color',clrmap(3,:));
	for n=1:nbRepros
		plot(r(n).Data(i,:),'-','linewidth',1,'color',colTmp(:,n));
	end
	axis([1,nbData,-1.8,1.8]);
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',20);
	ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',20);
end
% print('-dpng','graphs/OCrepr_taskA02.png');


% %% Plots Task B
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clrmap = lines(nbFrames);
% figure('PaperPosition',[0 0 10 6],'position',[10,10,1000,600]); hold on; axis off;
% for m=1:nbFrames
% 	plotGMM(Mu(1:2,m), Sigma(1:2,1:2,m), clrmap(m,:),.4);
% end
% for n=1:nbRepros
% 	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',2,'color',[0 0 0]);
% 	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',15,'color',[0 0 0]);
% end
% axis equal; axis([-1.8,1.8,-1.1,1.1]); 
% % print('-dpng','graphs/OCrepr_taskB01.png');
% 
% %Timeline plot
% figure; 
% for i=1:nbVarPos
% 	subplot(2,1,i); hold on;
% 	for m=1:nbFrames
% 		errorbar(m*nbData/3, Mu(i,m), Sigma(i,i,m)^.5, 'linewidth',2,'color',clrmap(m,:));
% 		plot(m*nbData/3, Mu(i,m), '.','markersize',20,'color',clrmap(m,:));
% 	end
% 	for n=1:nbRepros
% 		plot(r(n).Data(i,:),'-','linewidth',1,'color',[0 0 0]);
% 	end
% 	axis([1,nbData,-1.8,1.8]);
% 	set(gca,'xtick',[],'ytick',[]);
% 	xlabel('$t$','interpreter','latex','fontsize',20);
% 	ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',20);
% end
% % print('-dpng','graphs/OCrepr_taskB02.png');


% %% Plots Task C
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clrmap = lines(nbFrames);
% figure('PaperPosition',[0 0 10 6],'position',[10,10,1000,600]); hold on; axis off;
% for m=1:nbFrames
% 	plotGMM(Mu(1:2,m), Sigma(1:2,1:2,m), clrmap(m,:),.4);
% end
% for n=1:nbRepros
% 	plot(r(n).Data(1,:), r(n).Data(2,:),'-','linewidth',2,'color',[0 0 0]);
% 	plot(r(n).Data(1,1), r(n).Data(2,1),'.','markersize',15,'color',[0 0 0]);
% end
% axis equal; axis([-1.8,1.8,-1.1,1.1]); 
% % print('-dpng','graphs/OCrepr_taskC01.png');
% 
% %Timeline plot
% figure; 
% for i=1:nbVarPos
% 	subplot(2,1,i); hold on;
% 	for m=1:2
% 		errorbar(m*nbData/2, Mu(i,m), Sigma(i,i,m)^.5, 'linewidth',2,'color',clrmap(m,:));
% 		plot(m*nbData/2, Mu(i,m), '.','markersize',20,'color',clrmap(m,:));
% 	end
% 	patch([1 nbData nbData 1 1], [-Sigma(i,i,3)^.5 -Sigma(i,i,3)^.5 Sigma(i,i,3)^.5 Sigma(i,i,3)^.5 -Sigma(i,i,3)^.5]+Mu(i,3), clrmap(3,:),'linewidth',2,'edgecolor',clrmap(3,:),'facealpha',.4);
% 	plot([1 nbData], [Mu(i,3) Mu(i,3)], '-','linewidth',4,'color',clrmap(3,:));
% 	for n=1:nbRepros
% 		plot(r(n).Data(i,:),'-','linewidth',1,'color',[0 0 0]);
% 	end
% 	axis([1,nbData,-1.8,1.8]);
% 	set(gca,'xtick',[],'ytick',[]);
% 	xlabel('$t$','interpreter','latex','fontsize',20);
% 	ylabel(['$x_' num2str(i) '$'],'interpreter','latex','fontsize',20);
% end
% % print('-dpng','graphs/OCrepr_taskC02.png');

pause;
close all;