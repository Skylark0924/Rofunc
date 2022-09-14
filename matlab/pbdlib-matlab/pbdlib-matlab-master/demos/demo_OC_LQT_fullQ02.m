function demo_OC_LQT_fullQ02
% Batch LQT exploiting full Q matrix to constrain the motion of two agents.
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
nbData = 200; %Number of datapoints
% nbData = 20; %Number of datapoints
nbRepros = 3; %Number of stochastic reproductions
nbAgents = 2; %Number of agents
nbVarPos = 2 * nbAgents; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8; %dt^nbDeriv; %Control cost in LQR
R = speye((nbData-1)*nbVarPos) * rfactor; %Control cost matrix
Rx = speye(nbData*nbVar) * rfactor;


%% Dynamical System settings for several agents (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(kron(A1d, speye(nbVarPos/nbAgents)), speye(nbAgents)); %Discrete nD for several agents
B = kron(kron(B1d, speye(nbVarPos/nbAgents)), speye(nbAgents)); %Discrete nD for several agents

%Build Sx and Su transfer matrices
Su = sparse(nbVar*nbData, nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1), speye(nbVar));
M = B;
for n=2:nbData
	id1 = (n-1)*nbVar+1:nbData*nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*nbVar+1:n*nbVar; 
	id2 = 1:(n-1)*nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:nbVarPos), M]; 
end


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = nbData;
MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);
for t=1:length(tl)
	id(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
	Q(id(:,t), id(:,t)) = eye(nbVarPos);
	MuQ(id(:,t)) = rand(nbVarPos,1) - 0.5;
end

% %Constraining the two agents to meet in the middle of the motion
% tlm = nbData/2;
% idm = [1:nbVarPos] + (tlm-1) * nbVar;
% Q(idm,idm) = [eye(2), -eye(2); -eye(2), eye(2)];

%Constraining the two agents to be at a given position with an offset at time t1 and t2
t1 = nbData/2;
t2 = t1+50;
idm(1:2) = [1:2] + (t1-1) * nbVar;
idm(3:4) = [3:4] + (t2-1) * nbVar;
Q(idm,idm) = [eye(2), -eye(2); -eye(2), eye(2)];
MuQ(idm(1:2)) = [0; 0]; 
MuQ(idm(3:4)) = MuQ(idm(1:2)) + [.05; 0]; %Meeting point with desired offset between the two agents 

% %Constraining the two agents to meet at a given angle at the end of the motion
% %Proposed meeting point for the two agents
% tlm = nbData;
% idm = [1:nbVar] + (tlm-1) * nbVar;
% MuQ(idm(3:4)) = MuQ(idm(1:2)); 
% Q(idm(1:2), idm(3:4)) = -eye(2)*1E0;
% Q(idm(3:4), idm(1:2)) = -eye(2)*1E0;
% %Velocity correlations between the two agents
% a = pi/2; %desired angle
% V = [cos(a) -sin(a); sin(a) cos(a)]; %rotation matrix
% Q(idm(5:6), idm(5:6)) = eye(2)*1E0;
% Q(idm(7:8), idm(7:8)) = eye(2)*1E0;
% Q(idm(5:6), idm(7:8)) = V;
% Q(idm(7:8), idm(5:6)) = V';

% %Constraining the two agents to meet at a given angle in the middle of the motion
% tlm = nbData/2;
% idm = [1:nbVar] + (tlm-1) * nbVar;
% % MuQ(idm(1:2)) = [1; 1]; %Proposed meeting point for the two agents
% % MuQ(idm(3:4)) = MuQ(idm(1:2)); 
% Q(idm(1:2), idm(1:2)) = eye(2)*1E0;
% Q(idm(3:4), idm(3:4)) = eye(2)*1E0;
% Q(idm(1:2), idm(3:4)) = -eye(2)*1E0;
% Q(idm(3:4), idm(1:2)) = -eye(2)*1E0;
% %Velocity correlations between the two agents
% a = pi/2; %desired angle
% V = [cos(a) -sin(a); sin(a) cos(a)]; %rotation matrix
% Q(idm(5:6), idm(5:6)) = eye(2)*1E0;
% Q(idm(7:8), idm(7:8)) = eye(2)*1E0;
% Q(idm(5:6), idm(7:8)) = V;
% Q(idm(7:8), idm(5:6)) = V';


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [rand(nbVarPos,1)-0.5; zeros(nbVar-nbVarPos,1)];
u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting

% mse =  abs(MuQ'*Q*MuQ - rq'/Rq*rq) ./ (model.nbVarPos*nbData);
uSigma = inv(Su' * Q * Su + R); % * mse; % + eye((nbData-1)*model.nbVarU) * 1E-10;
xSigma = Su * uSigma * Su';


%% Stochastic sampling by exploiting the nullspace structure of the problem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate structured stochastic u through Bezier curves
nbRBF = 18;
H = zeros(nbRBF, nbData-1);
tl = linspace(0, 1, nbData-1);
nbDeg = nbRBF - 1;
for i=0:nbDeg
	H(i+1,:) = factorial(nbDeg) / (factorial(i) * factorial(nbDeg-i)) * (1-tl).^(nbDeg-i) .* tl.^i; %Bernstein basis functions
end

%Nullspace planning
[V,D] = eig(Q);
U = V * D.^.5;
N = eye((nbData-1)*nbVarPos) - Su' * U / (U' * (Su * Su') * U + Rx) * U' * Su; %Nullspace projection matrix
up = Su' * U / (U' * (Su * Su') * U + Rx) * U' * (MuQ - Sx * x0); %Principal task
% up = (Su' * Q * Su + R) \ (Su' * Q * (MuQ - Sx * x0));

for n=1:nbRepros
	w = randn(nbVarPos,nbRBF) .* 1E0; %Random weights
	un = w * H; %Reconstruction of signals by weighted superposition of basis functions
	uTmp = up + N * un(:); %2E1 * randn((nbData-1)*nbVarPos,1); 
	r(n).x = reshape(Sx*x0+Su*uTmp, nbVar, nbData); %Reshape data for plotting
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 2400 600]); 
subplot(1,4,1); hold on; axis off;
% set(0,'DefaultAxesLooseInset',[1,1,1,1]*1E2);
% set(gca,'LooseInset',[1,1,1,1]*1E2);
%Agent 1
% for t=1:nbData
% 	plotGMM(rx(1:2,t), xSigma(nbVar*(t-1)+[1,2],nbVar*(t-1)+[1,2]).*1E-8, [.2 .2 .2], .03); %Plot uncertainty
% end	
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), 'o','linewidth',2,'markersize',8,'color',[0 0 0]);
% plot(rx(1,tlm), rx(2,tlm), '.','markersize',35,'color',[0 .6 0]); %meeting point
plot(rx(1,t1), rx(2,t1), '.','markersize',35,'color',[0 .6 0]); %meeting point
plot(rx(1,t1), rx(2,t1), '.','markersize',35,'color',[0 .6 0]); %meeting point
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',35,'color',[0 0 0]);
%Agent 2
% for t=1:nbData
% 	plotGMM(rx(3:4,t), xSigma(nbVar*(t-1)+[3,4],nbVar*(t-1)+[3,4]).*1E-8, [.2 .2 .2], .03); %Plot uncertainty
% end	
plot(rx(3,:), rx(4,:), '-','linewidth',2,'color',[.8 0 0]);
plot(rx(3,1), rx(4,1), 'o','linewidth',2,'markersize',8,'color',[.8 0 0]);
% plot(rx(3,tlm), rx(4,tlm), '.','markersize',35,'color',[0 .6 0]); %meeting point
plot(rx(3,t2), rx(4,t2), '.','markersize',35,'color',[0 .6 0]); %meeting point
plot(MuQ(id(3,:)), MuQ(id(4,:)), '.','markersize',35,'color',[.8 0 0]);
axis equal; axis([-.5,.5,-.5,.5]);
% print('-dpng','graphs/MPC_fullQ04.png');

for n=1:nbRepros
	subplot(1,4,1+n); hold on; axis off;
	%Agent 1 (old)
	plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[.8 .8 .8]);
% 	plot(rx(1,tlm), rx(2,tlm), '.','markersize',35,'color',[.6 1 .6]); %meeting point
	%Agent 2 (old)
	plot(rx(3,:), rx(4,:), '-','linewidth',2,'color',[.8 .8 .8]);
% 	plot(rx(3,tlm), rx(4,tlm), '.','markersize',35,'color',[.8 1 .8]); %meeting point
	%Agent 1
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',2,'color',[0 0 0]);
	plot(r(n).x(1,1), r(n).x(2,1), 'o','linewidth',2,'markersize',8,'color',[0 0 0]);
% 	plot(r(n).x(1,tlm), r(n).x(2,tlm), '.','markersize',35,'color',[0 .6 0]); %meeting point
	plot(r(n).x(1,t1), r(n).x(2,t1), '.','markersize',35,'color',[0 .6 0]); %meeting point
	plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',35,'color',[0 0 0]);
	%Agent 2
	plot(r(n).x(3,:), r(n).x(4,:), '-','linewidth',2,'color',[.8 0 0]);
	plot(r(n).x(3,1), r(n).x(4,1), 'o','linewidth',2,'markersize',8,'color',[.8 0 0]);
% 	plot(r(n).x(3,tlm), r(n).x(4,tlm), '.','markersize',35,'color',[0 .6 0]); %meeting point
	plot(r(n).x(3,t2), r(n).x(4,t2), '.','markersize',35,'color',[0 .6 0]); %meeting point
	plot(MuQ(id(3,:)), MuQ(id(4,:)), '.','markersize',35,'color',[.8 0 0]);
	axis equal; axis([-.5,.5,-.5,.5]);
end
% print('-dpng','graphs/LQT_fullQ02a.png');


% %Visualize Q
% xlim = [1 size(Q,1); 1 size(Q,1)];
% figure('position',[1030 10 1000 1000],'color',[1 1 1]); hold on; axis off;
% set(0,'DefaultAxesLooseInset',[0,0,0,0]);
% set(gca,'LooseInset',[0,0,0,0]);
% colormap(gca, flipud(gray));
% imagesc(abs(Q));
% plot([xlim(1,1),xlim(1,1:2),xlim(1,2:-1:1),xlim(1,1)], [xlim(2,1:2),xlim(2,2:-1:1),xlim(2,1),xlim(2,1)],'-','linewidth',2,'color',[0,0,0]);
% axis equal; axis ij;

waitfor(h);
