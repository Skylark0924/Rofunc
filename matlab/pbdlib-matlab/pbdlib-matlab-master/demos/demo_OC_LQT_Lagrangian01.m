function demo_OC_LQT_Lagrangian01
% Batch LQT with Lagrangian in matrix form to force first and last point to coincide in order to form a periodic motion.
% (see also demo_MPC_noInitialState01.m)
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Berio19MM,
% 	author="Berio, D. and Fol Leymarie, F. and Calinon, S.",
% 	title="Interactive Generation of Calligraphic Trajectories from {G}aussian Mixtures",
% 	booktitle="Mixture Models and Applications",
% 	publisher="Springer",
% 	editor="Bouguila, N. and Fan, W.", 
% 	year="2019",
% 	pages="23--38",
% 	doi="10.1007/978-3-030-23876-6_2"
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
nbData = 200; %Number of datapoints
nbPoints = 4; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-10; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarPos) * rfactor; %Control cost matrix

nbRepros = 20;


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A1d = zeros(nbDeriv);
% for i=0:nbDeriv-1
% 	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
% end
% B1d = zeros(nbDeriv,1); 
% for i=1:nbDeriv
% 	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
% end

A = zeros(nbDeriv);
A(1:nbDeriv-1, 2:nbDeriv) = eye(nbDeriv-1);
B = zeros(nbDeriv, 1);
B(nbDeriv,:) = 1.;
A1d = eye(nbDeriv) + A * dt;
B1d = B * dt;
        
A = kron(A1d, speye(nbVarPos)); %Discrete nD
B = kron(B1d, speye(nbVarPos)); %Discrete nD

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
Mu0 = [[-1;1], [1;1], [1;-1], [-1;-1]];
tl = linspace(1,nbData,nbPoints+2);
tl = round(tl(2:end-1)); %[nbData/2, nbData];

% Mu0 = [[-1;1], [1;1], [1;-1], [-1;-1], [-1;1]];
% tl = linspace(1,nbData,nbPoints); %+2);
% tl = round(tl); %%(2:end-1)); %[nbData/2, nbData];

MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);
for t=1:length(tl)
	id(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
	Q(id(:,t), id(:,t)) = eye(nbVarPos);
	MuQ(id(:,t)) = Mu0(:,t); %rand(nbVarPos,1);
end


% %% Standard batch LQT (for comparison)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x0 = MuQ(1:nbVar); 
% u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
% rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting


%% Batch LQT with Lagrangian in matrix form
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = zeros(nbVar,nbData*nbVar); %Linear constraint C*x=0
C(:,1:nbVar) = -eye(nbVar) * 1E3; %p1-p2=0 (for both position and speed)
C(:,end-nbVar+1:end) = eye(nbVar) * 1E3; %p1-p2=0 (for both position and speed)

%Augmented solution by finding the optimal initial state together with the optimal control commands
S = [Su, Sx];
Ra = blkdiag(R, zeros(nbVar));
Ct = C * S;
Rqc = [S' * Q * S + Ra, Ct'; Ct, zeros(nbVar)];
u = Rqc \ [S' * Q * MuQ; zeros(nbVar,1)];
rx = S * u(1:end-nbVar,:); %Reshape data for plotting (u(1:end-nbVar,:) removes the Lagrange multipliers part)

uSigma = inv(Rqc); % + eye((nbData-1)*nbVarU) * 1E-10;
xSigma = [S, C'] * uSigma * [S, C']';


%% Stochastic sampling by exploiting distribution on x
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nbEigs = 3; %Number of principal eigencomponents to keep
[V,D] = eigs(xSigma);
for n=1:nbRepros
	%xtmp = Mu + V(:,1:nbEigs) * D(1:nbEigs,1:nbEigs).^.5 * randn(nbEigs,1);
	r(n).x = real(rx + V * D.^.5 * randn(size(D,1),1) * 2E-4);
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
for t=1:nbData
	plotGMM(rx(nbVar*(t-1)+[1,2]), xSigma(nbVar*(t-1)+[1,2],nbVar*(t-1)+[1,2]).*1E-7, [.2 .2 .2], .1);
end	
plot(rx(1:nbVar:end), rx(2:nbVar:end), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1), rx(2), '.','markersize',40,'color',[0 0 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',40,'color',[.8 0 0]);
for n=1:nbRepros
	plot(r(n).x(1:nbVar:end), r(n).x(2:nbVar:end), '-','linewidth',1,'color',[0 .6 0]);
end
axis equal; 
% print('-dpng','graphs/demo_MPC_Lagrangian01.png');

pause(10);
close all;
