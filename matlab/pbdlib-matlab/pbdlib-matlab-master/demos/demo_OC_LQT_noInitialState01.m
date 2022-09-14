function demo_OC_LQT_noInitialState01
% Batch LQT solution determining the optimal initial state together with the optimal control commands.
% (see also demo_MPC_Lagrangian01.m)
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
nbPoints = 4; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8; %1E10;  dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarPos) * rfactor; %Control cost matrix


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
end
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
tl = linspace(1,nbData,nbPoints+1);
tl = round(tl(2:end)); %[nbData/2, nbData];
MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);
for t=1:length(tl)
	id(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
	Q(id(:,t), id(:,t)) = eye(nbVarPos);
	MuQ(id(:,t)) = rand(nbVarPos,1) - 0.5;
end


%% Batch LQT reproduction without specifying initial point
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = [Sx, Su];
Ra = blkdiag(zeros(nbVar), R);
ua = (S' * Q * S + Ra) \ S' * Q * MuQ; 
rx = reshape(S*ua, nbVar, nbData); %Reshape data for plotting

x0 = ua(1:nbVar,1); %Optimal starting point 
% u = ua(nbVar+1:end,1); %Optimal control commands

% %Standard LQT solution
% x0 = zeros(nbVar,1);
% u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
% rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1000 1000],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), '.','markersize',40,'color',[0 0 0]);
plot(x0(1,1), x0(2,1), 'o','markersize',20,'color',[0 0 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',40,'color',[.8 0 0]);
axis equal; 
% print('-dpng','graphs/demo_MPC_noInitialState01.png');

pause;
close all;