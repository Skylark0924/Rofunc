function demo_trajGMM01
% Trajectory synthesis using a GMM with dynamic features (trajectory GMM)
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
nbStates = 4; %Number of components in the GMM
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static&dynamic features (D=2 for [x,dx], D=3 for [x,dx,ddx], etc.)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-1; %Time step (without rescaling, large values such as 1 has the advantage of creating clusers based on position information)
nbData = 200; %Number of datapoints in a trajectory

%PHI matrix (without border condition)
op1D = zeros(nbDeriv);
op1D(1,end) = 1;
for i=2:nbDeriv
	op1D(i,:) = (op1D(i-1,:) - circshift(op1D(i-1,:),[0,-1])) / dt;
end
op = zeros(nbData*nbDeriv, nbData);
op(1:nbDeriv, 1:nbDeriv) = op1D;
PHI0 = zeros(nbData*nbDeriv, nbData);
for t=0:nbData-nbDeriv
	PHI0 = PHI0 + circshift(op, [nbDeriv*t,t]);
end
%Application to multiple dimensions and multiple demonstrations
PHI1 = kron(PHI0, eye(nbVarPos));


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = [rand(nbVarPos,nbStates); zeros(nbVar-nbVarPos,nbStates)];
Gamma = repmat(eye(nbVar), [1,1,nbStates]); %Precision matrix
Gamma(:,:,2:3) = Gamma(:,:,2:3) * 1E-2; %Lower precision for viapoints
rq = kron(1:nbStates, ones(1,nbData/nbStates)); %State sequence rq

%Create single Gaussian N(MuQ,GammaQ^-1) based on state sequence rq
MuQ = reshape(Mu(:,rq), nbVar*nbData, 1); 
Gtmp = Gamma(:,:,rq);
Q = kron(speye(nbData), ones(nbVar));
Q(logical(Q)) = Gtmp(:);


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xhat = (PHI1' * Q * PHI1) \ PHI1' * Q * MuQ; 
rx = reshape(xhat, nbVarPos, nbData); %Reshape data for plotting


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[620 10 600 600]); hold on; axis off;
plot(rx(1,:), rx(2,:), '-','lineWidth',2,'color',[0 0 0]);
plot(Mu(1,:), Mu(2,:), '.','markersize',25,'color',[.8 0 0]);
axis equal; 

%print('-dpng','graphs/demo_trajGMM_illustr01.png');
pause;
close all;