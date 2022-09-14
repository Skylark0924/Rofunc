function demo_OC_LQT_online_minimal01
% MPC recomputed in an online manner with a time horizon.
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
nbD = 50; %Size of the time window for MPC computation
nbPoints = 3; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbD-1)*nbVarPos) * rfactor; %Control cost matrix


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

%Build Su and Sx transfer matrices
Su = sparse(nbVar*nbD, nbVarPos*(nbD-1));
Sx = kron(ones(nbD,1), speye(nbVar));
M = B;
for n=2:nbD
	id1 = (n-1)*nbVar+1:nbD*nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*nbVar+1:n*nbVar; 
	id2 = 1:(n-1)*nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:nbVarPos), M]; 
end


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(1,nbData,nbPoints+1);
tl = round(tl(2:end)); 
MuQ0 = zeros(nbVar*nbData,1); 
Q0 = zeros(nbVar*nbData);
for t=1:length(tl)
	id0(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
	Q0(id0(:,t), id0(:,t)) = eye(nbVarPos);
	MuQ0(id0(:,t)) = rand(nbVarPos,1) - 0.5;
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(nbVar,1);
rx = zeros(nbVar,nbData);
% figure; hold on;
% h = [];
for t=1:nbData
	rx(:,t) = x; %Log data 
	
	if t==20
		tl = linspace(1,nbData-20,nbPoints+1);
		tl = round(tl(2:end)); 
		MuQ0 = zeros(nbVar*nbData,1); 
		Q0 = zeros(nbVar*nbData);
		for t=1:length(tl)
			id0(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
			Q0(id0(:,t), id0(:,t)) = eye(nbVarPos);
			MuQ0(id0(:,t)) = rand(nbVarPos,1) - 0.5;
		end
	end
	
	id = [t:min(t+nbD-1,nbData), ones(1,t-nbData+nbD-1)]; %Time steps involved in the MPC computation
	id2 = [];
	for s=1:nbD
		id2 = [id2; [1:nbVar]' + (id(s)-1) * nbVar];
	end
	MuQ = MuQ0(id2,1);
	Q = Q0(id2,id2);

	u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x); %Compute control commands
	x = A * x + B * u(1:nbVarPos); %Update state with first control command
	
% 	delete(h);
% 	h(1) = plot(MuQ(1:nbVar:end),'k.');
% 	h(2) = plot(x(1),'r.');
% 	axis([1,nbD,min(MuQ0),max(MuQ0)]);
% 	pause(.001);
end

%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1000 1000]); hold on; axis off;
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), '.','markersize',35,'color',[0 0 0]);
plot(MuQ0(id0(1,:)), MuQ0(id0(2,:)), '.','markersize',35,'color',[.8 0 0]);
for t=1:length(tl)
	text(MuQ0(id0(1,t)), MuQ0(id0(2,t))+2E-2, num2str(t));
end
axis equal; 
% print('-dpng','graphs/MPC_minimal01.png');

pause;
close all;