function demo_OC_LQT_nullspace02
% Batch LQT with nullspace formulation: ballistic task with an augmented state space
%
% If this code is useful for your research, please cite the related publication:
% @article{Girgin19,
% 	author="Girgin, H. and Calinon, S.",
% 	title="Nullspace Structure in Model Predictive Control",
% 	journal="arXiv:1905.09679",
% 	year="2019",
% 	pages="1--16"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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
t_release = 50; %Time step when object is released
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * (nbDeriv+1); %Dimension of augmented state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarPos) * rfactor;
Rx = speye(nbData*nbVar) * rfactor;


%% Dynamical System settings (augmented state space, discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ac0 = kron([0, 1, 0; 0, 0, 0; 0, 0, 0], speye(nbVarPos));
Ad0 = speye(nbVar) + Ac0 * dt;

% Ac = kron([0, 1, 0; 0, 0, 0; 0, 0, 0], speye(nbVarPos));
% Ac(4,end) = -9.81 * 1E0; %gravity
% Ad = speye(nbVar) + Ac * dt;

Ac = kron([0, 1, 0; 0, 0, 1; 0, 0, 0], speye(nbVarPos));
Ad = speye(nbVar) + Ac * dt;

% Bd = kron([dt^2/2; dt; 0], speye(nbVarPos));
Bc = kron([0; 1; 0], speye(nbVarPos));
Bd = Bc * dt;

% %Build Sx and Su transfer matrices(for homogeneous A and B)
% Su = sparse(nbVar*nbData, nbVarPos*(nbData-1));
% Sx = kron(ones(nbData,1), speye(nbVar));
% M = Bd;
% for n=2:nbData
% 	id1 = (n-1)*nbVar+1:nbData*nbVar;
% 	Sx(id1,:) = Sx(id1,:) * Ad;
% 	id1 = (n-1)*nbVar+1:n*nbVar; 
% 	id2 = 1:(n-1)*nbVarPos;
% 	Su(id1,id2) = M;
% 	M = [Ad*M(:,1:nbVarPos), M];
% end

%Build Sx and Su transfer matrices(for heterogeneous A and B)
A = zeros(nbVar,nbVar,nbData-1);
B = zeros(nbVar,nbVarPos,nbData-1);
for t=1:nbData-1
	A(:,:,t) = Ad;
	if t<t_release
% 		A(:,:,t) = Ad0;
		B(:,:,t) = Bd;
	end
end
[Su, Sx] = transferMatrices(A, B);


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);
id = [1:nbVarPos] + (nbData-1) * nbVar;
Q(id,id) = eye(nbVarPos);
MuQ(id) = rand(nbVarPos,1) - 0.5;


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate structured stochastic u through Bezier curves
nbRBF = 18;
H = zeros(nbRBF,nbData-1);
tl = linspace(0,1,nbData-1);
nbDeg = nbRBF - 1;
for i=0:nbDeg
	H(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-tl).^(nbDeg-i) .* tl.^i; %Bernstein basis functions
end
w = randn(nbVarPos,nbRBF) .* 1E1; %Random weights
un = w * H; %Reconstruction of signals by weighted superposition of basis functions
un(:,t_release+1:end) = 0;

%Nullspace planning
% x0 = [zeros(nbVarPos*nbDeriv,1); ones(nbVarPos,1)];
x0 = [zeros(nbVarPos*nbDeriv,1); 0; -9.81*1E-1];
u0 = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
rx0 = reshape(Sx*x0+Su*u0, nbVar, nbData); %Reshape data for plotting

[V,D] = eig(Q);
U = V * D.^.5;
N = eye((nbData-1)*nbVarPos) - Su' * U / (U' * (Su * Su') * U + Rx) * U' * Su; %Nullspace projection matrix
u = Su' * U / (U' * (Su * Su') * U + Rx) * U' * (MuQ - Sx * x0) + N * un(:);  
rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
plot(rx0(1,:), rx0(2,:), '-','linewidth',2,'color',[.7 .7 .7]);
plot(rx0(1,1), rx0(2,1), '.','markersize',30,'color',[.7 .7 .7]);
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
h(1) = plot(rx(1,1), rx(2,1), '.','markersize',30,'color',[0 0 0]);
h(2) = plot(rx(1,t_release), rx(2,t_release), '.','markersize',30,'color',[0 .6 0]);
plot(rx0(1,t_release), rx0(2,t_release), '.','markersize',30,'color',[.7 1 .7]);
h(3) = plot(MuQ(id(1)), MuQ(id(2)), '.','markersize',30,'color',[.8 0 0]);
legend(h, {'Initial point','Release of object','Target to reach'});
axis equal; 
% print('-dpng','graphs/demo_MPC_nullspace02.png');

%Timeline Plots
figure('position',[830,10,800,800]); 
for j=1:nbVarPos
	subplot(nbVarPos,1,j); hold on; %axis off;
	plot(1:nbData-1, u0(j:nbVarPos:end), '-','linewidth',1,'color',[.7 .7 .7]);
	plot(1:nbData-1, u(j:nbVarPos:end), '-','linewidth',1,'color',[0 0 0]);
% 	set(gca,'xtick',[],'ytick',[]);
	plot([t_release,t_release], [min(u(j:nbVarPos:end)), max(u(j:nbVarPos:end))], '-','linewidth',2,'color',[0 .6 0]);
	xlabel('$t$','interpreter','latex','fontsize',28); 
	ylabel(['$u_' num2str(j) '$'],'interpreter','latex','fontsize',28);
end
axis tight;

% %Visualize Su matrix
% figure('position',[10 1050 400 650],'name','Su'); 
% axes('Position',[0.01 0.01 .98 .98]); hold on; set(gca,'linewidth',2); 
% colormap(flipud(gray));
% pcolor([abs(Su) zeros(size(Su,1),1); zeros(1,size(Su,2)+1)]); %dummy values for correct display
% shading flat; axis ij; axis equal tight;
% set(gca,'xtick',[],'ytick',[]);

pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
function [Su, Sx] = transferMatrices(A, B)
	[nbVarX, nbVarU, nbData] = size(B);
	nbData = nbData+1;
	Sx = kron(ones(nbData,1), speye(nbVarX)); 
	Su = sparse(zeros(nbVarX*(nbData-1), nbVarU*(nbData-1)));
	for t=1:nbData-1
		id1 = (t-1)*nbVarX+1:t*nbVarX;
		id2 = t*nbVarX+1:(t+1)*nbVarX;
		id3 = (t-1)*nbVarU+1:t*nbVarU;
		Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);
		Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
		Su(id2,id3) = B(:,:,t);	
	end
end