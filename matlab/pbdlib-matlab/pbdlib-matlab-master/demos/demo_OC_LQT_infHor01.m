function demo_OC_LQT_infHor01
% Discrete infinite horizon linear quadratic regulation (with precision matrix only on position).
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
% Written by Sylvain Calinon (http://calinon.ch/) and Danilo Bruno (danilo.bruno@iit.it)
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
nbData = 100; %Number of datapoints
nbRepros = 10; %Number of reproductions

nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
nbVar = nbVarPos * nbDeriv; %Dimension of state vector in the tangent space
dt = 1E-2; %Time step duration
rfactor = 4E-2;	%Control cost in LQR 

%Control cost matrix
R = eye(nbVarPos) * rfactor;
% [Ar,~] = qr(randn(nbVarPos));
% R = Ar*diag([1,.1])*Ar' * rfactor;

%Target and desired covariance
xTar = [randn(nbVarPos,1); zeros(nbVarPos*(nbDeriv-1),1)];

[Ar,~] = qr(randn(nbVarPos));
xCov = Ar * diag(rand(nbVarPos,1)) * Ar' * 1E-1;


%% Discrete dynamical System settings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


%% Iterative discrete LQR with infinite horizon
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = blkdiag(inv(xCov), zeros(nbVarPos*(nbDeriv-1))); %Precision matrix
P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
L = (B' * P * B + R) \ B' * P * A; %Feedback gain (discrete version)

%Test ratio between kp and kv
if nbDeriv>1
	kp = eigs(L(:,1:nbVarPos));
	kv = eigs(L(:,nbVarPos+1:end));
	ratio = kv ./ (2 * kp.^.5)
end

% figure; hold on;
% plotGMM(zeros(2,1), L(:,1:2), [.8 0 0],.3);
% plotGMM(zeros(2,1), L(:,3:4), [.8 0 0],.3);
% axis equal;
% pause;
% close all;

for n=1:nbRepros
	x = [ones(nbVarPos,1)+randn(nbVarPos,1)*5E-1; zeros(nbVarPos*(nbDeriv-1),1)];
	for t=1:nbData		
		r(n).Data(:,t) = x; 
		u = L * (xTar - x); %Compute acceleration (with only feedback terms)
		x = A * x + B * u;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plots
figure('position',[10,10,650,650]); hold on; axis off; grid off; 
plotGMM(xTar(1:2), xCov(1:2,1:2), [.8 0 0], .3);
for n=1:nbRepros
% 	for t=1:nbData
% 		coltmp = [.3 1 .3] * (nbData-t)/nbData;
% 		plot(r(n).Data(1,t), r(n).Data(2,t), '.','markersize',12,'color',coltmp);
% 	end
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',1,'color',[0 0 0]);
end
axis equal; 
%print('-dpng','graphs/demo_MPC_infHor01.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'};
figure('position',[720 10 600 650],'color',[1 1 1]); 
for j=1:nbVar
subplot(nbVar+1,1,j); hold on;
for n=1:nbRepros
	plot(r(n).Data(j,:), '-','linewidth',1,'color',[0 0 0]);
end
if j<7
	ylabel(labList{j},'fontsize',14,'interpreter','latex');
end
end
%Speed profile
if nbDeriv>1
subplot(nbVar+1,1,nbVar+1); hold on;
for n=1:nbRepros
	sp = sqrt(r(n).Data(3,:).^2 + r(n).Data(4,:).^2);
	plot(sp, '-','linewidth',1,'color',[0 0 0]);
end
ylabel('$|\dot{x}|$','fontsize',14,'interpreter','latex');
xlabel('$t$','fontsize',14,'interpreter','latex');
end
%print('-dpng','graphs/demo_LQR_infHor01.png');

pause;
close all;