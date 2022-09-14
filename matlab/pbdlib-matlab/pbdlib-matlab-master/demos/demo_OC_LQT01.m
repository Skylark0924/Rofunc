function demo_OC_LQT01
% Batch LQT with viapoints and simple/double/triple integrator system.
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
nbData = 100; %Number of datapoints
nbPoints = 2; %Number of viapoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 1; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-1; %Time step duration
q = 1E0; %Tracking cost in LQR
r = 1E-3; %dt^nbDeriv;	%Control cost in LQR

%Time occurrence of viapoints
tl = linspace(1, nbData, nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * nbVar + [1:nbVarPos]';


%% Dynamical System settings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nbDeriv==1
	%Build Su0 and Sx0 transfer matrices (for nbDeriv=1)
	Su0 = [zeros(nbVar, nbVar*(nbData-1)); kron(tril(ones(nbData-1)), eye(nbVar)*dt)];
	Sx0 = kron(ones(nbData,1), eye(nbVar));
else
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
	%Build Su0 and Sx0 transfer matrices
	Su0 = sparse(nbVar*nbData, nbVarPos*(nbData-1));
	Sx0 = kron(ones(nbData,1), speye(nbVar));
	M = B;
	for n=2:nbData
		id1 = (n-1)*nbVar+1:nbData*nbVar;
		Sx0(id1,:) = Sx0(id1,:) * A;
		id1 = (n-1)*nbVar+1:n*nbVar; 
		id2 = 1:(n-1)*nbVarPos;
		Su0(id1,id2) = M;
		M = [A*M(:,1:nbVarPos), M]; 
	end
end
Su = Su0(idx,:);
Sx = Sx0(idx,:);


%% Task setting (viapoints passing)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = rand(nbVarPos, nbPoints); %Viapoints
Q = speye(nbVarPos * nbPoints) * q; %Tracking weight matrix
R = speye(nbVarPos * (nbData-1)) * r; %Control weight matrix (at trajectory level)


%% Batch LQT reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = zeros(nbVar,1);
u = (Su' * Q * Su + R) \ Su' * Q * (Mu(:) - Sx * x0); 
rx = reshape(Sx0*x0+Su0*u, nbVar, nbData); %Reshape data for plotting
u = reshape(u, 2, nbData-1); %Reshape data for plotting
% uSigma = inv(Su' * Q * Su + R); % + eye((nbData-1)*nbVarU) * 1E-10;
% xSigma = Su0 * uSigma * Su0';


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 600 600]); hold on; axis off;
% for t=1:nbData
% 	plotGMM(rx(1:2,t), xSigma(nbVar*(t-1)+[1,2],nbVar*(t-1)+[1,2]).*1E-6, [.2 .2 .2], .1);
% end	
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), '.','markersize',35,'color',[0 0 0]);
plot(Mu(1,:), Mu(2,:), '.','markersize',35,'color',[.8 0 0]);
axis equal; 
% print('-dpng','graphs/demo_OC_LQT01.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = min(nbVar,4);
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$'};
figure('position',[620 10 600 850],'color',[1 1 1]); 
%State profile
for j=1:nbVar
	subplot(nbVar+nbVarPos,1,j); hold on;
	plot(rx(j,:), '-','linewidth',2,'color',[0 0 0]);
	if j<=nbVarPos
		plot(tl, Mu(j,:), '.','markersize',40,'color',[.8 0 0]);
	end
	axis([1, nbData, min(rx(j,:))-1E-2, max(rx(j,:))+1E-2]);
	ylabel(labList{j},'fontsize',18,'interpreter','latex');
end
%Control profile
for j=1:nbVarPos
	subplot(nbVar+nbVarPos,1,nbVar+j); hold on;
	plot(u(j,:), '-','linewidth',2,'color',[0 0 0]);
	axis([1, nbData-1, min(u(j,:))-1E-6, max(u(j,:))+1E-6]);
	ylabel(['$u_' num2str(j) '$'],'fontsize',18,'interpreter','latex');
end
xlabel('$t$','fontsize',18,'interpreter','latex');

pause;
close all;