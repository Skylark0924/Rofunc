function demo_Kalman01
% Kalman filter computed as a feedback term or as a product of Gaussians.
%
% If this code is useful for your research, please cite the related publication:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
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
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 0.01; %Time step duration

Sx = diag([1E-1, 1E-4, 1E-1*dt, 1E-4*dt]); %Model accurate on y and inaccurate on x
Sy = diag([1E-4, 1E-1]); %Sensor accurate on x and inaccurate on y 

[Vx,Dx] = eig(Sx); %Eigencomponents
[Vy,Dy] = eig(Sy); %Eigencomponents


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
C = [eye(nbVarPos), zeros(nbVar-nbVarPos, nbVarPos)]; %Position sensor
% C = eye(nbVarPos); %Position+velocity sensor


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = [ones(1, nbData-1); zeros(1,nbData-1)]; %Simulate constant acceleration along x
xd =  B * u(:,1);
for t=1:nbData-1
	xd(:,t+1) = A * xd(:,t) + B * u(:,t);
end


%% Reproduction with Kalman filter (see e.g., https://www.mathworks.com/help/control/ug/kalman-filtering.html)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ex = Vx * Dx.^.5 * randn(nbVar,nbData); %Simulate noise on state
ey = Vy * Dy.^.5 * randn(nbVarPos,nbData); %Simulate noise on sensor

%Simulate system evolution without Kalman filter
x = zeros(nbVar,1);
for t=1:nbData-1
	%Time update
	x = A * x + B * u(:,t) + Vx*Dx.^.5 * ex(:,t); %x[t+1|t]
	%Simulate noise
	y = C * x + Vy*Dy.^.5 * ey(:,t); %Sensor data
	%Log data
	r.y(:,t) = y; 
end

%Simulate system evolution with Kalman filter
x = zeros(nbVar,1);
P = Sx;
for t=1:nbData-1
	%Time update
	x = A * x + B * u(:,t); %x[t+1|t]
	P = A * P * A' + Sx; %P[t+1|t]
	%Simulate noise
	y = C * x + Vy*Dy.^.5 * ey(:,t); %Sensor data
	%Measurement update 
	K = P * C' / (C * P * C' + Sy); %Kalman gain	
	x = x + K * (y - C * x); %x[t|t]
	P = (eye(nbVar) - K * C) * P; %P[t|t]
	%Log data
	r.yf(:,t) = C * x;
end


%% Reproduction with Kalman filter as product of Gaussians (PoG)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(nbVar,1);
S = Sx;
for t=1:nbData-1
	%Gaussian 1
	Mu(:,1) = A * x + B * u(:,t); 
	Sigma(:,:,1) = A * S * A' + Sx; 
	Q(:,:,1) = inv(A * S * A' + Sx); 
	QMu(:,1) = Q(:,:,1) * Mu(:,1);
	
	%Simulate noise
	y = C * Mu(:,1) + Vy*Dy.^.5 * ey(:,t); %Sensor data
	
	%Gaussian 2
	Mu(:,2) = pinv(C) * y;
	Sigma(:,:,2) = pinv(C) * Sy * pinv(C)'; 
	Q(:,:,2) = C' / Sy * C;
	QMu(:,2) = C' / Sy * y; 

	%Product of Gaussians (PoG)
	QTmp = zeros(nbVar);
	MuTmp = zeros(nbVar,1);
	for i=1:2
		QTmp = QTmp + Q(:,:,i);
		MuTmp = MuTmp + QMu(:,i);
	end
	S = inv(QTmp);
	x = S * MuTmp;

	%Log data
	r.yf2(:,t) = C * x; 
	r.Mu(:,:,t) = Mu;
	r.Sigma(:,:,:,t) = Sigma;
	r.S(:,:,t) = S;
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1200 800],'color',[1 1 1]);
subplot(nbVarPos+1,1,1); hold on; axis off;
h(1) = plot(xd(1,:), xd(2,:), '-','linewidth',2,'color',[.7 .7 .7]);
h(2) = plot(r.y(1,:), r.y(2,:), '-','linewidth',1,'color',[.2 .2 .2]);
h(3) = plot(r.yf(1,:), r.yf(2,:), '-','linewidth',2,'color',[.8 0 0]);
h(4) = plot(r.yf2(1,:), r.yf2(2,:), '--','linewidth',2,'color',[0 .6 0]);
for t=round(linspace(1,nbData-1,5))
	hh(1,:) = plotGMM(r.Mu(1:2,1,t), r.Sigma(1:2,1:2,1,t), [.8 0 .8], .3); %Model
	hh(2,:) = plotGMM(r.Mu(1:2,2,t), r.Sigma(1:2,1:2,2,t), [0 .8 .8], .3); %Sensor
	hh(3,:) = plotGMM(r.yf2(:,t), r.S(1:2,1:2,t), [.8 0 0], .6); %Product
end
legend([h, hh(:,1)'], {'Ground truth','Observations','Filtered process','Filtered process (computed with PoG)','Model covariance','Sensor covariance','Resulting PoG'},'location','eastoutside','fontsize',14);
axis equal; 

%Timeline plot
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'}; 
for j=1:nbVarPos
	subplot(nbVarPos+1,1,1+j); hold on;	
	plot(xd(j,:), '-','linewidth',.5,'color',[.7 .7 .7]);
	plot(r.y(j,:), '-','linewidth',1,'color',[0 0 0]);
	plot(r.yf(j,:), '-','linewidth',2,'color',[.8 0 0]);
	plot(r.yf2(j,:), '--','linewidth',2,'color',[0 .6 0]);
	axis([1, nbData, min(x(:))-5E-1, max(x(:))+5E-1]);
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','fontsize',16,'interpreter','latex');
	ylabel(labList{j},'fontsize',16,'interpreter','latex');
end

%print('-dpng','graphs/demo_Kalman01.png');
pause;
close all;