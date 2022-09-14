function demo_OC_LQT_recursive01
% Recursive computation of linear quadratic tracking (with feedback and feedforward terms).
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
nbPoints = 1; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-1; %Time step duration
rfactor = 1E0; %dt^nbDeriv;	%Control cost in LQR
R = eye(nbVarPos) * rfactor; %Control cost matrix


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


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(1, nbData/2, nbPoints+1);
tl = round(tl(2:end)); 
% MuQ = zeros(nbVar*nbData,1); 
% Q = zeros(nbVar*nbData);
for t=1:length(tl)
	id(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
% 	MuQ(id(:,t)) = rand(nbVarPos,1) - 0.5;
% % 	Q(id(:,t), id(:,t)) = eye(nbVarPos);
% 	
% 	id2(:,t) = [1:nbVar] + (tl(t)-1) * nbVar;
% 	Q(id2(:,t), id2(:,t)) = diag([ones(1,nbVarPos), ones(1,nbVarPos)]);
end
% Mu = reshape(MuQ, nbVar, nbData);

MuQ = kron(ones(nbData,1), [1; 1; zeros(nbVar-nbVarPos,1)]);
Q = blkdiag(zeros((nbData/2)*nbVar), kron(eye(nbData/2), diag([1,1,zeros(1,nbVar-nbVarPos)])) * 1E1);
Mu = reshape(MuQ, nbVar, nbData);

% Q(end-1:end,end-1:end) = eye(2) .* 1E3;
% Q(end-10:end,end-10:end)


% %% Iterative LQT reproduction (finite horizon, discrete version)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P = zeros(nbVar, nbVar, nbData);
% P(:,:,end) = Q(end-nbVar+1:end,end-nbVar+1:end);
% d = zeros(nbVar, nbData);
% 
% %Backward computation
% for t=nbData-1:-1:1
% 	P(:,:,t) = Q((t-1)*nbVar+1:t*nbVar,(t-1)*nbVar+1:t*nbVar) - A' * (P(:,:,t+1) * B / (B' * P(:,:,t+1) * B + R) * B' * P(:,:,t+1) - P(:,:,t+1)) * A;
% 	d(:,t) = (A' - A' * P(:,:,t+1) * B / (R + B' * P(:,:,t+1) * B) * B' ) * (P(:,:,t+1) * (A * Mu(:,t) - Mu(:,t+1)) + d(:,t+1));
% end
% 
% %Reproduction with feedback (FB) and feedforward (FF) terms
% X = zeros(nbVar,1);
% for t=1:nbData
% 	K(:,:,t) = (B' * P(:,:,t) * B + R) \ B' * P(:,:,t) * A; %FB gain
% 
% % 	%Test ratio between kp and kv
% % 	figure; hold on;
% % 	plotGMM(zeros(2,1), K(:,1:2,t), [.8 0 0],.3);
% % 	plotGMM(zeros(2,1), K(:,3:4,t), [.8 0 0],.3);
% % 	axis equal;
% % 	pause;
% % 	close all;
% 
% 	uff(:,t) = -(B' * P(:,:,t) * B + R) \ B' * (P(:,:,t) * (A * Mu(:,t) - Mu(:,t)) + d(:,t)); %Feedforward term
% 	u = K(:,:,t) * (Mu(:,t) - X) + uff(:,t); %Acceleration command with FB and FF terms
% 	X = A * X + B * u; %Update of state vector
% 	
% 	rx(:,t) = X; %Log data
% end


%% Iterative LQT reproduction (as in Table 2 on p.2 of http://web.mst.edu/~bohner/papers/tlqtots.pdf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = zeros(nbVarPos, nbVar, nbData-1);
Kff = zeros(nbVarPos, nbVar, nbData-1);
P = zeros(nbVar, nbVar, nbData);
P(:,:,end) = Q(end-nbVar+1:end,end-nbVar+1:end);
v = zeros(nbVar, nbData);
v(:,end) = Q(end-nbVar+1:end,end-nbVar+1:end) * Mu(:,end);

%Backward computation
for t=nbData-1:-1:1
	Kff(:,:,t) = (B' * P(:,:,t+1) * B + R) \ B'; %FF
	K(:,:,t) = Kff(:,:,t) * P(:,:,t+1) * A; %FB
	P(:,:,t) = A' * P(:,:,t+1) * (A - B * K(:,:,t)) + Q((t-1)*nbVar+1:t*nbVar,(t-1)*nbVar+1:t*nbVar);
	v(:,t) = (A - B * K(:,:,t))' * v(:,t+1) + Q((t-1)*nbVar+1:t*nbVar,(t-1)*nbVar+1:t*nbVar) * Mu(:,t);
	kp(:,t) = eigs(K(:,1:2,t));
	kv(:,t) = eigs(K(:,3:4,t));
	%Test ratio between kp and kv
	ratio(:,t) = kv(:,t) ./ (2*kp(:,t)).^.5;
end

%Reproduction with feedback (FB) and feedforward (FF) terms
x = zeros(nbVar,1);
rx(:,1) = x; %Log data
for t=2:nbData
	u = -K(:,:,t-1) * x + Kff(:,:,t-1) * v(:,t); %Acceleration command with FB and FF terms
	x = A * x + B * u; %Update of state vector
	rx(:,t) = x; %Log data
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800]); hold on; axis off;
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), '.','markersize',50,'color',[0 0 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',50,'color',[.8 0 0]);
axis equal; 
%print('-dpng','graphs/LQT_x01.png');


%% Plot timeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[820 10 800 800]);
% %phase
% alpha = 2;
% tIn = [1:nbData]*dt;
% sIn = exp(-alpha * tIn);
% subplot(2,1,1); hold on; grid on;
% plot(tIn, sIn, 'r-');
% plot(tIn, 1-sIn, 'r--');
% xlabel('t'); ylabel('s');

% %Q
% subplot(2,1,1); hold on; grid on;
% plot(diag(Q(1:4:end,1:4:end)), '-','linewidth',2,'color',[0 0 0]);
% xlabel('t'); ylabel('Q');

% %kp-kv ratio
% subplot(2,1,1); hold on; grid on;
% plot(ratio(1,:), '-','linewidth',2,'color',[0 0 0]);
% xlabel('t'); ylabel('kp-kv ratio');
% axis([1, nbData/2+20, 0, max(ratio(1,:))+.1]);

%x
subplot(2,1,1); hold on; grid on;
plot([nbData/2 nbData], [MuQ(1) MuQ(1)], '-','linewidth',8,'color',[.8 0 0]);
plot(rx(1,:), '-','linewidth',4,'color',[0 0 0]);
axis([1, nbData/2+20, 0, 1.05]);
set(gca,'linewidth',2,'xtick',100,'ytick',MuQ(1),'xticklabel',{},'yticklabel',{});
xlabel('$t$','interpreter','latex','fontsize',34); 
ylabel('$x$','interpreter','latex','fontsize',34);

%Kp,Kv
subplot(2,1,2); hold on; grid on;
plot(kp(1,:), '-','linewidth',4,'color',[0 0 .8]);
plot(kv(1,:), '-','linewidth',4,'color',[0 .6 0]);
axis([1, nbData/2+20, 0, max([kp(1,:), kv(1,:)])+.1]);
set(gca,'linewidth',2,'xtick',100,'ytick',[kv(1,120),kp(1,120)],'xticklabel',{},'yticklabel',{});
xlabel('$t$','interpreter','latex','fontsize',34); 
ylabel('$\kappa^{\scriptscriptstyle{P}},\kappa^{\scriptscriptstyle{V}}$','interpreter','latex','fontsize',34);
%print('-dpng','graphs/demo_MPC_iterativeLQT01.png');

% %Plot P
% figure('position',[820 10 800 800]); hold on; axis off;
% for t=round(linspace(nbData-1, 1, 20))
% 	plotGMM(zeros(2,1), P(1:2,1:2,t), [.7 .7 .7], .1);
% end

pause;
close all;