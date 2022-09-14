function demo_OC_LQT_fullQ01
% Batch LQT exploiting full Q matrix (e.g., by constraining the motion to pass through a common point at different time steps).
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
% nbData = 40; %Number of datapoints
nbPoints = 3; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8; %dt^nbDeriv;	%Control cost in LQR
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

% %Constraining the motion before and after the keypoints to cross at given time steps
% toff = 8;
% for k=1:nbPoints-1
% 	id2(1:2,1) = id(:,k) - toff * nbVar;
% 	id2(1:2,2) = id(:,k) + toff * nbVar;
% 	MuQ(id2(:,1)) = MuQ(id(:,1));
% 	MuQ(id2(:,2)) = MuQ(id(:,1));
% 	Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
% end

% %Constraining the velocity before and after the keypoints to be correlated, effectively creating a straight line (version 1)
% toff = 4;
% for k=1:nbPoints-1
% 	id2(1:2,1) = id(:,k) - toff * nbVar + nbVarPos;
% 	id2(1:2,2) = id(:,k) + toff * nbVar + nbVarPos;
% 	Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
% end

%Constraining the velocity before and after the keypoints to be correlated, effectively creating a straight line (version 2)
toff = 12;
for k=1:nbPoints-1
	for t=1:toff
		id2(1:2,1) = id(:,k) - t * nbVar + nbVarPos;
		id2(1:2,2) = id(:,k) + t * nbVar + nbVarPos;
		Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
		Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
		Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
		Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
	end
end

% %Constraining the velocity before and after the keypoints to form a desired angle (version 1)
% a = pi/2; %desired angle
% V = [cos(a) -sin(a); sin(a) cos(a)]; %rotation matrix
% toff = 12;
% for k=1:nbPoints-1
% 	id2(1:2,1) = id(:,k) - toff * nbVar + nbVarPos;
% 	id2(1:2,2) = id(:,k) + toff * nbVar + nbVarPos;
% 	Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,1), id2(:,2)) = V;
% 	Q(id2(:,2), id2(:,1)) = V';
% end

% %Constraining the velocity before and after the keypoints to form a desired angle (version 2)
% a = pi/2; %desired angle
% V = [cos(a) -sin(a); sin(a) cos(a)]; %rotation matrix
% toff = 12;
% for k=1:nbPoints-1
% 	for t=1:toff
% 		id2(1:2,1) = id(:,k) - t * nbVar + nbVarPos;
% 		id2(1:2,2) = id(:,k) + t * nbVar + nbVarPos;
% 		Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 		Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 		Q(id2(:,1), id2(:,2)) = V;
% 		Q(id2(:,2), id2(:,1)) = V';
% 	end
% end

% %Constraining the velocities when reaching the keypoints to be correlated (version 1)
% toff = 12;
% for k=1:nbPoints-1
% 	id2(1:2,1) = id(:,k) - toff * nbVar + nbVarPos;
% 	id2(1:2,2) = id(:,k+1) - toff * nbVar + nbVarPos;
% 	Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
% end
	
% %Constraining the velocities when reaching the keypoints to be correlated (version 2)
% toff = 12;
% for k=1:nbPoints-1
% 	for t=1:toff
% 		id2(1:2,1) = id(:,k) - t * nbVar + nbVarPos;
% 		id2(1:2,2) = id(:,k+1) - t * nbVar + nbVarPos;
% 		Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 		Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 		Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% 		Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;	
% 	end
% end

% %Constraining the position and velocity of the first and last point to be correlated
% id2(1:2,1) = 1:nbVarPos;
% id2(1:2,2) = [1:nbVarPos] + (nbData-1) * nbVar;
% MuQ(id2(:,2)) = MuQ(id2(:,1));
% Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
% toff = 4;
% for t=1:toff
% 	id2(1:2,1) = ([1:nbVarPos] + (nbData-1)*nbVar) - t * nbVar + nbVarPos;
% 	id2(1:2,2) = [1:nbVarPos] + t * nbVar + nbVarPos;
% 	Q(id2(:,1), id2(:,1)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,2)) = eye(nbVarPos)*1E0;
% 	Q(id2(:,1), id2(:,2)) = -eye(nbVarPos)*1E0;
% 	Q(id2(:,2), id2(:,1)) = -eye(nbVarPos)*1E0;
% end

%%Constraining R (same control command applied during a time interval)
%toff = 12;
%for k=1:nbPoints-1
%	tt = tl(k) + [-toff,toff];
%	id2 = tt(1) * nbVarPos +1 : tt(2) * nbVarPos;
%	R(id2,id2) = -1E-1 * kron(ones(2*toff), eye(nbVarPos)) + 2 * 1E-1 * eye(length(id2));
%end



%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = zeros(nbVar,1);
u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 800 800],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(rx(1,1), rx(2,1), '.','markersize',35,'color',[0 0 0]);
for k=1:nbPoints-1
	plot(rx(1,tl(k)-toff), rx(2,tl(k)-toff), '.','markersize',25,'color',[0 .6 0]);
	plot(rx(1,tl(k)+toff), rx(2,tl(k)+toff), '.','markersize',25,'color',[0 0 .8]);
end
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',35,'color',[.8 0 0]);
axis equal; 
% print('-dpng','graphs/LQT_fullQ01a.png');

%Plot 1D
figure; hold on;
plot(u(1:nbVarPos:end),'k-');
plot(tl, 0, '.','markersize',25,'color',[0 .6 0]);
xlabel('t'); ylabel('u_1');

% %Visualize Q
% xlim = [1 size(Q,1); 1 size(Q,1)];
% figure('position',[1030 10 1000 1000],'color',[1 1 1]); hold on; axis off;
% set(0,'DefaultAxesLooseInset',[0,0,0,0]);
% set(gca,'LooseInset',[0,0,0,0]);
% colormap(gca, flipud(gray));
% imagesc(abs(Q));
% plot([xlim(1,1),xlim(1,1:2),xlim(1,2:-1:1),xlim(1,1)], [xlim(2,1:2),xlim(2,2:-1:1),xlim(2,1),xlim(2,1)],'-','linewidth',2,'color',[0,0,0]);
% axis equal; axis ij;
% print('-dpng','graphs/MPC_fullQ_Q01b.png');

waitfor(h);
