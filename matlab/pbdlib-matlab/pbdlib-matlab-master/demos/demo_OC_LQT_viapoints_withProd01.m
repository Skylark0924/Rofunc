function demo_OC_LQT_viapoints_withProd01
% Batch LQT with viapoints computed as a product of trajectory distributions in control space.
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
nbPoints = 3; %Number of Gaussians in the GMM
nbData = nbPoints * 30; %Number of datapoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-8;	%Control cost in LQR
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
A = kron(A1d, eye(nbVarPos)); %Discrete nD
B = kron(B1d, eye(nbVarPos)); %Discrete nD

%Build Sx and Su transfer matrices 
Su = zeros(nbVar*nbData, nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1), eye(nbVar)); 
M = B;
for n=2:nbData
	id1 = (n-1)*nbVar+1:nbData*nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*nbVar+1:n*nbVar; 
	id2 = 1:(n-1)*nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(1, nbData, nbPoints+1);
tl = round(tl(2:end)); 
MuQ = zeros(nbVar*nbData, 1); 
Q = zeros(nbVar*nbData);
Qi = zeros(nbVar*nbData, nbVar*nbData, nbPoints);
for i=1:nbPoints
	id(:,i) = [1:nbVarPos] + (tl(i)-1) * nbVar;
	Q(id(:,i), id(:,i)) = eye(nbVarPos);
	Qi(id(:,i), id(:,i), i) = eye(nbVarPos);
	MuQ(id(:,i)) = rand(nbVarPos,1) - 0.5;
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Standard computation
x0 = zeros(nbVar, 1); 
u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
x = Sx * x0 + Su * u;
uSigma = inv(Su' * Q * Su + R); 
xSigma = Su * uSigma * Su';

% figure; j=1;
% subplot(2,1,1); plot(diag(xSigma(j:4:end,j:4:end)));
% subplot(2,1,2); plot(diag(uSigma(j:2:end,j:2:end)));
% pause; close all;
% return;

%Computation as product of trajectory distributions in control space
xi = zeros(nbData*nbVar, nbPoints);
ui = zeros((nbData-1)*nbVarPos, nbPoints);
uiQ = zeros((nbData-1)*nbVarPos, (nbData-1)*nbVarPos, nbPoints);
% uiSigma = zeros((nbData-1)*nbVarPos, (nbData-1)*nbVarPos, nbPoints);
% xiSigma = zeros(nbData*nbVar, nbData*nbVar, nbPoints);
Ri = speye((nbData-1)*nbVarPos) * 1E-15; %to avoid numerical issue
u2 = zeros((nbData-1)*nbVarPos, 1);
for i=1:nbPoints
	ui(:,i) = (Su' * Qi(:,:,i) * Su + Ri) \ Su' * Qi(:,:,i) * (MuQ - Sx * x0); 
	xi(:,i) = Sx * x0 + Su * ui(:,i);
	uiQ(:,:,i) = Su' * Qi(:,:,i) * Su + Ri; 
% 	uiSigma(:,:,i) = inv(uiQ(:,:,i));
% 	xiSigma(:,:,i) = Su * uiSigma(:,:,i) * Su';
	u2 = u2 + uiQ(:,:,i) * ui(:,i); 
end
u2 = (sum(uiQ,3) + R) \ u2;
x2 = Sx * x0 + Su * u2;

% figure; j=1;
% subplot(2,1,1); hold on;
% for i=1:nbPoints
% 	plot(diag(xiSigma(j:4:end,j:4:end,i)));
% end
% subplot(2,1,2); hold on;
% for i=1:nbPoints
% 	plot(diag(uiSigma(j:2:end,j:2:end,i)));
% end
% pause; close all;
% return;


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(nbPoints);
%x1-x2 
figure('position',[10 10 600 600],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
for t=1:nbData
	plotGMM(x((t-1)*nbVar+[1:2]), xSigma((t-1)*nbVar+[1,2], (t-1)*nbVar+[1,2]).*1E-5, [.2 .2 .2], .1);
end
% for i=1:nbPoints
% 	plot(xi(1:4:end,i), xi(2:4:end,i), '-','linewidth',2,'color',clrmap(i,:));
% end
plot(x(1:4:end), x(2:4:end), '-','linewidth',2,'color',[0 0 0]);
plot(x(1), x(2), '.','markersize',15,'color',[0 0 0]);
plot(x2(1:4:end), x2(2:4:end), '-','linewidth',2,'color',[0 .6 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',35,'color',[.8 0 0]);
axis equal;
% print('-dpng','graphs/LQT_x01.png');

%u1-u2
figure('position',[10 700 600 600],'color',[1 1 1],'name','u1-u2 plot'); hold on; axis off;
for t=1:nbData-1
	plotGMM(u((t-1)*nbVarPos+[1:2]), uSigma((t-1)*nbVarPos+[1,2], (t-1)*nbVarPos+[1,2]).*1E-7, [.2 .2 .2], .02);
end
% for i=1:nbPoints
% 	plot(ui(1:2:end,i), ui(2:2:end,i), '-','linewidth',2,'color',clrmap(i,:));
% end
plot(u(1:2:end), u(2:2:end), '-','linewidth',1,'color',[0 0 0]);
plot(u(1), u(2), '.','markersize',20,'color',[0 0 0]);
plot(u2(1:2:end), u2(2:2:end), '-','linewidth',1,'color',[0 .6 0]);
axis equal;
% print('-dpng','graphs/LQT_u01.png');


%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[620 10 1000 1290],'color',[1 1 1]); 
for j=1:nbVarPos
	subplot(nbVar,1,j); hold on;
	if j>nbVarPos
		plot([1,nbData],[0,0],':','color',[.5 .5 .5]);
	end
	S = diag(xSigma(j:nbVar:end, j:nbVar:end)) .* 1E-3;
	patch([1:nbData nbData:-1:1], [x(j:nbVar:end)-S(:); flipud(x(j:nbVar:end)+S(:))], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
	plot(x(j:nbVar:end), '-','linewidth',1,'color',[0 0 0]);
	for i=1:nbPoints
% 		errorbar(tl(i), MuQ(id(j,i)), Q(id(j,i),id(j,i)).^-.5, 'color',[.8 0 0]);
		plot(tl(i), MuQ(id(j,i)), '.','markersize',15,'color',[.8 0 0]);
	end
	ylabel(['$x_' num2str(j) '$'],'fontsize',14,'interpreter','latex');
	xlabel('$t$','fontsize',14,'interpreter','latex');
	set(gca,'xtick',[],'ytick',[]);
	axis tight;
end

%Control profile
for j=1:nbVarPos
	subplot(nbVar,1,nbVarPos+j); hold on;
	S = diag(uSigma(j:2:end, j:2:end)) .* 1E-7;
	patch([1:nbData-1 nbData-1:-1:1], [u(j:2:end)-S(:); flipud(u(j:2:end)+S(:))], [.2 .2 .2],'edgecolor',[0 0 0],'facealpha', .2, 'edgealpha', .2);
% 	for i=1:nbPoints
% 		plot(ui(j:2:end,i).*1E1, '-','linewidth',2,'color',clrmap(i,:));
% 	end
	plot(u(j:2:end,:), '-','linewidth',1,'color',[0 0 0]);
	ylabel(['$u_' num2str(j) '$'],'fontsize',14,'interpreter','latex');
	xlabel('$t$','fontsize',14,'interpreter','latex');
	set(gca,'xtick',[],'ytick',[]);
	axis tight;
end
% print('-dpng','graphs/LQT_txu01.png');

pause;
close all;