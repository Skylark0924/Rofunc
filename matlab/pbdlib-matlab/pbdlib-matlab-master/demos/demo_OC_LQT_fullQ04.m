function demo_OC_LQT_fullQ04
% Batch LQT problem exploiting full Q matrix to constrain the motion of two agents in a ballistic task mimicking a bimanual tennis serve problem
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
nbAgents = 3; %Number of agents
nbVarPos = 2 * nbAgents; %Dimension of position data (here: x1,x2 for the three agents)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * (nbDeriv+1); %Dimension of state vector (incl. offset to apply gravity effect)
nbVarU = 4; %Number of control variables (acceleration commands of the two hands)
dt = 1E-2; %Time step duration
rfactor = 1E-8; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarU) * rfactor; %Control cost matrix

m = [2*1E-1, 2*1E-1, 2*1E-1]; %Mass of agents (left hand, right hand and ball) 
g = [0; -9.81]; %Gravity vector

tEvent = [50, 100, 150]; %Time steps for ball release and ball hitting 
x01 = [1.6; 0]; %Inital position of Agent 1 (left hand) 
x02 = [2; 0]; %Inital position of Agent 2 (right hand) 
xTar = [1; -.2]; %Final position of Agent 3 (ball)

% tEvent = [40, 110, 190]; %40 50 %Time steps for ball release and ball hitting 
% x01 = [1.4; .2]; %Inital position of Agent 1 (left hand) 
% x02 = [2; -.1]; %Inital position of Agent 2 (right hand) 
% xTar = [1.3; 0]; %Final position of Agent 3 (ball)


%% Linear dynamical system parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ac1 = kron([0, 1, 0; 0, 0, 1; 0, 0, 0], eye(2));
Bc1 = kron([0; 1; 0], eye(2));
Ac = kron(eye(nbAgents), Ac1);
Bc = [kron(eye(2), Bc1); zeros(6,nbVarU)]; %Ball is not directly controlled

Ad = eye(nbVar) + Ac * dt; %Parameters for discrete dynamical system
Bd = Bc * dt; %Parameters for discrete dynamical system

%Set heterogeneous A and B
A = repmat(Ad, [1,1,nbData-1]);
B = repmat(Bd, [1,1,nbData-1]);

%Set Agent 3 state (ball) equals to Agent 1 state (left hand) until tEvent(1)
A(13:16,:,1:tEvent(1)) = 0;
% A(13:14,13:14,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]);
A(13:14,1:2,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]);
A(13:14,3:4,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]) * dt;
A(15:16,3:4,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]);
A(15:16,5:6,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]) * dt;	
% A(13:18,:,1)

%Set Agent 3 state (ball) equals to Agent 2 state (right hand) at tEvent(2)
A(13:16,:,tEvent(2)) = 0;
% A(13:14,13:14,tEvent(2)) = eye(2);
A(13:14,7:8,tEvent(2)) = eye(2);
A(13:14,9:10,tEvent(2)) = eye(2) * dt;
A(15:16,9:10,tEvent(2)) = eye(2);
A(15:16,11:12,tEvent(2)) = eye(2) * dt;
% A(13:18,:,tEvent(2))

[Su, Sx] = transferMatrices(A, B); %Build transfer matrices

% figure('position',[10 10 1600 1200]); hold on; axis off;
% imagesc(Su);
% axis tight; axis equal; axis ij;


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);

%Agent 1 (left hand) and Agent 2 (right hand) must come back to initial pose at tEvent(3) and stay here 
for t=tEvent(3):nbData
	id = [1:4] + (t-1) * nbVar; %Left hand
	Q(id,id) = eye(4) * 1E3;
	MuQ(id) = [x01; zeros(2,1)];
	id = [7:10] + (t-1) * nbVar; %Right hand
	Q(id,id) = eye(4) * 1E3;
	MuQ(id) = [x02; zeros(2,1)];

% 	id = [1:2] + (t-1) * nbVar; %Left hand
% 	Q(id,id) = eye(2);
% 	MuQ(id) = x01;
% 	id = [7:8] + (t-1) * nbVar; %Right hand
% 	Q(id,id) = eye(2);
% 	MuQ(id) = x02;

% 	id = [3:4] + (t-1) * nbVar; %Left hand
% 	Q(id,id) = eye(2);
% 	MuQ(id) = zeros(2,1);
% 	id = [9:10] + (t-1) * nbVar; %Right hand
% 	Q(id,id) = eye(2);
% 	MuQ(id) = zeros(2,1);
end

%Agent 3 (ball) must reach desired target at the end of the movement
id = [13:14] + (nbData-1) * nbVar;
Q(id,id) = eye(2);
MuQ(id) = xTar;

%Agent 2 and Agent 3 must meet at tEvent(2) (right hand hitting the ball)
id = [7:8,13,14] + (tEvent(2)-1) * nbVar;
MuQ(id) = rand(4,1);
MuQ(id(3:4)) = MuQ(id(1:2)); %Proposed common meeting point for the two agents (can be any point)
Q(id,id) = eye(4);
Q(id(1:2), id(3:4)) = -eye(2);
Q(id(3:4), id(1:2)) = -eye(2);


%% Batch LQT reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [x01; zeros(2,1); m(1)*g; ...
      x02; zeros(2,1); m(2)*g; ...
      x01; zeros(2,1); m(3)*g]; 
u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
rx = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting

% uSigma = inv(Su' * Q * Su + R); 
% xSigma = Su * uSigma * Su';


%% 2D plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 1200 800],'color',[1 1 1]); hold on; axis off;
% %Uncertainty
% for t=1:nbData
% 	plotGMM(rx(1:2,t), xSigma(nbVar*(t-1)+[1,2],nbVar*(t-1)+[1,2])*3E-7, [0 0 0], .04); %Agent 1 (left hand)
% end	
% for t=1:nbData
% 	plotGMM(rx(7:8,t), xSigma(nbVar*(t-1)+[7,8],nbVar*(t-1)+[7,8])*3E-7, [.6 .6 .6], .08); %Agent 2 (right hand)
% end	
% for t=1:nbData
% 	plotGMM(rx(13:14,t), xSigma(nbVar*(t-1)+[13,14],nbVar*(t-1)+[13,14])*3E-7, [.8 .4 0], .08); %Agent 3 (ball)
% end

%Agents
hf(1) = plot(rx(1,:), rx(2,:), '-','linewidth',4,'color',[0 0 0]); %Agent 1 (left hand)
hf(2) = plot(rx(7,:), rx(8,:), '-','linewidth',4,'color',[.6 .6 .6]); %Agent 2 (right hand)
hf(3) = plot(rx(13,:), rx(14,:), ':','linewidth',4,'color',[.8 .4 0]); %Agent 3 (ball)

%Events
hf(4) = plot(rx(1,1), rx(2,1), '.','markersize',40,'color',[0 0 0]); %Initial position (left hand)
hf(5) = plot(rx(7,1), rx(8,1), '.','markersize',40,'color',[.6 .6 .6]); %Initial position (right hand)
hf(6) = plot(rx(1,tEvent(1)), rx(2,tEvent(1)), '.','markersize',40,'color',[0 .6 0]); %Release of ball
hf(7) = plot(rx(7,tEvent(2)), rx(8,tEvent(2)), '.','markersize',40,'color',[0 0 .8]); %Hitting of ball
hf(8) = plot(xTar(1), xTar(2), '.','markersize',40,'color',[.8 0 0]); %Ball target
plot2DArrow(rx(1:2,tEvent(1)), diff(rx(1:2,tEvent(1):tEvent(1)+1),1,2)*12E0, [0 .6 0], 4, .01);
plot2DArrow(rx(7:8,tEvent(2)), diff(rx(7:8,tEvent(2):tEvent(2)+1),1,2)*12E0, [0 0 .8], 4, .01);
axis equal; axis([1, 2.2, -0.3, 0.2]); 

text(rx(1,1)+.01, rx(2,1)-.01,'$t=0$','interpreter','latex','fontsize',20);
text(rx(1,1)+.01, rx(2,1)-.03,'$t=\frac{3T}{4}$','interpreter','latex','fontsize',20);
text(rx(7,1)+.01, rx(8,1)+.02,'$t=0$','interpreter','latex','fontsize',20);
text(rx(7,1)+.01, rx(8,1),'$t=\frac{3T}{4}$','interpreter','latex','fontsize',20);
text(rx(1,tEvent(1))-.01, rx(2,tEvent(1))-.02,'$t=\frac{T}{4}$','interpreter','latex','fontsize',20);
text(rx(13,tEvent(2))+.01, rx(14,tEvent(2))+.01,'$t=\frac{T}{2}$','interpreter','latex','fontsize',20);
text(xTar(1)+.02, xTar(2),'$t=T$','interpreter','latex','fontsize',20);
legend(hf, {'Left hand motion (Agent 1)','Right hand motion (Agent 2)','Ball motion (Agent 3)', ...
'Left hand initial point','Right hand initial point','Ball releasing point','Ball hitting point','Ball target'}, 'fontsize',20,'location','northwest'); 
legend('boxoff');
%print('-dpng','graphs/iLQR_tennisServe01.png');


% %% 2D animated plot 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 1100 450],'color',[1 1 1]); hold on; axis off;
% plot(rx(1,1), rx(2,1), '.','markersize',40,'color',[0 0 0]); %Initial position (left hand)
% plot(rx(7,1), rx(8,1), '.','markersize',40,'color',[.6 .6 .6]); %Initial position (right hand)
% plot(xTar(1), xTar(2), '.','markersize',40,'color',[.8 0 0]); %Ball target
% axis equal; axis([1, 2.2, -0.3, 0.2]); 
% %Animation
% for t=1:nbData
% 	h(1) = plot(rx(1,1:t), rx(2,1:t), '-','linewidth',4,'color',[0 0 0]); %Agent 1 (left hand)
% 	h(2) = plot(rx(7,1:t), rx(8,1:t), '-','linewidth',4,'color',[.6 .6 .6]); %Agent 2 (right hand)
% 	h(3) = plot(rx(13,1:t), rx(14,1:t), ':','linewidth',4,'color',[.8 .4 0]); %Agent 3 (ball)
% % 	h(1) = patch('xdata',rx(1,[1:t,t:-1:1]),'ydata',rx(2,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[0 0 0],'edgealpha',.2); %Agent 1 (left hand)
% % 	h(2) = patch('xdata',rx(7,[1:t,t:-1:1]),'ydata', rx(8,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[.6 .6 .6],'edgealpha',.2); %Agent 2 (right hand)
% % 	h(3) = patch('xdata',rx(13,[1:t,t:-1:1]),'ydata', rx(14,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[.8 .4 0],'edgealpha',.2); %Agent 3 (ball) %'linestyle',':'
% 	
% 	h(4) = plot(rx(1,t), rx(2,t), '.','markersize',60,'color',[0 0 0]);
% 	h(5) = plot(rx(7,t), rx(8,t), '.','markersize',60,'color',[.6 .6 .6]);
% 	h(6) = plot(rx(13,t), rx(14,t), '.','markersize',50,'color',[.8 .4 0]);
% 	drawnow;
% 	print('-dpng',['graphs/anim/mpc01_' num2str(t,'%.3d') '.png']);
% 	delete(h);
% end


% %% Sampling plot 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 2200 900],'color',[1 1 1]); hold on; axis off;
% nbSamples = 50;
% [V,D] = eig(uSigma);
% plot(rx(1,1), rx(2,1), '.','markersize',40,'color',[0 0 0]); %Initial position (left hand)
% plot(rx(7,1), rx(8,1), '.','markersize',40,'color',[.6 .6 .6]); %Initial position (right hand)
% plot(xTar(1), xTar(2), '.','markersize',40,'color',[.8 0 0]); %Ball target
% for n=1:nbSamples
% 	u2 = u + real(V*D.^.5) * randn(size(u)).*2E-4;
% 	rx = reshape(Sx*x0+Su*u2, nbVar, nbData); %Reshape data for plotting
% 	%Agents
% 	patch('xdata',rx(1,[1:t,t:-1:1]),'ydata',rx(2,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[0 0 0],'edgealpha',.2); %Agent 1 (left hand)
% 	patch('xdata',rx(7,[1:t,t:-1:1]),'ydata', rx(8,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[.6 .6 .6],'edgealpha',.2); %Agent 2 (right hand)
% 	patch('xdata',rx(13,[1:t,t:-1:1]),'ydata', rx(14,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[.8 .4 0],'edgealpha',.2); %Agent 3 (ball) %'linestyle',':
% 	%Events
% 	plot(rx(1,tEvent(1)), rx(2,tEvent(1)), '.','markersize',20,'color',[0 .6 0]); %Release of ball
% 	plot(rx(7,tEvent(2)), rx(8,tEvent(2)), '.','markersize',20,'color',[0 0 .8]); %Hitting of ball
% % 	plot2DArrow(rx(1:2,tEvent(1)), diff(rx(1:2,tEvent(1):tEvent(1)+1),1,2)*12E0, [0 .6 0], 4, .01);
% % 	plot2DArrow(rx(7:8,tEvent(2)), diff(rx(7:8,tEvent(2):tEvent(2)+1),1,2)*12E0, [0 0 .8], 4, .01);
% end
% axis equal; axis([1, 2.2, -0.3, 0.2]); 
% % print('-dpng','graphs/MPC_tennisServe01f.png'); %convert -gravity Center -crop 80x80%+10-50 MPC_tennisServe01f.png MPC_tennisServe_cropped01f.png


% %% Timeline plots
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 2200 1200],'color',[1 1 1]);
% ttl = {'Left hand','','Right hand','','Ball',''};
% %x
% id = [1:2,7:8,13:14];
% for n=1:6
% 	subplot(6,2,1+2*(n-1)); hold on; grid on; box on; title(ttl(n));
% 	plot(rx(id(n),:),'k.');
% 	if n<3
% 		plot([1,tEvent(3)], [rx(id(n),1) rx(id(n),tEvent(3))], '.','markersize',40,'color',[0 0 0]);
% 	end
% 	if n>2 && n<5
% 		plot([1,tEvent(3)], [rx(id(n),1) rx(id(n),tEvent(3))], '.','markersize',40,'color',[.6 .6 .6]);
% 	end
% 	if n<3 || n>4
% 		plot(tEvent(1), rx(id(n),tEvent(1)), '.','markersize',40,'color',[0 .6 0]);
% 	end
% 	if n>2
% 		plot(tEvent(2), rx(id(n),tEvent(2)), '.','markersize',40,'color',[0 0 .8]);
% 	end
% 	if n>4
% 		plot(nbData, rx(id(n),nbData), '.','markersize',40,'color',[.8 0 0]);
% 	end
% 	m = round(min(rx(id(n),:)) + (max(rx(id(n),:)) - min(rx(id(n),:))) / 2, 1);
% 	axis([1, nbData, m-.6, m+.6]);
% 	set(gca,'xtick',[1,tEvent,nbData],'xticklabel',{},'ytick',sort([m-.6, m+.6, 0]),'fontsize',14);
% 	ylabel(['$x_' num2str(n) '$'],'interpreter','latex','fontsize',22);
% end
% set(gca,'xtick',[1,tEvent,nbData],'xticklabel',{'0','T/4','T/2','3T/4','T'},'fontsize',14);
% ylabel(['$x_' num2str(n) '$'],'interpreter','latex','fontsize',22);
% %dx
% id = [3:4,9:10,15:16];
% for n=1:6
% 	subplot(6,2,2*n); hold on; grid on; box on; title(ttl(n));
% 	plot(rx(id(n),:),'k.');
% 	if n<3
% 		plot([1,tEvent(3)], [rx(id(n),1) rx(id(n),tEvent(3))], '.','markersize',40,'color',[0 0 0]);
% 	end
% 	if n>2 && n<5
% 		plot([1,tEvent(3)], [rx(id(n),1) rx(id(n),tEvent(3))], '.','markersize',40,'color',[.6 .6 .6]);
% 	end
% 	if n<3 || n>4
% 		plot(tEvent(1), rx(id(n),tEvent(1)), '.','markersize',40,'color',[0 .6 0]);
% 	end
% 	if n>2 && n<5
% 		plot(tEvent(2), rx(id(n),tEvent(2)), '.','markersize',40,'color',[0 0 .8]);
% 	end
% 	if n>4
% 		plot(tEvent(2)+1, rx(id(n),tEvent(2)+1), '.','markersize',40,'color',[0 0 .8]);
% 	end
% 	axis([1, nbData, -1.2, 1.2]);
% 	set(gca,'xtick',[1,tEvent,nbData],'xticklabel',{},'ytick',[-1.2, 0, 1.2],'fontsize',14);
% 	ylabel(['$\dot{x}_' num2str(n) '$'],'interpreter','latex','fontsize',22);
% end
% set(gca,'xtick',[1,tEvent,nbData],'xticklabel',{'0','T/4','T/2','3T/4','T'},'fontsize',14);
% ylabel(['$\dot{x}_' num2str(n) '$'],'interpreter','latex','fontsize',22);
% % print('-dpng','graphs/MPC_tennisServe02.png');


%% Additional plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %Timeline plots (u)
% figure('position',[10 10 2200 1200],'color',[1 1 1]);
% ttl = {'Left hand','','Right hand','','Ball (not controlled)',''};
% for n=1:nbVarPos
% 	subplot(6,1,n); hold on; grid on; box on; title(ttl(n));
% 	plot(u(n:nbVarPos:end),'k.');
% 	set(gca,'xtick',[1,tEvent,nbData],'ytick',[0]);
% 	ylabel(['$u_' num2str(n) '$'],'interpreter','latex','fontsize',14);
% end

% %Visualize Q
% figure('position',[1030 10 1000 1000],'color',[1 1 1],'name','Covariances'); hold on; box on; 
% set(gca,'linewidth',2); title('Q','fontsize',14);
% colormap(gca, flipud(gray));
% pcolor(abs(Q));
% set(gca,'xtick',[1,size(Q,1)],'ytick',[1,size(Q,1)]);
% axis square; axis([1 size(Q,1) 1 size(Q,1)]); shading flat;

waitfor(h);
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
