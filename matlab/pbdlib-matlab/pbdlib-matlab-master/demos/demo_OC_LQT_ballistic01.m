function demo_OC_LQT_ballistic01
% Batch LQT with augmented state to solve simple ballistic problem
%
% Sylvain Calinon, 2020

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 100; %Number of datapoints
nbVarX = 6; %Dimension of state vector
nbVarU = 2; %Dimension of state vector
dt = 1E-2; %Time step duration
rfactor = 1E-7; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarU) * rfactor; %Control cost matrix

m = 1; %Object mass
f = [0; -m*9.81]; %Gravity vector
x01 = [0; 0]; %Initial position 
xTar = [1; 0]; %Target position 
tRelease = 10; %Time step when ball is released


%% Linear dynamical system parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ac = kron([0, 1, 0; 0, 0, 1; 0, 0, 0], eye(2));
Bc = kron([0; 1; 0], eye(2));
Ad = eye(nbVarX) + Ac * dt; %Parameters for discrete dynamical system
Bd = Bc * dt; %Parameters for discrete dynamical system

%Build Sx and Su matrices for heterogeneous A and B
A = repmat(Ad,[1,1,nbData-1]);
B = repmat(Bd,[1,1,nbData-1]);
% B(:,:,2:end) = 0; %Control command at first time step only
B(:,:,tRelease:end) = 0; %Ball released at time step tRelease

[Su, Sx] = transferMatrices(A, B); 


%% Task setting (sparse reference to reach target at end of the motion)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Q = blkdiag(sparse((nbData-1)*nbVarX,(nbData-1)*nbVarX), diag([1E4, 1E4, 0, 0, 0, 0])); %Sparse precision matrix (at trajectory level)
MuQ = [sparse((nbData-1)*nbVarX,1); [xTar; 0; 0; 0; 0]]; %Sparse reference (at trajectory level)


%% Batch LQT 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [x01; zeros(2,1); f]; 
u = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * x0); 
rx = reshape(Sx*x0+Su*u, nbVarX, nbData); %Reshape data for plotting

uSigma = inv(Su' * Q * Su + R); 
xSigma = Su * uSigma * Su';


%% 2D plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; axis off;
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
set(gca,'LooseInset',[0,0,0,0]);
% %Uncertainty
% for t=1:nbData
% 	plotGMM(rx(1:2,t), xSigma(nbVar*(t-1)+[1,2],nbVar*(t-1)+[1,2]) * 1E-3, [0 0 0], .04); %Ball motion
% end
%Ball motion
plot(rx(1,:), rx(2,:), '-','linewidth',2,'color',[0 0 0]);
h(1) = plot(rx(1,1), rx(2,1), '.','markersize',70,'color',[0 0 0]); %Initial position
h(2) = plot(rx(1,tRelease), rx(2,tRelease), '.','markersize',70,'color',[0 .6 0]); %Release position
h(3) = plot(xTar(1), xTar(2), '.','markersize',70,'color',[.8 0 0]); %Target position
axis equal; %axis([-.05 1.2,-.05 1.1]);
% %Animation
% for t=1:nbData
% 	ha(1) = plot(rx(1,t), rx(2,t), '.','markersize',70,'color',[0 0 0]);
% 	drawnow;
% 	delete(ha);
% end
legend(h,{'Initial point','Release point','Target point'},'fontsize',30,'location','northwest'); 
legend('boxoff');
% print('-dpng','graphs/MPC_ballistic_basic01.png');


% %% 2D sampling plot 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 1200 1200],'color',[1 1 1]); hold on; axis off;
% set(0,'DefaultAxesLooseInset',[0,0,0,0]);
% set(gca,'LooseInset',[0,0,0,0]);
% nbSamples = 50;
% [V,D] = eig(uSigma);
% for n=1:nbSamples
% 	u2 = u + real(V*D.^.5) * randn(size(u)).*1E-2;
% 	rx2 = reshape(Sx*x0+Su*u2, nbVar, nbData); %Reshape data for plotting
% 	patch('xdata',rx2(1,[1:t,t:-1:1]),'ydata',rx2(2,[1:t,t:-1:1]),'linewidth',4,'edgecolor',[0 0 0],'edgealpha',.05); %Ball motion
% 	plot(rx2(1,tRelease), rx2(2,tRelease), '.','markersize',30,'color',[0 .6 0]); %Release of ball
% % 	plot2DArrow(rx(1:2,tEvent(1)), diff(rx(1:2,tEvent(1):tEvent(1)+1),1,2)*12E0, [0 .6 0], 4, .01);
% % 	plot2DArrow(rx(7:8,tEvent(2)), diff(rx(7:8,tEvent(2):tEvent(2)+1),1,2)*12E0, [0 0 .8], 4, .01);
% end
% 
% plot(rx(1,1), rx(2,1), '.','markersize',70,'color',[0 0 0]); %Initial position
% plot(xTar(1), xTar(2), '.','markersize',70,'color',[.8 0 0]); %Target position
% axis equal; axis([-.05 1.2,-.05 1.1]);
% % print('-dpng','graphs/MPC_ballistic02.png');


% %% Additional plot 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 1600 1200]); hold on; axis off;
% colormap(repmat(linspace(1,0,64),3,1)');
% imagesc(Su);
% axis tight; axis equal; axis ij;

pause(10);
close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%
function [Su, Sx] = transferMatrices(A, B)
	[nbVarX, nbVarU, nbData] = size(B);
	nbData = nbData+1;
	Sx = kron(ones(nbData,1), eye(nbVarX)); 
	for t=1:nbData-1
		id1 = (t-1)*nbVarX+1:t*nbVarX;
		id2 = t*nbVarX+1:(t+1)*nbVarX;
		Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);	
	end
	Su = zeros(nbVarX*(nbData-1), nbVarU*(nbData-1));
	for t=1:nbData-1
		id1 = (t-1)*nbVarX+1:t*nbVarX;
		id2 = t*nbVarX+1:(t+1)*nbVarX;
		id3 = (t-1)*nbVarU+1:t*nbVarU;
		Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
		Su(id2,id3) = B(:,:,t);	
	end
end
