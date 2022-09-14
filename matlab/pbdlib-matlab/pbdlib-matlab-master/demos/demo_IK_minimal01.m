function demo_IK_minimal01
% Forward and inverse kinematics for a planar robot with minimal computation
%
% Sylvain Calinon, 2021

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2; %Time step
nbData = 100; %Number of datapoints
model.nbVarX = 3; %Number of articulations
model.l = [2; 2; 1]; %Length of each link
x(:,1) = [3*pi/4; -pi/2; -pi/4]; %Initial pose
%x(:,1) = ones(model.nbVarX,1) * pi / model.nbVarX; %Initial pose


%% IK 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fh = [3; 1]; 
for t=1:nbData-1
	f(:,t) = fkin(x(:,t), model); %FK for a planar robot
	J = Jkin(x(:,t), model); %Jacobian
	dx = pinv(J) * (fh - f(:,t)) * 1E1;
	x(:,t+1) = x(:,t) + dx * dt;	
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[20,10,800,650],'color',[1 1 1]); hold on; axis off;
for t=floor(linspace(1,nbData,2))
	colTmp = [.7,.7,.7] - [.7,.7,.7] * t/nbData;
	plotArm(x(:,t), model.l, [0;0;-2+t/nbData], .2, colTmp);
end
plot(fh(1), fh(2), '.','markersize',40,'color',[.8 0 0]);
plot(f(1,:), f(2,:), '.','markersize',20,'color',[0 .6 0]);
axis equal; axis tight;

%print('-dpng','graphs/demoIK_minimal01.png');
%pause;
%close all;
waitfor(h);
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics
function f = fkin(x, model)
	L = tril(ones(model.nbVarX));
	f = [model.l' * cos(L * x); ...
		 model.l' * sin(L * x)];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation
function J = Jkin(x, model)
	L = tril(ones(model.nbVarX));
	J = [-sin(L * x)' * diag(model.l) * L; ...
		  cos(L * x)' * diag(model.l) * L];
end
