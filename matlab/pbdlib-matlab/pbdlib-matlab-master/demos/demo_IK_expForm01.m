function demo_IK_expForm01
% Forward and inverse kinematics for a planar robot with exp(1i*q) formulation and analytical Jacobian 
%
% Sylvain Calinon, 2020

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 1E-2; %Time step
nbData = 100;


%% Robot parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbDOFs = 3; %Number of articulations
model.l = [2, 2, 1]; %Length of each link
% q(:,1) = [3*pi/4; -pi/2; -pi/4]; %Initial pose
q(:,1) = [pi-3*pi/4; pi/2; pi/4]; %Initial pose
% q = rand(model.nbDOFs,1);
x(:,1) = fkine(q(:,1), model);


%% IK 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xh = [2; 2; pi/4]; %x1,x2,o
% xh = [x(1:2,1); pi/4]; %x1,x2,o
for t=1:nbData-1
	J = jacob0(q(:,t), model); %Jacobian
	
% 	dxh = 5E0 * (xh - x(:,t)); %Error by ignoring manifold
	dxh = 5E0 * [xh(1:2) - x(1:2,t); -logmap(x(3,t), xh(3))]; %Error by considering manifold
	
	dq = pinv(J) * dxh;
	q(:,t+1) = q(:,t) + dq * dt;
	x(:,t+1) = fkine(q(:,t+1), model); %FK for a planar robot
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[20,10,800,650],'color',[1 1 1]); hold on; axis off;
for t=floor(linspace(1,nbData,20))
	colTmp = [.7,.7,.7] - [.7,.7,.7] * t/nbData;
	plotArm(q(:,t), model.l, [0;0;-2+t/nbData], .2, colTmp);
% 	plot(x(1,t), x(2,t), '.','markersize',50,'color',[.8 0 0]);
end
plot(xh(1), xh(2), '.','markersize',40,'color',[.8 0 0]);
plot2DArrow(xh(1:2), [cos(xh(3)); sin(xh(3))], [.8 0 0], 5);
plot(x(1,:), x(2,:), '.','markersize',20,'color',[0 .6 0]);
axis equal; axis tight;

%print('-dpng','graphs/demoIK_expForm01.png');
waitfor(h);
end

function u = logmap(x, x0)
	u = imag(log(exp(x0*1i)' * exp(x*1i)));
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics
function x = fkine(q, model)
% 	x = 0;
% 	for n=1:model.nbDOFs
% 		x = x + model.l(n) * exp(1i * sum(q(1:n)));
% 	end

% 	x = model.l * ones(1,model.nbDOFs) * exp(1i * tril(ones(model.nbDOFs)) * q); %Computation in matrix form
	x = model.l * exp(1i * tril(ones(model.nbDOFs)) * q); %Computation in matrix form
	x = [real(x); imag(x); (mod(sum(q)+pi, 2*pi) - pi)]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation
function J = jacob0(q, model)
% 	J = zeros(1,model.nbDOFs);
% 	J2 = zeros(1,model.nbDOFs);
% 	for n=1:model.nbDOFs
% 		for m=n:model.nbDOFs
% 			J(n) = J(n) + model.l(n) * 1i * exp(1i * sum(q(1:m))); 
% 		end
% 		M = [zeros(1,n-1), ones(1,model.nbDOFs-n+1)]
% 		J2(n) = model.l(n) * 1i * M * exp(1i * tril(ones(model.nbDOFs)) * q);
% 	end
% 	J = [real(J); imag(J)]

% 	J = model.l * 1i * exp(1i * tril(ones(model.nbDOFs)) * q)' * tril(ones(model.nbDOFs)); %Computation in matrix form
	J = 1i * exp(1i * tril(ones(model.nbDOFs)) * q).' * diag(model.l) * tril(ones(model.nbDOFs)); %Computation in matrix form
	J = [real(J); imag(J); ones(1, model.nbDOFs)]; %x1,x2,o
end
