%%	  iLQR applied to a planar manipulator task with obstacles avoidance
%% 	  (viapoints task with position+orientation including bounding boxes on f(x))
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Sylvain Calinon <https://calinon.ch>
%%
%%    This file is part of RCFS.
%%
%%    RCFS is free software: you can redistribute it and/or modify
%%    it under the terms of the GNU General Public License version 3 as
%%    published by the Free Software Foundation.
%%
%%    RCFS is distributed in the hope that it will be useful,
%%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%%    GNU General Public License for more details.
%%
%%    You should have received a copy of the GNU General Public License
%%    along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function iLQR_manipulator_obstacle

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 50; %Number of datapoints
param.nbIter = 50; %Number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbObstacles = 2; %Number of obstacles
param.nbVarX = 3; %State space dimension (q1,q2,q3)
param.nbVarU = 3; %Control space dimension (dq1,dq2,dq3)
param.nbVarF = 3; %Objective function dimension (x1,x2,o)
param.l = [2, 2, 1]; %Robot links lengths
param.sz2 = [.5, .8]; %Size of obstacles
param.q = 1E2; %Tracking weight term
param.q2 = 1E0; %Obstacle avoidance weight term
param.r = 1E-3; %Control weight term

param.Mu = [[3; -1; 0]]; %Viapoints (x1,x2,o)

param.Mu2 = [[2.8; 2.0; pi/4], [3.5; .5; -pi/6]]; %Obstacles (x1,x2,o)
% param.Mu2 = [[-1.2; 1.5; pi/4], [-0.5; 2.5; -pi/6]]; %Obstacles (x1,x2,o)
for t=1:param.nbObstacles
	param.A2(:,:,t) = [cos(param.Mu2(3,t)), -sin(param.Mu2(3,t)); sin(param.Mu2(3,t)), cos(param.Mu2(3,t))]; %Orientation
	param.R2(:,:,t) = param.A2(:,:,t) * diag(param.sz2); %Eigendecomposition of covariance matrix S2=R2*R2'
	param.U2(:,:,t) = param.A2(:,:,t) * diag(param.sz2.^-1); %Eigendecomposition of precision matrix Q2=U2*U2'
end

R = speye((param.nbData-1) * param.nbVarU) * param.r; %Control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands
x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), param); %Tracking objective
	
	[f2, J2, id2] = f_avoid(x, param); %Avoidance objective
	Su2 = Su0(id2,:);
	
	du = (Su' * J' * J * Su * param.q + Su2' * J2' * J2 * Su2 * param.q2 + R) \ ...
			(-Su' * J' * f(:) * param.q - Su2' * J2' * f2(:) * param.q2 - u * param.r); %Gradient
		
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = norm(f(:))^2 * param.q + norm(f2(:))^2 * param.q2 + norm(u)^2 * param.r;
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_reach(xtmp(:,tl), param); %Tracking objective
		ftmp2 = f_avoid(xtmp, param); %Avoidance objective
		cost = norm(ftmp(:))^2 * param.q + norm(ftmp2(:))^2 * param.q2 + norm(utmp)^2 * param.r;
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha;
	
	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.f = [];
for j=param.nbVarU:-1:1
	r.f = [r.f; fkin(x(1:j,:), param.l(1:j))];
end
r.u = reshape(u, param.nbVarU, param.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
ftmp = fkin0(x(:,1), param.l);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(end)), param.l);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
al = linspace(-pi, pi, 50);
for t=1:param.nbObstacles
	msh = param.R2(:,:,t) * [cos(al); sin(al)] + repmat(param.Mu2(1:2,t), 1, 50);
	patch(msh(1,:), msh(2,:), [.8 .8 .8],'linewidth',2,'edgecolor',[.4 .4 .4]);
end
for j=param.nbVarU:-1:1
	plot(r.f((j-1)*param.nbVarF+1,:), r.f((j-1)*param.nbVarF+2,:), '-','linewidth',1,'color',[.2 .2 .2]);
	plot(r.f((j-1)*param.nbVarF+1,:), r.f((j-1)*param.nbVarF+2,:), '.','markersize',10,'color',[.2 .2 .2]);
end
plot(r.f(1,:), r.f(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.f(1,:), r.f(2,:), '.','markersize',10,'color',[0 0 0]);
plot(r.f(1,[1,tl]), r.f(2,[1,tl]), '.','markersize',30,'color',[0 0 0]);
axis equal; 

pause(10);
end 


%%%%%%%%%%%%%%%%%%%%%%
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
	e(1:2,:) = f(1:2,:) - f0(1:2,:);
	e(3,:) = imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'));
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for end-effector (in robot coordinate system)
function f = fkin(x, L)
	T = tril(ones(length(L)));
	f = [L * cos(T * x); ...
		L * sin(T * x); ...
		mod(sum(x,1)+pi, 2*pi) - pi]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, L)
	T = tril(ones(size(x,1)));
	T2 = T * diag(L);
	f = [T2 * cos(T * x), ...
		T2 * sin(T * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = jkin(x, L)
	T = tril(ones(length(L)));
	J = [-sin(T * x)' * diag(L) * T; ...
		cos(T * x)' * diag(L) * T; ...
		ones(1, length(L))]; %x1,x2,o
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
% 	f = fkin(x, param) - param.Mu; %Error by ignoring manifold
	f = logmap(fkin(x, param.l), param.Mu); %Error by considering manifold
	J = []; 
	for t=1:size(x,2)		
		Jtmp = jkin(x(:,t), param.l);
		J = blkdiag(J, Jtmp);
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for an obstacle avoidance task
function [f, J, id] = f_avoid(x, param)
	f=[]; J=[]; id=[];
	for i=1:param.nbObstacles
		for j=1:param.nbVarU
			for t=1:param.nbData
				xee = fkin(x(1:j,t), param.l(1:j));
				e = param.U2(:,:,i)' * (xee(1:2) - param.Mu2(1:2,i));
				ftmp = 1 - e' * e; %quadratic form
				%Bounding boxes 
				if ftmp > 0
					f = [f; ftmp];
					Jrob = [jkin(x(1:j,t), param.l(1:j)), zeros(param.nbVarF, param.nbVarU-j)];
					Jtmp = -e' * param.U2(:,:,i)' * Jrob(1:2,:); %quadratic form
					J = blkdiag(J, Jtmp);
					id = [id, (t-1) * param.nbVarU + [1:param.nbVarU]'];
				end
			end
		end
	end
end
