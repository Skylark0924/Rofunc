%%    iLQR applied to a planar manipulator for a viapoints task (batch formulation)
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

function iLQR_manipulator

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step length
param.nbData = 50; %Number of datapoints
param.nbIter = 300; %Number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarU = 3; %Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3; %Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2; 2; 1]; %Robot links lengths
param.sz = [.2, .3]; %Size of objects
param.r = 1E-6; %Control weight term

% param.Mu = [[2; 1; -pi/2], [3; 1; -pi/2]]; %Viapoints
param.Mu = [[2; 1; -pi/6], [3; 2; -pi/3]]; %Viapoints 
% param.Mu = [3; 0; -pi/2]; %Viapoints 
for t=1:param.nbPoints
	param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Object orientation matrices
end

R = speye((param.nbData-1)*param.nbVarU) * param.r; %Control weight matrix (at trajectory level)
Q = speye(param.nbVarF * param.nbPoints) * 1E0; %Precision matrix
% Q = kron(eye(param.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)

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
	[f, J] = f_reach(x(:,tl), param); %Residuals and Jacobians
	du = (Su' * J' * Q * J * Su + R) \ (-Su' * J' * Q * f(:) - u * param.r); %Gauss-Newton update
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = f(:)' * Q * f(:) + norm(u)^2 * param.r; %Cost
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
		ftmp = f_reach(xtmp(:,tl), param); %Residuals
		cost = ftmp(:)' * Q * ftmp(:) + norm(utmp)^2 * param.r; %Cost
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


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
msh0 = diag(param.sz) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:param.nbPoints
	msh(:,:,t) = param.A(:,:,t) * msh0 + repmat(param.Mu(1:2,t), 1, size(msh0,2));
end

h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
colMat = lines(param.nbPoints);

ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);

ftmp = fkin0(x(:,tl(1)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.6 .6 .6]);

ftmp = fkin0(x(:,tl(2)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);

for t=1:param.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:));
end

ftmp = fkin(x, param); 
plot(ftmp(1,:), ftmp(2,:), 'k-','linewidth',2);
plot(ftmp(1,[1,tl]), ftmp(2,[1,tl]), 'k.','markersize',20);
axis equal; 

waitfor(h);
end

%%%%%%%%%%%%%%%%%%%%%%
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
	e = [f(1:2,:) - f0(1:2,:); ...
	     imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'))'];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for end-effector (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * cos(L * x); ...
	     param.l' * sin(L * x); ...
	     mod(sum(x,1)+pi, 2*pi) - pi]; %f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(size(x,1)));
	f = [L * diag(param.l) * cos(L * x), ...
	     L * diag(param.l) * sin(L * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = Jkin(x, param)
	L = tril(ones(size(x,1)));
	J = [-sin(L * x)' * diag(param.l) * L; ...
	      cos(L * x)' * diag(param.l) * L; ...
	      ones(1, size(x,1))]; 
end

%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
% 	f = fkin(x, param) - param.Mu; %Residuals by ignoring manifold
	f = logmap(fkin(x, param), param.Mu); %Residuals by considering manifold
	
	J = []; 
	for t=1:size(x,2)
		f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered forward kinematics
		
		Jtmp = Jkin(x(:,t), param);
		Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
		%Bounding boxes (optional)
		for i=1:2
			if abs(f(i,t)) < param.sz(i)
				f(i,t) = 0;
				Jtmp(i,:) = 0;
			else
				f(i,t) = f(i,t) - sign(f(i,t)) * param.sz(i);
			end
		end
		
		J = blkdiag(J, Jtmp);
	end
end
