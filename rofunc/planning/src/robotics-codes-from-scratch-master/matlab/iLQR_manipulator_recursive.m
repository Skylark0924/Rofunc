%%    iLQR applied to a planar manipulator for a viapoints task (recursive formulation to find a controller)
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

function iLQR_manipulator_recursive

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step length
param.nbData = 50; %Number of datapoints
param.nbIter = 100; %Number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarU = 3; %Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3; %Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2; 2; 1]; %Robot links lengths
param.sz = [.2, .3]; %Size of objects
param.q = 1E0; %Tracking weighting term
param.r = 1E-6; %Control weighting term

param.Mu = [[2; 1; -pi/6], [3; 2; -pi/3]]; %Viapoints 
for t=1:param.nbPoints
	param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Object orientation matrices
end

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Transfer matrices (for linear system as single integrator)
A = eye(param.nbVarX);
B = eye(param.nbVarX, param.nbVarU) * param.dt;
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));

%Initializations
du = zeros(param.nbVarU, param.nbData-1);
utmp = zeros(param.nbVarU, param.nbData-1);
xtmp = zeros(param.nbVarX, param.nbData);
k = zeros(param.nbVarU, param.nbData-1);
K = zeros(param.nbVarU, param.nbVarX, param.nbData-1);
Luu = repmat(eye(param.nbVarU) * param.r, [1,1,param.nbData]);
Fx = repmat(A, [1,1,param.nbData]);
Fu = repmat(B, [1,1,param.nbData]);

x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose
uref = zeros(param.nbVarU, param.nbData-1); %Initial commands
xref = reshape(Su0 * uref(:) + Sx0 * x0, param.nbVarX, param.nbData); %Initial states

for n=1:param.nbIter
	[f, J] = f_reach(xref(:,tl), param); %Residuals and Jacobians

	Lu = uref * param.r;
	Lx = zeros(param.nbVarX, param.nbData);
	Lxx = zeros(param.nbVarX, param.nbVarX, param.nbData);
	for t=1:param.nbPoints %Tracking objective (sparse)
		Lx(:,tl(t)) = J(:,:,t)' * f(:,t) * param.q;
		Lxx(:,:,tl(t)) = J(:,:,t)' * J(:,:,t) * param.q;
	end

	%Backward pass
	Vx = Lx(:,param.nbData); %Initialization
	Vxx = Lxx(:,:,param.nbData); %Initialization
	for t=param.nbData-1:-1:1
		Qx = Lx(:,t) + Fx(:,:,t)' * Vx;
		Qu = Lu(:,t) + Fu(:,:,t)' * Vx;
		Qxx = Lxx(:,:,t) + Fx(:,:,t)' * Vxx * Fx(:,:,t);
		QuuInv = inv(Luu(:,:,t) + Fu(:,:,t)' * Vxx * Fu(:,:,t)); 
		Qux = Fu(:,:,t)' * Vxx * Fx(:,:,t);
		k(:,t) = -QuuInv * Qu; %Update the feedforward terms
		K(:,:,t) = -QuuInv * Qux; %Update the feedback gains
		Vx = Qx - Qux' * QuuInv * Qu; %Propagate the gradients
		Vxx = Qxx - Qux' * QuuInv * Qux; %Propagate the Hessians
	end
	
	%Forward pass, including step size estimate (backtracking line search method)
	alpha = 1;
	cost0 = norm(f(:))^2 * param.q + norm(uref(:))^2 * param.r; %Cost
	while 1
		xtmp(:,1) = x0;
		for t=1:param.nbData-1
			du(:,t) = alpha * k(:,t) + K(:,:,t) * (xtmp(:,t) - xref(:,t));
			utmp(:,t) = uref(:,t) + du(:,t);
			xtmp(:,t+1) = A * xtmp(:,t) + B * utmp(:,t); %System evolution
		end
		ftmp = f_reach(xtmp(:,tl), param); %Residuals 
		cost = norm(ftmp(:))^2 * param.q + norm(utmp(:))^2 * param.r; %Cost
		
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	uref = uref + du * alpha;
	xref = reshape(Su0 * uref(:) + Sx0 * x0, param.nbVarX, param.nbData);
	
	if norm(du(:) * alpha) < 1E-3
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Simulate reproduction with perturbation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xn = [.5; zeros(param.nbVarX-1, 1)]; %Simulated perturbation on the state
tn = round(param.nbData/3); %Time occurrence of perturbation
x(:,1) = x0;
for t=1:param.nbData-1
	if t==tn
		x(:,t) = x(:,t) + xn; %Simulated perturbation on the state
	end	
	u(:,t) = uref(:,t) + K(:,:,t) * (x(:,t) - xref(:,t));
	x(:,t+1) = A * x(:,t) + B * u(:,t); %System evolution
end


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
plot(ftmp(1,tn-1:tn), ftmp(2,tn-1:tn), 'g.','markersize',20); %Perturbation
plot(ftmp(1,tn-1:tn), ftmp(2,tn-1:tn), 'g-','linewidth',3); %Perturbation
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
	
	J = zeros(param.nbVarF, param.nbVarX, param.nbPoints); 
	for t=1:size(x,2)
		f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered forward kinematics
		
		Jtmp = Jkin(x(:,t), param);
		Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
%		%Bounding boxes (optional)
%		for i=1:2
%			if abs(f(i,t)) < param.sz(i)
%				f(i,t) = 0;
%				Jtmp(i,:) = 0;
%			else
%				f(i,t) = f(i,t) - sign(f(i,t)) * param.sz(i);
%			end
%		end
		
		J(:,:,t) = Jtmp;
	end
end
