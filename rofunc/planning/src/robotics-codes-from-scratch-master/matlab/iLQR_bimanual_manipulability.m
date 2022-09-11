%%    iLQR applied to a planar bimanual robot problem with a cost on tracking a desired 
%%    manipulability ellipsoid at the center of mass (batch formulation)
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

function iLQR_bimanual_manipulability

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E0; %Time step size
param.nbIter = 100; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbData = 10; %Number of datapoints
param.nbVarX = 5; %State space dimension ([q1,q2,q3] for left arm, [q1,q4,q5] for right arm)
param.nbVarU = param.nbVarX; %Control space dimension (dq1,dq2,dq3,dq4,dq5)
param.nbVarF = 4; %Task space dimension ([x1,x2] for left end-effector, [x3,x4] for right end-effector)
param.l = ones(param.nbVarX,1) * 2; %Robot links lengths
param.r = 1E-6; %Control weighting term
param.MuS = [10, 2; 2, 4]; %Desired manipulability ellipsoid

R = speye(param.nbVarU * (param.nbData-1)) * param.r; %Control weight matrix (at trajectory level)
Q = kron(speye(param.nbPoints), diag([0, 0, 0, 0])); %Precision matrix for end-effectors tracking 
Qc = kron(speye(param.nbData), diag([0, 0])); %Precision matrix for continuous CoM tracking 

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1);
x0 = [pi/3; pi/2; pi/3; -pi/3; -pi/4]; %Initial pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter	
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_manipulability(x(:,tl), param); %Residuals and Jacobians
	du = (Su' * J' * J * Su + R) \ (-Su' * J' * f - u * param.r); %Gauss-Newton update  
	
	%Estimate step size with line search method
	alpha = 1;
	cost0 = norm(f)^2 + norm(u)^2 * param.r; %Cost
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_manipulability(xtmp(:,tl), param); %Residuals
		cost = norm(ftmp)^2 + norm(utmp)^2 * param.r; %Cost
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
tl = [1, tl];
al = linspace(-pi, pi, 50);
h = figure('position',[10,10,1200,800],'color',[1,1,1]); hold on; axis off;
fc = fkin_CoM(x, param); %Forward kinematics for center of mass 
%Plot desired manipulability ellipsoid
[V,D] = eig(param.MuS);
msh = V * D.^.5 * [cos(al); sin(al)] * .52 + repmat(fc(:,end), 1, 50);
patch(msh(1,:), msh(2,:), [1 .8 .8],'linewidth',2,'edgecolor',[1 .7 .7]);
%Plot robot manipulability ellipsoid
J = Jkin_CoM(x(:,end), param);
S = J * J';
[V,D] = eig(S);
msh = V * D.^.5 * [cos(al); sin(al)] * .5 + repmat(fc(:,end), 1, 50);
patch(msh(1,:), msh(2,:), [.6 .6 .6],'linewidth',2,'edgecolor',[.4 .4 .4]);
%Plot bimanual robot
ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
%Plot CoM
fc = fkin_CoM(x, param); %Forward kinematics for center of mass 
plot(fc(1,1), fc(2,1), 'o','linewidth',4,'markersize',12,'color',[.5 .5 .5]); %Plot CoM
plot(fc(1,tl(end)), fc(2,tl(end)), 'o','linewidth',4,'markersize',12,'color',[.2 .2 .2]); %Plot CoM
%Plot end-effectors paths
ftmp = fkin(x, param);
plot(ftmp(1,:), ftmp(2,:), 'k-','linewidth',1);
plot(ftmp(3,:), ftmp(4,:), 'k-','linewidth',1);
plot(ftmp(1,tl), ftmp(2,tl), 'k.','markersize',20);
plot(ftmp(3,tl), ftmp(4,tl), 'k.','markersize',20);
axis equal;

waitfor(h);
end 


%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all articulations of a bimanual robot (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(3));
	fl = [L * diag(param.l(1:3)) * cos(L * x(1:3)), ...
	      L * diag(param.l(1:3)) * sin(L * x(1:3))]'; 
	fr = [L * diag(param.l([1,4:5])) * cos(L * x([1,4:5])), ...
	      L * diag(param.l([1,4:5])) * sin(L * x([1,4:5]))]';
	f = [fliplr(fl), zeros(2,1), fr];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for end-effectors of a bimanual robot (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(3));
	f = [param.l(1:3)' * cos(L * x(1:3,:)); ...
	     param.l(1:3)' * sin(L * x(1:3,:)); ...
	     param.l([1,4:5])' * cos(L * x([1,4:5],:)); ...
	     param.l([1,4:5])' * sin(L * x([1,4:5],:))];
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for center of mass of a bimanual robot (in robot coordinate system)
function f = fkin_CoM(x, param)
	L = tril(ones(3));
	f = [param.l(1:3)' * L * cos(L * x(1:3,:)) + param.l([1,4:5])' * L * cos(L * x([1,4:5],:)); ...
	     param.l(1:3)' * L * sin(L * x(1:3,:)) + param.l([1,4:5])' * L * sin(L * x([1,4:5],:))] / 6;
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian for end-effectors of a bimanual robot (in robot coordinate system)
function J = Jkin(x, param)
	L = tril(ones(3));
	J = [-sin(L * x([1,2,3]))' * diag(param.l([1,2,3])) * L; ...
	      cos(L * x([1,2,3]))' * diag(param.l([1,2,3])) * L];
	J(3:4,[1,4:5]) = [-sin(L * x([1,4,5]))' * diag(param.l([1,4,5])) * L; ...
	                   cos(L * x([1,4,5]))' * diag(param.l([1,4,5])) * L];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian for center of mass of a bimanual robot (in robot coordinate system)
function J = Jkin_CoM(x, param)
	L = tril(ones(3));
	Jl = [-sin(L * x(1:3))' * L * diag(param.l(1:3)' * L); ...
	       cos(L * x(1:3))' * L * diag(param.l(1:3)' * L)] / 6;
	Jr = [-sin(L * x([1,4:5]))' * L * diag(param.l([1,4:5])' * L); ...
	       cos(L * x([1,4:5]))' * L * diag(param.l([1,4:5])' * L)] / 6;
	J = [(Jl(:,1) + Jr(:,1)), Jl(:,2:end), Jr(:,2:end)]; %Jacobian for center of mass
end

%%%%%%%%%%%%%%%%%%%%%%
%Residuals f for manipulability tracking
function f = rman(x, param)
	G = param.MuS^-.5;
	for t=1:size(x,2);
%		Jt = Jkin(x(:,t), param); %Jacobian for end-effectors
		Jt = Jkin_CoM(x(:,t), param); %Jacobian for center of mass
		
		St = Jt * Jt'; %Manipulability matrix
		%E = logm(G * St * G);
		[V, D] = eig(G * St * G);
		E = V * diag(log(diag(D))) / V; %Equivalent to logm() function
		%f(:,t) = E(:); %Stacked residuals (including redundant entries)
		%Stacked residuals by taking into account matrix symmetry
		E = tril(E) .* (eye(2) + tril(ones(2), -1) * 2^.5);
		f(:,t) = E(E~=0);
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian for manipulability tracking with numerical computation
function J = Jman_num(x, param)
	e = 1E-6;
	X = repmat(x, [1, param.nbVarX]);
	F1 = rman(X, param);
	F2 = rman(X + eye(param.nbVarX) * e, param);
	J = (F2 - F1) / e;
end

%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for manipulability tracking 
%(c=f'*f is the cost, g=J'*f is the gradient, H=J'*J is the approximated Hessian)
function [f, J] = f_manipulability(x, param)
	f = rman(x, param); %Residuals
	J = [];
	for t=1:size(x,2);
		J = blkdiag(J, Jman_num(x(:,t), param)); %Jacobians
	end
end
