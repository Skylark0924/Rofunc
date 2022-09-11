%%    iLQR applied to a planar manipulator for a tracking problem involving the center of mass (CoM) and the end-effector (batch formulation)
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

function iLQR_manipulator_CoM

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-1; %Time step size
param.nbData = 10; %Number of datapoints
param.nbIter = 50; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbVarX = 5; %State space dimension (q1,q2,q3,q4,q5)
param.nbVarU = 5; %Control space dimension (dq1,dq2,dq3,dq4,dq5)
param.nbVarF = 2; %Task space dimension (x1,x2)
param.l = ones(param.nbVarX,1) * 2; %Links lengths
param.Mu = [3.5; 4]; %Target point for end-effector
param.MuCoM = [.4; 0]; %Target point for center of mass
param.szCoM = .6; %CoM allowed width
param.r = 1E-5; %Control weight term

R = speye(param.nbVarU * (param.nbData-1)) * param.r; %Control weight matrix (at trajectory level)
Q = speye(param.nbVarF * param.nbPoints) * 1E0; %Precision matrix for end-effector
Qc = kron(speye(param.nbData), diag([1E0, 0])); %Precision matrix for CoM (by considering only horizontal CoM location)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1);
a = .7;
x0 = [pi/2-a; 2*a; -a; pi-pi/4; 3*pi/4]; %Initial pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';
Su = Su0(idx,:);

for n=1:param.nbIter	
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), param); %Forward kinematics and Jacobian for end-effector
	[fc, Jc] = f_reach_CoM(x, param); %Forward kinematics and Jacobian for center of mass
	du = (Su' * J' * Q * J * Su + Su0' * Jc' * Qc * Jc * Su0 + R) \ (-Su' * J' * Q * f - Su0' * Jc' * Qc * fc - u * param.r); %Gradient 

	%Estimate step size with line search method
	alpha = 1;
	cost0 = f' * Q * f + fc' * Qc * fc + norm(u)^2 * param.r; %for end-effector and CoM
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_reach(xtmp(:,tl), param);
		fctmp = f_reach_CoM(xtmp, param);
		cost = ftmp' * Q * ftmp + fctmp' * Qc * fctmp + norm(utmp)^2 * param.r; %for end-effector and CoM
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha; %Update control by following gradient

	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;

plot([-1,3], [0,0], '-','linewidth',2,'color',[.2 .2 .2]); %Plot ground
msh = diag([param.szCoM, 3.5]) * [-1 -1 1 1 -1; -1 1 1 -1 -1] + repmat(param.MuCoM + [0; 3.5], 1, 5); 
patch(msh(1,:), msh(2,:), [.8 0 0],'edgecolor','none','facealpha',.1); %Plot CoM bounding box
ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
fc = fkin_CoM(x, param); 
plot(fc(1,1), fc(2,1), 'o','linewidth',4,'markersize',8,'color',[.5 .5 .5]); %Plot CoM
plot(fc(1,tl(end)), fc(2,tl(end)), 'o','linewidth',4,'markersize',8,'color',[.2 .2 .2]); %Plot CoM
plot(param.Mu(1,:), param.Mu(2,:), 'r.','markersize',30); %Plot end-effector target
axis equal; 

waitfor(h);
end 


%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * cos(L * x); ...
	     param.l' * sin(L * x)];
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
	      cos(L * x)' * diag(param.l) * L]; %x1,x2
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for end-effector
function [f, J] = f_reach(x, param)
	f = fkin(x, param) - param.Mu; 
	f = f(:);
	
	J = []; 
	for t=1:size(x,2)
		J = blkdiag(J, Jkin(x(:,t), param));
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for center of mass (in robot coordinate system, with mass located at the joints)
function f = fkin_CoM(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * L * cos(L * x); ...
	     param.l' * L * sin(L * x)] / param.nbVarX;
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for center of mass
function [f, J] = f_reach_CoM(x, param)
	f = fkin_CoM(x, param) - param.MuCoM; 
	
	J = []; 
	L = tril(ones(size(x,1)));
	for t=1:size(x,2)
		Jtmp = [-sin(L * x(:,t))' * L * diag(param.l' * L); ...
		         cos(L * x(:,t))' * L * diag(param.l' * L)] / param.nbVarX;
				 
		%Bounding boxes (optional)
		for i=1:1
			if abs(f(i,t)) < param.szCoM
				f(i,t) = 0;
				Jtmp(i,:) = 0;
			else
				f(i,t) = f(i,t) - sign(f(i,t)) * param.szCoM;
			end
		end
	
		J = blkdiag(J, Jtmp);
	end
	f = f(:);
end
