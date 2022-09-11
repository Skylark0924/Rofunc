%%    iLQR applied to a 2D point-mass system with the objective of constantly  
%%    maintaining a desired distance to an object 
%%
%%    Copyright (c) 2022 Idiap Research Institute, https://www.idiap.ch/
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

function iLQR_distMaintenance

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 50; %Number of datapoints
param.nbIter = 100; %Maximum number of iterations for iLQR
param.nbVarX = 2; %State space dimension (x1,x2)
param.nbVarU = 2; %Control space dimension (dx1,dx2)
param.Mu = [1.0; 0.3]; %Object location
param.dist = .4; %Distance to maintain
param.q = 1E0; %Distance maintenance weight term
param.r = 1E-3; %Control weight term

R = speye((param.nbData-1) * param.nbVarU) * param.r; %Control weight matrix (at trajectory level)


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands
x0 = zeros(param.nbVarX, 1); %Initial state

%Transfer matrices (for linear system as single integrator)
Su = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx = kron(ones(param.nbData,1), eye(param.nbVarX));

for n=1:param.nbIter
	x = reshape(Su * u + Sx * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_dist(x, param); %Residuals and Jacobians (distance maintenance objective)
	
	du = (Su' * J' * J * Su * param.q + R) \ (-Su' * J' * f(:) * param.q - u * param.r); %Gauss-Newton update
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = norm(f(:))^2 * param.q + norm(u)^2 * param.r; %Cost
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su * utmp + Sx * x0, param.nbVarX, param.nbData);
		ftmp = f_dist(xtmp, param); %Residuals (avoidance objective)
		cost = norm(ftmp(:))^2 * param.q + norm(utmp)^2 * param.r; %Cost
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha;
	
	if norm(du * alpha) < 5E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
al = linspace(-pi, pi, 50);
h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
msh = param.dist * [cos(al); sin(al)] + repmat(param.Mu(1:2), 1, 50);
patch(msh(1,:), msh(2,:), [1 .8 .8],'linewidth',2,'edgecolor',[.8 .4 .4]);
plot(param.Mu(1), param.Mu(2), '.','markersize',25,'color',[.8 0 0]);
plot(x(1,:), x(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(x(1,[1,end]), x(2,[1,end]), '.','markersize',25,'color',[0 0 0]);
axis equal; 

waitfor(h);
end 

%%%%%%%%%%%%%%%%%%%%%%
%Residuals f and Jacobians J for maintaining a desired distance to an object (fast version with computation in matrix form)
function [f, J] = f_dist(x, param)
	e = x - repmat(param.Mu(1:2), 1, param.nbData);
	f = 1 - sum(e.^2, 1)' / param.dist^2; %Residuals
	Jtmp = repmat(-e'/param.dist^2, 1, param.nbData); 
	J = Jtmp .* kron(eye(param.nbData), ones(1,param.nbVarU)); %Jacobians 
end

%%%%%%%%%%%%%%%%%%%%%%%%
%%%Residuals f and Jacobians J for obstacle avoidance (slow version with computation using a loop over time steps)
%function [f, J] = f_dist(x, param)
%	f=[]; J=[]; 
%	for t=1:size(x,2)
%		e = x(:,t) - param.Mu(1:2);
%		ftmp = 1 - e' * e / param.dist^2; 
%		f = [f; ftmp]; %Residuals
%		Jtmp = -e' / param.dist^2; 
%		J = blkdiag(J, Jtmp); %Jacobians
%	end
%end
