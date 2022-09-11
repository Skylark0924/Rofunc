%%    iLQR applied to a 2D point-mass system reaching a target while avoiding obstacles defined by ellipses
%%    (avoidance cost in the form (1-e^2)'*Q*(1-e^2) = f'*f with f=U'*(1-e^2) and Q=U*U', which is a correct Gauss-Newton assumption)
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

function iLQR_obstacle

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 101; %Number of datapoints
param.nbIter = 300; %Maximum number of iterations for iLQR
param.nbPoints = 1; %Number of viapoints
param.nbObstacles = 2; %Number of obstacles
param.nbVarX = 2; %State space dimension (x1,x2)
param.nbVarU = 2; %Control space dimension (dx1,dx2)
param.sz2 = [.4, .6]; %Size of obstacles
param.q = 1E2; %Tracking weight term
param.q2 = 1E0; %Obstacle avoidance weight term
param.r = 1E-3; %Control weight term

param.Mu = [3; 3; pi/6]; %Viapoints (x1,x2,o)

% param.Mu2 = [rand(2,param.nbObstacles)*3; rand(1,param.nbObstacles)*pi];
param.Mu2 = [[1; 0.6; pi/4], [2.0; 2.5; -pi/6]]; %Obstacles (x1,x2,o)
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
u = zeros(param.nbVarU*(param.nbData-1), 1);
x0 = zeros(param.nbVarX, 1);

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
	
	if norm(du * alpha) < 5E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.u = reshape(u, param.nbVarU, param.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
al = linspace(-pi, pi, 50);

figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
for t=1:param.nbPoints
	plot(param.Mu(1,t), param.Mu(2,t), '.','markersize',40,'color',[.8 0 0]);
end
for t=1:param.nbObstacles
	msh = param.R2(:,:,t) * [cos(al); sin(al)] + repmat(param.Mu2(1:2,t), 1, 50);
	patch(msh(1,:), msh(2,:), [.8 .8 .8],'linewidth',2,'edgecolor',[.4 .4 .4]);
end
plot(r.x(1,:), r.x(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.x(1,1:5:end), r.x(2,1:5:end), '.','markersize',15,'color',[0 0 0]);
plot(r.x(1,[1,tl(1:end-1)]), r.x(2,[1,tl(1:end-1)]), '.','markersize',30,'color',[0 0 0]);
axis equal; 

pause(10);
end 

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
	f = x - param.Mu(1:2,:); %Error by ignoring manifold
	J = eye(param.nbVarX * size(x,2));
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for an obstacle avoidance task
function [f, J, id, idt] = f_avoid(x, param)
	f=[]; J=[]; id=[]; idt=[];
	for t=1:size(x,2)
		for i=1:param.nbObstacles			
			e = param.U2(:,:,i)' * (x(:,t) - param.Mu2(1:2,i));
			ftmp = 1 - e' * e; %quadratic form
			%Bounding boxes 
			if ftmp > 0
				f = [f; ftmp];
				Jtmp = -e' * param.U2(:,:,i)'; %quadratic form
				J = blkdiag(J, Jtmp);
				id = [id, (t-1) * param.nbVarU + [1:param.nbVarU]];
				idt = [idt, t];
			end
		end
	end
end
