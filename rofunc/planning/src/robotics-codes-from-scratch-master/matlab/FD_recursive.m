%%	  Forward dynamics in recursive form
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Amirreza Razmjoo <amirreza.razmjoo@idiap.ch>, Sylvain Calinon <https://calinon.ch>
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

function FD_recursive

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 500; %Number of datapoints
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %Position space dimension (q1,q2,q3) --> State Space (q1,q2,q3,dq1,dq2,dq3) 
param.nbVarU = 3; %Control space dimension (tau1,tau2,tau3)
param.nbVarF = 3; %Objective function dimension (x1,x2,o)
param.l = [1, 1, 1]; %Robot links lengths
param.m = [1, 1, 1]; %Robot links masses
param.g = 9.81; %gravity norm
param.kv = 1; %Joint damping



%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands
x = [0; 0;0; zeros(param.nbVarX,1)]; %Initial robot pose

%Auxiliary matrices
m = param.m;
l = param.l;
dt = param.dt;
nbDOFs = length(l);
nbData = (size(u,1) / nbDOFs) + 1;
Tm = triu(ones(nbDOFs)) .* repmat(m, nbDOFs, 1);
T=tril(ones(nbDOFs));

% Forward Dynamics
for t=1:nbData-1	
		
	 %Elementwise computation of G,M,C
	for k=1:nbDOFs
		G(k,1) = -sum(m(k:nbDOFs)) * param.g * l(k) * cos(T(k,:)*x(1:nbDOFs,t));
		for i=1:nbDOFs
			S = sum(m(k:nbDOFs) .* heaviside([k:nbDOFs]-i+0.1));
			M(k,i) = l(k) * l(i) * cos(T(k,:)*x(1:nbDOFs,t) - T(i,:)*x(1:nbDOFs,t)) * S;
			C(k,i) = -l(k) * l(i) * sin(T(k,:)*x(1:nbDOFs,t) - T(i,:)*x(1:nbDOFs,t)) * S;
		end
	end
	G= T'*G;
	M = T'*M*T;

	%Update pose
	tau = u((t-1)*nbDOFs+1:t*nbDOFs);
	ddq = (M) \ (tau + G + T'*C * (T*x(nbDOFs+1:2*nbDOFs,t)).^2) - T*x(nbDOFs+1:2*nbDOFs,t) * param.kv; %With external force and joint damping
	x(:,t+1) = x(:,t) + [x(nbDOFs+1:2*nbDOFs,t); ddq] * dt;
end %t
	
%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('position',[10,10,800,800]); hold on; axis off;
h1 = plot(0,0, 'k-','linewidth',2); %link
h2 = plot(0,0, 'k.','markersize',30); %joint/end-effector
axis equal; axis([-1,1,-1,1]*sum(param.l));

% Kinematic simulation
for t=1:param.nbData
	f = fkin0(x(1:param.nbVarX,t), param); %End-effector position
	set(h1, 'XData', f(1,:), 'YData', f(2,:));
	set(h2, 'XData', f(1,:), 'YData', f(2,:));
	drawnow;
	pause(param.dt)
end
end

%%%%%%%%%%%%%%%%%%%%%%
% Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	T = tril(ones(size(x,1)));
	T2 = tril(repmat(param.l, size(x,1), 1));
	f = [T2 * cos(T * x), ...
		 T2 * sin(T * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
% heaviside function (not available in octave)
function y = heaviside(x)
	y = 0;
	if x > 0
		y = 1;
	elseif x == 0
		y = 0.5;
	elseif x < 0
		y = 0;
	end
end