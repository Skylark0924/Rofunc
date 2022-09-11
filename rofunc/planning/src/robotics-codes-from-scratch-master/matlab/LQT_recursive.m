%% Linear quadratic tracking (LQT) applied to a viapoint task (recursive formulation 
%% based on augmented state space to find a controller)
%% 
%% Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%% Written by Sylvain Calinon <https://calinon.ch>
%% 
%% This file is part of RCFS.
%% 
%% RCFS is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License version 3 as
%% published by the Free Software Foundation.
%% 
%% RCFS is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%% GNU General Public License for more details.
%% 
%% You should have received a copy of the GNU General Public License
%% along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function LQT_recursive

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 100; %Number of datapoints
param.nbPoints = 2; %Number of viapoints
param.nbVarU = 2; %Dimension of control commands (here: u1,u2)
param.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarU * param.nbDeriv; %Dimension of state vector
param.nbVarX = param.nbVar + 1; %Augmented state space
param.dt = 1E-2; %Time step duration
param.rfactor = 1E-6; %Control cost in LQR

%Dynamical System for augmented state space
A1d = zeros(param.nbDeriv);
for i=0:param.nbDeriv-1
	A1d = A1d + diag(ones(param.nbDeriv-i,1),i) * param.dt^i / factorial(i); %Discrete 1D
end
B1d = zeros(param.nbDeriv,1); 
for i=1:param.nbDeriv
	B1d(param.nbDeriv-i+1) = param.dt^i / factorial(i); %Discrete 1D
end
A0 = kron(A1d, eye(param.nbVarU)); %Discrete nD
B0 = kron(B1d, eye(param.nbVarU)); %Discrete nD
A = [A0, zeros(param.nbVar,1); zeros(1,param.nbVar), 1]; %Augmented A
B = [B0; zeros(1,param.nbVarU)]; %Augmented B

%Sparse reference with a set of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
%Mu = [rand(param.nbVarU, param.nbPoints); zeros(param.nbVar-param.nbVarU, param.nbPoints)];
Mu = [2 3; 5 1; zeros(param.nbVar-param.nbVarU, param.nbPoints)];

%%Definition of augmented precision matrix Qa based on covariance matrix Sigma0
%Sigma0 = diag([ones(1,param.nbVarU)*1E-5, ones(1,param.nbVar-param.nbVarU)*1E3]); %Covariance matrix
%for i=1:param.nbPoints
%	Sigma(:,:,i) = [Sigma0 + Mu(:,i) * Mu(:,i)', Mu(:,i); Mu(:,i)', 1]; %Embedding of Mu in Sigma
%end
%Qa = zeros(param.nbVarX, param.nbVarX, param.nbData);
%for i=1:param.nbPoints
%	Qa(:,:,tl(i)) = inv(Sigma(:,:,i));
%end

%Definition of augmented precision matrix Qa based on standard precision matrix Q0
Q0 = diag([ones(1,param.nbVarU)*1E0, zeros(1,param.nbVar-param.nbVarU)]); %Precision matrix
Qa = zeros(param.nbVarX, param.nbVarX, param.nbData);
for i=1:param.nbPoints
	Qa(:,:,tl(i)) = [eye(param.nbVar), zeros(param.nbVar,1); -Mu(:,i)', 1] * blkdiag(Q0, 1) * ...
	                [eye(param.nbVar), -Mu(:,i); zeros(1,param.nbVar), 1]; %Augmented precision matrix
end

Ra = eye(param.nbVarU) * param.rfactor; %Control cost matrix


%% LQR with recursive computation and augmented state space (including perturbation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%xn = [randn(param.nbVarU, 1) * 5E0; randn(param.nbVar-param.nbVarU, 1) * 0; 0]; %Simulated noise on state
xn = [-1; -.1; zeros(param.nbVarX-param.nbVarU, 1)]; %Simulated noise on state

P = zeros(param.nbVarX, param.nbVarX, param.nbData);
P(:,:,end) = Qa(:,:,end);
for t=param.nbData-1:-1:1
	P(:,:,t) = Qa(:,:,t) - A' * (P(:,:,t+1) * B / (B' * P(:,:,t+1) * B + Ra) * B' * P(:,:,t+1) - P(:,:,t+1)) * A;
end
%Reproduction with feedback controller on augmented state
for n=1:2
	x = [zeros(param.nbVar,1); 1]; %Augmented state space
	for t=1:param.nbData
		Ka = (B' * P(:,:,t) * B + Ra) \ B' * P(:,:,t) * A; %Feedback gain
		u = -Ka * x; %Feedback control on augmented state (resulting in feedback and feedforward terms on state)
		x = A * x + B * u; %Update of state vector
		if t==25 && n==2
			x = x + xn; %Simulated noise on the state
		end
		r(n).x(:,t) = x; %Log data
	end
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 800 800],'color',[1 1 1]); axis off; hold on; 
plot(r(1).x(1,1), r(1).x(2,1), 'k.','markersize',30);
plot(r(1).x(1,:), r(1).x(2,:), 'k:','linewidth',2);
plot(r(2).x(1,:), r(2).x(2,:), 'k-','linewidth',2);
plot(r(2).x(1,24:25), r(2).x(2,24:25), 'g.','markersize',20);
plot(Mu(1,:), Mu(2,:), 'r.','markersize',40);
axis equal;

waitfor(h);
