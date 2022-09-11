%%    Linear quadratic tracking (LQT) applied to a multi-agent system with a recursive formulation 
%%    based on least squares and an augmented state space, by using a precision matrix with nonzero 
%%    offdiagonal elements to find a controller in which the two agents coordinate their movements to 
%%    find an optimal meeting point
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

function LQT_recursive_LS_multiAgents

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 120; %Number of datapoints
param.nbPoints = 1; %Number of viapoints
param.nbAgents = 2; %Number of agents
param.nbVarU = 2 * param.nbAgents; %Dimension of control commands for two agents (here: u1,u2, u3,u4)
param.nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarU * param.nbDeriv; %Dimension of state vector
param.nbVarX = param.nbVar + 1; %Augmented state space
param.dt = 1E-1; %Time step duration
param.r = 1E-6; %Control cost in LQR

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

%Build Sx and Su transfer matrices for augmented state space
Sx = kron(ones(param.nbData,1), speye(param.nbVarX));
Su = sparse(param.nbVarX * param.nbData, param.nbVarU * (param.nbData-1));
M = B;
for t=2:param.nbData
	id1 = (t-1)*param.nbVarX+1:param.nbData*param.nbVarX;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (t-1)*param.nbVarX+1:t*param.nbVarX; 
	id2 = 1:(t-1)*param.nbVarU;
	Su(id1,id2) = M;
	M = [A*M(:,1:param.nbVarU), M]; 
end

%Sparse reference with a set of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
Mu = [2; 3; 5; 1; zeros(param.nbVar-param.nbVarU, 1)];

Ra = speye((param.nbData-1)*param.nbVarU) * param.r; %Control cost matrix

%Definition of augmented precision matrix Qa based on standard precision matrix Q0
Q0 = diag([ones(1,param.nbVarU)*1E0, zeros(1,param.nbVar-param.nbVarU)]); %Precision matrix
Qa = zeros(param.nbVarX * param.nbData);
for i=1:param.nbPoints
	id = [1:param.nbVarX] + (tl(i)-1) * param.nbVarX;
	Qa(id,id) = [eye(param.nbVar), zeros(param.nbVar,1); -Mu(:,i)', 1] * blkdiag(Q0, 1) * ...
	            [eye(param.nbVar), -Mu(:,i); zeros(1,param.nbVar), 1]; %Augmented precision matrix
end

%Request the two agents to meet in the middle of the motion (e.g., for a handover task)
tlm = round(param.nbData / 2);
idm = [1:param.nbVarX] + (tlm-1) * param.nbVarX;
Q0 = blkdiag([eye(2), -eye(2); -eye(2), eye(2)], zeros(param.nbVar-param.nbVarU)); %Precision matrix with offdiagonal elements
Qa(idm,idm) = blkdiag(Q0, 1); %Augmented precision matrix

%%Request the two agents to find a location to reach at different time steps (e.g., to drop and pick-up an object)
%tlm = [round(param.nbData / 3), round(2 * param.nbData / 3)]; %Dropping and picking times
%idm = [[1:param.nbVarX] + (tlm(1)-1) * param.nbVarX, [1:param.nbVarX] + (tlm(2)-1) * param.nbVarX];
%isub = [1, 2, param.nbVarX+3, param.nbVarX+4]; %[1,2] for Agent 1, [3,4] for Agent 2
%Qa(idm(isub),idm(isub)) = [eye(2), -eye(2); -eye(2), eye(2)]; %Constraint
%isub = [param.nbVarX, 2*param.nbVarX]; %Last elements of the corresponding augmented states
%Qa(idm(isub),idm(isub)) = eye(2); %Augmented state definition (does not influence the result)


%% LQR with least squares approach on augmented state space (including perturbation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xn = [-1; 1; 0; 0; zeros(param.nbVarX-param.nbVarU, 1)]; %Simulated noise on state

tn = round(param.nbData / 4);

F = (Su' * Qa * Su + Ra) \ Su' * Qa * Sx; 

Ka(:,:,1) = F(1:param.nbVarU,:);
P = eye(param.nbVarX);
for t=2:param.nbData-1
	id = (t-1)*param.nbVarU + [1:param.nbVarU];
	P = P / (A - B * Ka(:,:,t-1));
	Ka(:,:,t) = F(id,:) * P; %Feedback gain on augmented state
end
%Reproduction with feedback controller on augmented state
for n=1:2
	x = [-1; 0; 1; 0; zeros(param.nbVar-param.nbVarU,1); 1]; %Augmented state space
	for t=1:param.nbData-1
		u = -Ka(:,:,t) * x; %Feedback control on augmented state (resulting in feedback and feedforward terms on state)
		x = A * x + B * u; %Update of state vector
		if t==tn && n==2
			x = x + xn; %Simulated noise on the state
		end
		r(n).x(:,t) = x; %Log data
	end
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 800 800],'color',[1 1 1]); axis off; hold on; 
%Plot Agent 1
plot(r(1).x(1,1), r(1).x(2,1), 'k.','markersize',30); %Initial point
plot(r(1).x(1,:), r(1).x(2,:), 'k:','linewidth',2); %Motion without perturbation
plot(r(2).x(1,:), r(2).x(2,:), 'k-','linewidth',2); %Motion with perturbation
plot(r(2).x(1,tn-1:tn), r(2).x(2,tn-1:tn), 'g.','markersize',20); %Perturbation
plot(r(2).x(1,tn-1:tn), r(2).x(2,tn-1:tn), 'g-','linewidth',2); %Perturbation
plot(Mu(1,:), Mu(2,:), 'r.','markersize',30); %Target
%Plot Agent 2
plot(r(1).x(3,1), r(1).x(4,1), 'k.','markersize',30); %Initial point
plot(r(1).x(3,:), r(1).x(4,:), 'k:','linewidth',2); %Motion without perturbation
plot(r(2).x(3,:), r(2).x(4,:), 'k-','linewidth',2); %Motion with perturbation
plot(Mu(3,:), Mu(4,:), 'r.','markersize',30); %Target
%Plot meeting points
plot(r(1).x(1,tlm(1)-1), r(1).x(2,tlm(1)-1), 'b.','markersize',20);
plot(r(2).x(1,tlm(1)-1), r(2).x(2,tlm(1)-1), 'b.','markersize',20); 
axis equal;

waitfor(h);
