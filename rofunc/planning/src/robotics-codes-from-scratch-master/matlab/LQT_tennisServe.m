%%    Linear quadratic tracking (LQT) in a ballistic task mimicking a bimanual tennis 
%%    serve problem (batch formulation)
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

function LQT_tennisServe

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 200; %Number of datapoints
nbAgents = 3; %Number of agents (left hand, right hand, ball)
nbVarPos = 2 * nbAgents; %Dimension of position data (here: x1,x2 for the three agents)
nbDeriv = 2; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * (nbDeriv+1); %Dimension of state vector (position, velocity and force)
nbVarU = 4; %Number of control variables (acceleration commands for the two hands)
dt = 1E-2; %Time step duration
R = speye((nbData-1)*nbVarU) * 1E-8; %Control cost matrix

m = [2*1E-1, 2*1E-1, 2*1E-1]; %Mass of Agents (left hand, right hand and ball) 
g = [0; -9.81]; %Gravity vector

tEvent = [50, 100, 150]; %Time stamps when the ball is released, when the ball is hit, and when the hands come back to their initial pose 
x01 = [1.6; 0]; %Initial position of Agent 1 (left hand) 
x02 = [2; 0]; %Initial position of Agent 2 (right hand) 
xTar = [1; -.2]; %Desired target for Agent 3 (ball)


%% Linear dynamical system parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ac1 = kron([0, 1, 0; 0, 0, 1; 0, 0, 0], eye(2));
Bc1 = kron([0; 1; 0], eye(2));
Ac = kron(eye(nbAgents), Ac1);
Bc = [kron(eye(2), Bc1); zeros(6,nbVarU)]; %Ball is not directly controlled
%Parameters for discrete dynamical system
Ad = eye(nbVar) + Ac * dt; 
Bd = Bc * dt; 
%Initialize A and B
A = repmat(Ad, [1,1,nbData-1]);
B = repmat(Bd, [1,1,nbData-1]);
%Set Agent 3 state (ball) equals to Agent 1 state (left hand) until tEvent(1)
A(13:16,:,1:tEvent(1)) = 0;
A(13:14,1:2,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]);
A(13:14,3:4,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]) * dt;
A(15:16,3:4,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]);
A(15:16,5:6,1:tEvent(1)) = repmat(eye(2),[1,1,tEvent(1)]) * dt;	
%Set Agent 3 state (ball) equals to Agent 2 state (right hand) at tEvent(2)
A(13:16,:,tEvent(2)) = 0;
A(13:14,7:8,tEvent(2)) = eye(2);
A(13:14,9:10,tEvent(2)) = eye(2) * dt;
A(15:16,9:10,tEvent(2)) = eye(2);
A(15:16,11:12,tEvent(2)) = eye(2) * dt;
%Build transfer matrices
[Su, Sx] = transferMatrices(A, B); 


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = zeros(nbVar*nbData,1); %Sparse reference 
Q = zeros(nbVar*nbData); %Sparse precision matrix
%Agent 2 and Agent 3 must meet at tEvent(2) (right hand hitting the ball)
id = [7:8,13,14] + (tEvent(2)-1) * nbVar;
Q(id,id) = eye(4);
Q(id(1:2), id(3:4)) = -eye(2); %Common meeting point for the two agents
Q(id(3:4), id(1:2)) = -eye(2); %Common meeting point for the two agents
%Agent 1 (left hand) and Agent 2 (right hand) must come back to initial pose at tEvent(3) and stay here 
for t=tEvent(3):nbData
	id = [1:4] + (t-1) * nbVar; %Left hand
	Q(id,id) = eye(4) * 1E3;
	Mu(id) = [x01; zeros(2,1)];
	id = [7:10] + (t-1) * nbVar; %Right hand
	Q(id,id) = eye(4) * 1E3;
	Mu(id) = [x02; zeros(2,1)];
end
%Agent 3 (ball) must reach desired target at the end of the movement
id = [13:14] + (nbData-1) * nbVar;
Q(id,id) = eye(2);
Mu(id) = xTar;


%% Problem solved with linear quadratic tracking (LQT)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [x01; zeros(2,1); m(1)*g; ...
      x02; zeros(2,1); m(2)*g; ...
      x01; zeros(2,1); m(3)*g]; %Initial state
u = (Su' * Q * Su + R) \ Su' * Q * (Mu - Sx * x0); %Estimated control commands
x = reshape(Sx * x0 + Su * u, nbVar, nbData); %Generated trajectory


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h = figure('position',[10 10 1200 800],'color',[1 1 1]); hold on; axis off;
%Agents
hf(1) = plot(x(1,:), x(2,:), '-','linewidth',4,'color',[0 0 0]); %Agent 1 (left hand)
hf(2) = plot(x(7,:), x(8,:), '-','linewidth',4,'color',[.6 .6 .6]); %Agent 2 (right hand)
hf(3) = plot(x(13,:), x(14,:), ':','linewidth',4,'color',[.8 .4 0]); %Agent 3 (ball)
%Events
hf(4) = plot(x(1,1), x(2,1), '.','markersize',40,'color',[0 0 0]); %Initial position (left hand)
hf(5) = plot(x(7,1), x(8,1), '.','markersize',40,'color',[.6 .6 .6]); %Initial position (right hand)
hf(6) = plot(x(1,tEvent(1)), x(2,tEvent(1)), '.','markersize',40,'color',[0 .6 0]); %Release of ball
hf(7) = plot(x(7,tEvent(2)), x(8,tEvent(2)), '.','markersize',40,'color',[0 0 .8]); %Hitting of ball
hf(8) = plot(xTar(1), xTar(2), '.','markersize',40,'color',[.8 0 0]); %Ball target
legend(hf, {'Left hand motion (Agent 1)','Right hand motion (Agent 2)','Ball motion (Agent 3)', 'Left hand initial point', ...
            'Right hand initial point','Ball releasing point','Ball hitting point','Ball target'}, 'fontsize',20,'location','northwest'); 
legend('boxoff');
axis equal; 

waitfor(h);
end

%%%%%%%%%%%%%%%%%%%%%%%%%
function [Su, Sx] = transferMatrices(A, B)
	[nbVax, nbVarU, nbData] = size(B);
	nbData = nbData+1;
	Sx = kron(ones(nbData,1), speye(nbVax)); 
	Su = sparse(zeros(nbVax*(nbData-1), nbVarU*(nbData-1)));
	for t=1:nbData-1
		id1 = (t-1)*nbVax+1:t*nbVax;
		id2 = t*nbVax+1:(t+1)*nbVax;
		id3 = (t-1)*nbVarU+1:t*nbVarU;
		Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);
		Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
		Su(id2,id3) = B(:,:,t);	
	end
end		
