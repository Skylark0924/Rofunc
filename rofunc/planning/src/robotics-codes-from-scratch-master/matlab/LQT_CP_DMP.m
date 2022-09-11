%%    Linear quadratic tracking (LQT) with control primitives applied to a trajectory 
%%    tracking task, with a formulation similar to dynamical movement primitives (DMP),
%%    by using the least squares formulation of recursive LQR on an augmented state space 
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

function LQT_CP_DMP

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.nbData = 100; %Number of datapoints
param.nbSamples = 10; %Number of generated trajectory samples
param.nbVarU = 2; %Dimension of position data (here: x1,x2)
param.nbDeriv = 3; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
param.nbVar = param.nbVarU * param.nbDeriv; %Dimension of state vector
param.nbVarX = param.nbVar + 1; %Augmented state space
param.dt = 1E-2; %Time step duration
param.nbFct = 12; %Number of basis functions
param.basisName = 'RBF'; %PIECEWISE, RBF, BERNSTEIN, FOURIER
param.r = 1E-9; %Control cost in LQR 

%Dynamical System settings (for augmented state space)
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
A = [A0, zeros(param.nbVar,1); zeros(1,param.nbVar), 1]; %Augmented A (homogeneous)
B = [B0; zeros(1,param.nbVarU)]; %Augmented B (homogeneous)

%Build Sx and Su transfer matrices (for augmented state space)
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

%Build basis functions
if isequal(param.basisName,'PIECEWISE')
	phi = buildPhiPiecewise(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'RBF')
	phi = buildPhiRBF(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'BERNSTEIN')
	phi = buildPhiBernstein(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'FOURIER')
	phi = buildPhiFourier(param.nbData-1, param.nbFct);
end

%Application of basis functions to multidimensional control commands
Psi = kron(phi, eye(param.nbVarU)); 


%% Task description
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('../data/2Dletters/S.mat'); %Planar trajectories for writing alphabet letters
MuPos = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),param.nbData)); %Position
MuVel = gradient(MuPos) / param.dt;
MuAcc = gradient(MuVel) / param.dt;
Mu = [MuPos; MuVel; MuAcc; zeros(param.nbVar-3*param.nbVarU, param.nbData)]; %Position, velocity and acceleration profiles as references

%Task setting (tracking of acceleration profile and reaching of an end-point)
Q = kron(speye(param.nbData), ...
    diag([zeros(1,param.nbVarU*2), ones(1,param.nbVarU) * 1E-6, zeros(1,param.nbVar-3*param.nbVarU)])); %Precision matrix (for acceleration only)
Q(end-param.nbVar+1:end-param.nbVar+2*param.nbVarU, end-param.nbVar+1:end-param.nbVar+2*param.nbVarU) = eye(2*param.nbVarU) * 1E0; %Add last point as target (position and velocity)

%Weighting matrices in augmented state form
Qm = zeros(param.nbVarX * param.nbData);
for t=1:param.nbData
	id0 = [1:param.nbVar] + (t-1) * param.nbVar;
	id = [1:param.nbVarX] + (t-1) * param.nbVarX;
	Qm(id,id) = [eye(param.nbVar), zeros(param.nbVar,1); -Mu(:,t)', 1] * blkdiag(Q(id0,id0), 1) * ...
	            [eye(param.nbVar), -Mu(:,t); zeros(1,param.nbVar), 1];
end
Rm = speye((param.nbData-1)*param.nbVarU) * param.r; %Standard control weight matrix (at trajectory level)


%% Least squares formulation of recursive LQR with an augmented state space and and control primitives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%xn = [randn(param.nbVarU, 1) * 5E0; randn(param.nbVar-param.nbVarU, 1) * 0; 0]; %Simulated noise on state
xn = [3; -.5; zeros(param.nbVarX-param.nbVarU, 1)]; %Simulated noise on state

%F = (Su' * Qm * Su + Rm) \ Su' * Qm * Sx; %Standard F
W = (Psi' * Su' * Qm * Su * Psi + Psi' * Rm * Psi) \ Psi' * Su' * Qm * Sx;
F = Psi * W; %F with control primitives

%Reproduction with feedback controller on augmented state space (with CP)
Ka(:,:,1) = F(1:param.nbVarU,:);
P = eye(param.nbVarX);
for t=2:param.nbData-1
	id = (t-1)*param.nbVarU + [1:param.nbVarU];
	P = P / (A - B * Ka(:,:,t-1));
	Ka(:,:,t) = F(id,:) * P;
end
for n=1:2
	x = [Mu(:,1) + [2; 1; zeros(param.nbVar-2,1)]; 1]; %Augmented state space initialization (by simulating position offset)
	for t=1:param.nbData-1
		u = -Ka(:,:,t) * x; %Feedback control on augmented state (resulting in feedback and feedforward terms on state)
%		K = Ka(:,1:param.nbVar,t); %Feedback gain
%		uff = -Ka(:,end,t) - K * Mu(:,t); %Feedforward control command
%		u = K * (Mu(:,t) - x(1:end-1)) + uff; %Acceleration command with feedback and feedforward terms computed explicitly from Ka

		x = A * x + B * u; %Update of state vector	
		if t==25 && n==2
			x = x + xn; %Simulated noise on the state
		end
		r(n).x(:,t) = x; %State
	end
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h(1) = figure('position',[10 10 800 800]); hold on; axis off;
plot(Mu(1,:), Mu(2,:), 'b-','linewidth',2);
plot(Mu(1,end), Mu(2,end), 'r.','markersize',40);
plot(r(1).x(1,1), r(1).x(2,1), 'k.','markersize',30);
plot(r(1).x(1,:), r(1).x(2,:), 'k:','linewidth',2);
plot(r(2).x(1,:), r(2).x(2,:), 'k-','linewidth',2);
plot(r(2).x(1,24:25), r(2).x(2,24:25), 'g.','markersize',20);
axis equal; 

waitfor(h);
end


%Building piecewise constant basis functions
function phi = buildPhiPiecewise(nbData, nbFct) 
	phi = kron(eye(nbFct), ones(ceil(nbData/nbFct),1));
	phi = phi(1:nbData,:);
end

%Building radial basis functions (RBFs)
function phi = buildPhiRBF(nbData, nbFct) 
	t = linspace(0, 1, nbData);
	tMu = linspace(t(1)-1/(nbFct-3), t(end)+1/(nbFct-3), nbFct); %Repartition of centers to limit border effects
	sigma = 1 / (nbFct-2); %Standard deviation
	phi = exp(-(t' - tMu).^2 / sigma^2);
	
	%Optional rescaling
	%phi = phi ./ repmat(sum(phi,2), 1, nbFct); 
end

%Building Bernstein basis functions
function phi = buildPhiBernstein(nbData, nbFct)
	t = linspace(0, 1, nbData);
	phi = zeros(nbData, nbFct);
	for i=1:nbFct
		phi(:,i) = factorial(nbFct-1) ./ (factorial(i-1) .* factorial(nbFct-i)) .* (1-t).^(nbFct-i) .* t.^(i-1);
	end
end

%Building Fourier basis functions
function phi = buildPhiFourier(nbData, nbFct)
	t = linspace(0, 1, nbData);
	
	%Computation for general signals (incl. complex numbers)
	d = ceil((nbFct-1)/2);
	k = -d:d;
	phi = exp(t' * k * 2 * pi * 1i); 
	%phi = cos(t' * k * 2 * pi); %Alternative computation for real signal
		
%	%Alternative computation for real and even signal
%	k = 0:nbFct-1;
%	phi = cos(t' * k * 2 * pi);
%	%phi(:,2:end) = phi(:,2:end) * 2;
%	%invPhi = cos(k' * t * 2 * pi) / nbData;
%	%invPsi = kron(invPhi, eye(param.nbVar));
end
