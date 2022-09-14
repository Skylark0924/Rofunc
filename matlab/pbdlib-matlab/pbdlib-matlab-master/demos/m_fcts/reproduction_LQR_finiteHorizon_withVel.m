function r = reproduction_LQR_finiteHorizon_withVel(model, r, currPos, rFactor, Pfinal)
% Reproduction with a standard linear quadratic regulator of finite horizon
%
% Authors: Sylvain Calinon and Danilo Bruno, 2014
%
% This source code is given for free! In exchange, we would be grateful if you cite
% the following reference in any academic publication that uses this code or part of it:
%
% @inproceedings{Calinon14ICRA,
%   author="Calinon, S. and Bruno, D. and Caldwell, D. G.",
%   title="A task-parameterized probabilistic model with minimal intervention control",
%   booktitle="Proc. {IEEE} Intl Conf. on Robotics and Automation ({ICRA})",
%   year="2014",
%   month="May-June",
%   address="Hong Kong, China",
%   pages="3339--3344"
% }

nbData = size(r.currTar,2);
nbVarOut = model.nbVarPos;

%% LQR with cost = sum_t X(t)' Q(t) X(t) + u(t)' R u(t) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Definition of a double integrator system (DX = A X + B u with X = [x; dx])
A = kron([0 1; 0 0], eye(nbVarOut)); 
B = kron([0; 1], eye(nbVarOut)); 
%Initialize Q and R weighting matrices
Q = zeros(nbVarOut*2,nbVarOut*2,nbData);
for t=1:nbData
	Q(:,:,t) = inv(r.currSigma(:,:,t));
end
R = eye(nbVarOut) * rFactor;

if nargin<6
	%Pfinal = B*R*[eye(nbVarOut)*model.kP, eye(nbVarOut)*model.kV]
	Pfinal = B*R*[eye(nbVarOut)*0, eye(nbVarOut)*0]; %final feedback terms (boundary conditions)
end

%Auxiliary variables to minimize the cost function
P = zeros(nbVarOut*2, nbVarOut*2, nbData);
P(:,:,nbData) = Pfinal; %Compute P_T from the desired final feedback gains L_T
d = zeros(nbVarOut*2, nbData); %For optional feedforward term computation

tar = r.currTar;
dtar = gradient(tar,1,2)/model.dt;
%Backward integration of the Riccati equation and additional equation
for t=nbData-1:-1:1
	P(:,:,t) = P(:,:,t+1) + model.dt * (A'*P(:,:,t+1) + P(:,:,t+1)*A - P(:,:,t+1)*B*(R\B')*P(:,:,t+1) + Q(:,:,t+1)); 
	d(:,t) = d(:,t+1) + model.dt * ((A'-P(:,:,t+1)*B*(R\B'))*d(:,t+1) + P(:,:,t+1)*dtar(:,t+1) - P(:,:,t+1)*A*tar(:,t+1)); 
end

%Computation of the feedback term L 
L = zeros(nbVarOut, nbVarOut*2, nbData);
for t=1:nbData
	L(:,:,t) = R\B' * P(:,:,t); 
	M(:,t) = R\B' * d(:,t); %Optional feedforward term computation 
end

%% Reproduction with varying impedance parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = currPos(1:nbVarOut);
dx = zeros(nbVarOut,1); %currPos(nbVarOut+1:end);

for t=1:nbData
	%Compute acceleration (with both feedback and feedforward terms)	
	ddx = -L(:,:,t) * ([x;dx]-r.currTar(:,t)) + M(:,t); 
	
	%Update velocity and position
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;

	%Log data (with additional variables collected for analysis purpose)
	r.Data(:,t) = x;
	r.ddxNorm(t) = norm(ddx);
	%r.Kp(:,:,t) = L(:,1:nbVarOut,t);
	%r.Kv(:,:,t) = L(:,nbVarOut+1:end,t);
	r.kpDet(t) = det(L(:,1:nbVarOut,t));
	r.kvDet(t) = det(L(:,nbVarOut+1:end,t));
	%Note that if [V,D] = eigs(L(:,1:nbVarOut)), we have L(:,nbVarOut+1:end) = V * (2*D).^.5 * V'
end