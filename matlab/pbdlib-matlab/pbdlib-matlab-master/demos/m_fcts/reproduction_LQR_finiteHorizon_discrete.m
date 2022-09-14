function r = reproduction_LQR_finiteHorizon_discrete(model, r, currPos, rFactor)
%Discrete LQR
%Implementation based on "Predictive control for linear and hybrid systems", Borrelli, Bemporad and Morari, 2015 
%Sylvain Calinon, Martijn Zeestraten, 2015

nbData = size(r.currTar,2);
nbVarOut = model.nbVarPos;

%% LQR with cost = sum_t X(t)' Q(t) X(t) + u(t)' R u(t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Definition of a double integrator system (DX = A X + B u with X = [x; dx])
%A = kron([0 1; 0 0], eye(nbVarOut)); 
%B = kron([0; 1], eye(nbVarOut)); 
%C = kron([1,zeros(1,model.nbDeriv-1)],eye(model.nbVarPos)); % Output Matrix (we assume we only observe position)

% System matrix (create nbDeriv integrator system)
A = zeros(model.nbDeriv);
A(1:end-1,2:end) = eye(model.nbDeriv-1);
A = kron(A,eye(model.nbVarPos));

% Input matrix (we assume that we control the systen via the highest derivative, and that
% any higher derivative will result in a non-fully contrallable system
B = kron([zeros(model.nbDeriv-1,1);1],eye(model.nbVarPos)); 

% Discretize system (Euler method)
Ad = A*model.dt + eye(size(A));
Bd = B*model.dt;

%Initialize Q and R weighting matrices
R = eye(model.nbVarPos) * rFactor;

%Auxiliary variables to minimize the cost function
P = zeros(model.nbVar, model.nbVar, nbData);
P(:,:,nbData) = inv(r.currSigma(:,:,end));%
d = zeros(model.nbVar, nbData);

%Backward integration of the Riccati equation and additional equation
for t=nbData-1:-1:1
	Q = inv(r.currSigma(:,:,t));
	P(:,:,t) = Q - Ad' * (P(:,:,t+1) * Bd / (Bd' * P(:,:,t+1) * Bd + R) * Bd' * P(:,:,t+1) - P(:,:,t+1)) * Ad;
	d(:,t) = (Ad' - Ad'*P(:,:,t+1) * Bd * inv(R + Bd' * P(:,:,t+1) * Bd) * Bd' ) * (P(:,:,t+1) * (Ad * r.currTar(:,t) - r.currTar(:,t+1)) + d(:,t+1));
end

%Computation of the feedback term L 
L = zeros(nbVarOut, nbVarOut*model.nbDeriv, nbData);
M = zeros(nbVarOut, nbData);
for t=1:nbData
	L(:,:,t) = (Bd' * P(:,:,t) * Bd + R) \ Bd' * P(:,:,t) * Ad;
	M(:,t) = (Bd' * P(:,:,t) * Bd + R) \ Bd' * (P(:,:,t) * (Ad * r.currTar(:,t) - r.currTar(:,t)) + d(:,t)); 
end


%% Reproduction with varying impedance parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = currPos(1:nbVarOut);
dx = zeros(nbVarOut,1); %currPos(nbVarOut+1:end);

for t=1:nbData
	%Compute acceleration (with both feedback and feedforward terms)	
	%ddx = -L(:,:,t) * ([x;dx]-r.currTar(:,t)); 
	ddx = -L(:,:,t) * ([x;dx]-r.currTar(:,t)) - M(:,t); 
	
% 	st = Ad * [x;dx] + Bd * ddx;
% 	x = st(1:nbVarOut);
% 	dx = st(nbVarOut+1:2*nbVarOut);
	
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