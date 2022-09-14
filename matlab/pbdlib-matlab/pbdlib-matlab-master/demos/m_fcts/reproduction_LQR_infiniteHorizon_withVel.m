function [r,P] = reproduction_LQR_infiniteHorizon_withVel(model, r, currPos, rFactor)
% Reproduction with a linear quadratic regulator of infinite horizon
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
%nbVarOut = model.nbVarPos;
nbVarOut = 2;

%% LQR with cost = sum_t X(t)' Q(t) X(t) + u(t)' R u(t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Definition of a double integrator system (DX = A X + B u with X = [x; dx])
A = kron([0 1; 0 0], eye(nbVarOut)); 
B = kron([0; 1], eye(nbVarOut)); 
%Initialize Q and R weighting matrices
Q = zeros(nbVarOut*2,nbVarOut*2);
R = eye(nbVarOut) * rFactor;

%Variables for feedforward term computation (optional for movements with low dynamics)
%tar = [r.currTar; gradient(r.currTar,1,2)/model.dt];
%dtar = gradient(tar,1,2)/model.dt;
tar = r.currTar;
dtar = gradient(tar,1,2)/model.dt;


%% Reproduction with varying impedance parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = currPos(1:nbVarOut);
dx = currPos(nbVarOut+1:end); %zeros(nbVarOut,1);
for t=1:nbData
	%Current weighting term
    Q(:,:) = inv(r.currSigma(:,:,t));
		
	%care() is a function from the Matlab Control Toolbox to solve the algebraic Riccati equation,
	%also available with the control toolbox of GNU Octave
	%P = care(A, B, (Q+Q')/2, R); %(Q+Q')/2 is used instead of Q to avoid warnings for the symmetry of Q
	
	%Alternatively, the function below can be used for an implementation based on Schur decomposition
	%P = solveAlgebraicRiccati_Schur(A, B/R*B', (Q+Q')/2);
	
	%Alternatively, the function below can be used for an implementation based on Eigendecomposition
	%-> the only operator is eig([A -B/R*B'; -Q -A'])
	P = solveAlgebraicRiccati_eig(A, B/R*B', (Q+Q')/2); 
	
	%Variable for feedforward term computation (optional for movements with low dynamics)
	d = (P*B*(R\B')-A') \ (P*dtar(:,t) - P*A*tar(:,t)); 
	
	%Feedback term
	L = R\B'*P; 
	
	%Feedforward term
	M = R\B'*d; 
	
	%Compute acceleration (with only feedback terms)
    ddx =  -L * ([x;dx]-r.currTar(:,t));
	
	%Compute acceleration (with feedback and feedforward terms)
	%ddx =  -L * ([x;dx]-r.currTar(:,t)) + M; 
	
	%Update velocity and position
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;
	
	%Log data (with additional variables collected for analysis purpose)
	r.Data(:,t) = x;
  r.dx(:,t) = dx;
	r.ddx(:,t) = ddx;
	r.ddxNorm(t) = norm(ddx);
	%r.FB(:,t) = -L * [x-r.currTar(:,t); dx];
	%r.FF(:,t) = M;
	%r.Kp(:,:,t) = L(:,1:nbVarOut);
	%r.Kv(:,:,t) = L(:,nbVarOut+1:end);
	r.kpDet(t) = det(L(:,1:nbVarOut));
	r.kvDet(t) = det(L(:,nbVarOut+1:end));
	%Note that if [V,D] = eigs(L(:,1:nbVarOut)), we have L(:,nbVarOut+1:end) = V * (2*D).^.5 * V'
end