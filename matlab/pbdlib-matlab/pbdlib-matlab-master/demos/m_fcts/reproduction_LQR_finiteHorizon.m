function r = reproduction_LQR_finiteHorizon(model, r, currPos, rFactor, Pfinal)
% Reproduction with a linear quadratic regulator of finite horizon (continuous version).
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
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
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon (http://calinon.ch/) and Danilo Bruno 
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


[nbVarOut,nbData] = size(r.currTar);

%% LQR with cost = sum_t X(t)' Q(t) X(t) + u(t)' R u(t) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Definition of a double integrator system (DX = A X + B u with X = [x; dx])
A = kron([0 1; 0 0], eye(nbVarOut)); 
B = kron([0; 1], eye(nbVarOut)); 
%Initialize Q and R weighting matrices
Q = zeros(nbVarOut*2,nbVarOut*2,nbData);
for t=1:nbData
	Q(1:nbVarOut,1:nbVarOut,t) = inv(r.currSigma(:,:,t));
end
R = eye(nbVarOut) * rFactor;

if nargin<6
	%Pfinal = B*R*[eye(nbVarOut)*model.kP, eye(nbVarOut)*model.kV]
	Pfinal = B*R*[eye(nbVarOut)*0, eye(nbVarOut)*0]; %final feedback terms (boundary conditions)
end

%Auxiliary variables to minimize the cost function
P = zeros(nbVarOut*2, nbVarOut*2, nbData);
d = zeros(nbVarOut*2, nbData); %For optional feedforward term computation

%Feedback term
L = zeros(nbVarOut, nbVarOut*2, nbData);
%Compute P_T from the desired final feedback gains L_T,
P(:,:,nbData) = Pfinal;

%Variables for feedforward term computation (optional for movements with low dynamics)
%tar = [r.currTar; gradient(r.currTar,1,2)/model.dt];
tar = [r.currTar; zeros(nbVarOut,nbData)];
dtar = gradient(tar,1,2)/model.dt;

%Backward integration of the Riccati equation and additional equation
for t=nbData-1:-1:1
	P(:,:,t) = P(:,:,t+1) + model.dt * (A'*P(:,:,t+1) + P(:,:,t+1)*A - P(:,:,t+1)*B*(R\B')*P(:,:,t+1) + Q(:,:,t+1));
	%Optional feedforward term computation
	d(:,t) = d(:,t+1) + model.dt * ((A'-P(:,:,t+1)*B*(R\B'))*d(:,t+1) + P(:,:,t+1)*dtar(:,t+1) - P(:,:,t+1)*A*tar(:,t+1)); 
end
%Computation of the feedback term L and feedforward term M in u=-LX+M
for t=1:nbData
	L(:,:,t) = R\B' * P(:,:,t); %feedback term
	M(:,t) = R\B' * d(:,t); %feedforward term  
end

%% Reproduction with varying impedance parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = currPos;
dx = zeros(nbVarOut,1);

for t=1:nbData
	%Compute acceleration (with only feedback term)
	%ddx = -L(:,:,t) * [x-r.currTar(:,t); dx];
	
	%Compute acceleration (with both feedback and feedforward terms)
	ddx = -L(:,:,t) * [x-r.currTar(:,t); dx] + M(:,t); 
	
	%Update velocity and position
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;

	%Log data (with additional variables collected for analysis purpose)
	r.Data(:,t) = x;
	r.ddxNorm(t) = norm(ddx);
	%r.FB(:,t) = -L(:,:,t) * [x-r.currTar(:,t); dx];
	%r.FF(:,t) = M(:,t);
	%r.Kp(:,:,t) = L(:,1:nbVarOut,t);
	%r.Kv(:,:,t) = L(:,nbVarOut+1:end,t);
	r.kpDet(t) = det(L(:,1:nbVarOut,t));
	r.kvDet(t) = det(L(:,nbVarOut+1:end,t));
	%Note that if [V,D] = eigs(L(:,1:nbVarOut)), we have L(:,nbVarOut+1:end) = V * (2*D).^.5 * V'
end
