function demo_manipulabilityTransfer01
% This code shows how a robot can exploit its redundancy to modify its manipulability so that it matches, 
% as close as possible, a desired manipulability ellipsoid (possibly obtained from another robot or a human)
% The approach evaluates a cost function that measures the similarity between manipulabilities and computes a nullspace velocity  
% command designed to change the robot posture so that its manipulability ellipsoid becomes more similar to the desired one.
% Two cost functions are tested:
%   1. A. Ajoudani et al. ICRA'2015
%   2. Squared Stein divergence (metrics for manifolds of PD matrices)
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% If this code is useful for your research, please cite the related publication:
% @article{Jaquier18,
% 	author="Jaquier, N. and Rozo, L. and Caldwell, D. G. and Calinon, S.",
% 	title="Geometry-aware Manipulability Transfer",
% 	journal="arXiv:1811.11050",
% 	year="2018",
% 	pages="1--20"
% }
%
% Written by Leonel Rozo, 2017
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

addpath('./m_fcts/');
disp('This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).');


%% Auxiliary variables
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
typeCost = 1; % Defines the cost to be minimized
dt = 1E-2;	% Time step


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Robot parameters
nbDOFs = 4; %Nb of degrees of freedom
armLength = 4; % Links length
% Robot
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
J = robot.jacob0(q'); % Symbolic Jacobian

%% Computation of cost function and its gradient
% Computation of the cost function
Me_c = J(1:2,:)*J(1:2,:)';	% Current VME
Me_d = [20 -40 ; -40 150]; % Desired VME
% Me_d = [103.09 80.07 ; 80.07 70.12]; % Desired VME
% Me_d = [10 1 ; 1 200]; % Desired VME
[eVc, ~] = eig(Me_d);
des_vec = eVc(:,1)/norm(eVc(:,1)); % Desired VME major axis for cost type 1

switch typeCost
	case 1 % A. Ajoudani approach
		% Minus sign is included here because the cost function is being
		% minimized in the nullspace
		C = -inv((des_vec)' * Me_c * (des_vec));
		alpha = 100; % Gain for nullspace vector
		%     alpha = 850; % Gain for nullspace vector
		
		%   case 2 % Log-euclidean metrics
		% 1st version -> Takes TOO LONG to be evaluated
		% 		C = norm(logm(Me_d) - logm(Me_c) ,'fro');
		
		% 2nd version -> Also takes TOO LONG to be evaluated
		% Squared log-euclidean metrics
		% 		Me_delta = logm(Me_d) - logm(Me_c);
		% 		C = trace(Me_delta*ctranspose(Me_delta));
		
		% 3rd version -> Also takes TOO LONG to be evaluated
		% Squared log-euclidean metrics
		%     Me_delta = logm(Me_d) - logm(Me_c);
		%     C = sum(sum(Me_delta).^2);
		
	case 2 % Squared Stein divergence
		C = log(det(0.5*(Me_d+Me_c))) - 0.5*log(det(Me_d*Me_c));
		alpha = 15.5e0; % Gain for nullspace vector
		%     alpha = 12.5e0; % Gain for nullspace vector
end

% Cost gradient computation (symbolically)
C_gradient = [];
for i = 1 : nbDOFs
	C_gradient = [C_gradient ; diff(C,q(i))];
end

% Creating a MATLAB function from symbolic variable
Cgrad =  matlabFunction(C_gradient);
%   Cgrad =  matlabFunction(C_gradient,'File','ManEllip_Cgradient.m');


%% Testing Manipulability Transfer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
dxh = [0; 0]; % Desired Cartesian velocity
% q0 = [0.0 ; pi/4 ; 0.0 ; 0.0]; % Initial robot configuration
% q0 =[0.5403 ; 0.9143 ; 4.4748 ; 1.0517]; % Initial robot configuration
q0 =[pi/2 ; -pi/6; -pi/2 ; -pi/2]; % Initial robot configuration
qt = q0;
nbIter = 30;
it = 1; % Iterations counter
h1 = [];

figure('position',[10 10 1000 450],'color',[1 1 1]);
% figure('position',[2000 250 1000 450],'color',[1 1 1]);
while( it < nbIter )
	delete(h1);
	
	Jt = robot.jacob0(qt); % Current Jacobian
	Jt = Jt(1:2,:);
	Htmp = robot.fkine(qt); % Forward Kinematics (needed for plots)
	
	% Compatibility with 9.X and 10.X versions of robotics toolbox
	if isobject(Htmp) % SE3 object verification
		xt = Htmp.t(1:2);
	else
		xt = Htmp(1:2,end);
	end
	
	% Evaluating gradient at current configuration
	Cgrad_t = Cgrad(qt(1),qt(2),qt(3),qt(4));
	
	% Desired joint velocities
	dq_T1 = pinv(Jt)*dxh; % Main task joint velocities (given desired dx)
	dq_ns = -(eye(nbDOFs) - pinv(Jt)*Jt) * alpha * Cgrad_t; % Redundancy resolution
	
	% Plotting robot and VMEs
	subplot(1,2,1); hold on;
	if(it == 1)
		plotGMM(xt, 1E-2*Me_d, [0.2 0.8 0.2], .4); % Scaled matrix!
	end
	h1 = plotGMM(xt, 1E-2*(Jt*Jt'), [0.8 0.2 0.2], .4); % Scaled matrix!
	colTmp = [1,1,1] - [.7,.7,.7] * it/nbIter;
	plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp);
	axis square;
	axis equal;
	xlabel('x_1'); ylabel('x_2');
	
	% Plotting costs
	subplot (2,3,3); hold on;
	Me_ct = (Jt*Jt');
	C1t = inv((des_vec)' * Me_ct * (des_vec));
	plot(it, C1t, '.b');
	xlabel('t'); ylabel('c1_t');
	
	subplot (2,3,6); hold on;
	C2t = log(det(0.5*(Me_d+Me_ct))) - 0.5*log(det(Me_d*Me_ct));
	plot(it, C2t, 'xr');
	xlabel('t'); ylabel('c2_t');
	drawnow;
	
	% Updating joint position
	qt = qt + (dq_T1 + dq_ns)*dt;
	it = it + 1; % Iterations++
	% 	pause;
end