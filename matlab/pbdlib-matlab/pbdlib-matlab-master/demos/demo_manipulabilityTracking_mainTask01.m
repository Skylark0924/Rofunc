function demo_manipulabilityTracking_mainTask01
% Matching of a desired manipulability ellipsoid as the main task (no desired position) 
% using the formulation with the manipulability Jacobian (Mandel notation).
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% If this code is useful for your research, please cite the related publication:
% @inproceedings{Jaquier18RSS,
% 	author="Jaquier, N. and Rozo, L. and Caldwell, D. G. and Calinon, S.",
% 	title="Geometry-aware Tracking of Manipulability Ellipsoids",
% 	booktitle="Robotics: Science and Systems ({R:SS})",
% 	year="2018",
% 	pages="1--9"
% }
% 
% Copyright (c) 2017 Idiap Research Institute, http://idiap.ch/
% Written by Noémie Jaquier and Leonel Rozo
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
dt = 1E-2;	% Time step
nbIter = 65; % Number of iterations
Km = 3; % Gain for manipulability control in task space


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot parameters 
nbDOFs = 4; %Nb of degrees of freedom
armLength = 4; % Links length

% Robot
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
q = sym('q', [1 nbDOFs]);	% Symbolic robot joints
J = robot.jacob0(q'); % Symbolic Jacobian

% Define the desired manipulability
q_Me_d = [pi/16 ; pi/4 ; pi/8 ; -pi/8]; % task 1
% q_Me_d = [pi/2 ; -pi/6; -pi/2 ; -pi/2]; % task 2
J_Me_d = robot.jacob0(q_Me_d); % Current Jacobian
J_Me_d = J_Me_d(1:2,:);
Me_d = J_Me_d * J_Me_d';


%% Testing Manipulability Transfer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initial conditions
q0 = [pi/2 ; -pi/6; -pi/2 ; -pi/2]; % Initial robot configuration task 1
% q0 = [0.0 ; pi/4 ; pi/2 ; pi/8]; % Initial robot configuration task 2

qt = q0;
it = 1; % Iterations counter
h1 = [];
gmm_c = [];

% Initial end-effector position (compatibility with 9.X and 10.X versions of robotics toolbox)
Htmp = robot.fkine(q0); % Forward Kinematics
if isobject(Htmp) % SE3 object verification
	x0 = Htmp.t(1:2);
else
	x0 = Htmp(1:2,end);
end

figure('position',[10 10 1000 450],'color',[1 1 1]);

% Main control loop
while( it < nbIter )
	delete(h1);
	
	Htmp = robot.fkine(qt); % Forward Kinematics (needed for plots)
	Jt_full = robot.jacob0(qt); % Current Jacobian
    Jt = Jt_full(1:2,:);
	Me_ct = Jt * Jt'; % Current manipulability
	
	% Log data
    Me_track(:,:,it) = Me_ct;
	qt_track(:,it) = qt;
	
	% Current end-effector position
	if isobject(Htmp) % SE3 object verification
		xt = Htmp.t(1:2);
	else
		xt = Htmp(1:2,end);
	end

	% Compute manipulability Jacobian
	Jm_t = compute_red_manipulability_Jacobian(Jt_full, 1:2);
	
	% Compute desired joint velocities
	M_diff = logmap(Me_d, Me_ct);
	dq_T1 = pinv(Jm_t) * Km * symmat2vec(M_diff);
	
	% Plotting robot and manipulability ellipsoids
	subplot(1,2,1); hold on;
	if(it == 1)
		plotGMM(xt, 1E-2*Me_d, [0.2 0.8 0.2], .4); % Scaled matrix!
	end
	h1 = plotGMM(xt, 1E-2*Me_ct, [0.8 0.2 0.2], .4); % Scaled matrix!
	colTmp = [1,1,1] - [.8,.8,.8] * (it+10)/nbIter;
	plotArm(qt, ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp);
	axis square; axis equal;
	xlabel('x_1'); ylabel('x_2');
	
	subplot(1,2,2); hold on; axis equal;
	delete(gmm_c);
	gmm_c = plotGMM([0;0], 1E-2*Me_ct, [0.8 0.2 0.2], .1); % Scaled matrix!
	if(it == 1)
		plotGMM([0;0], 1E-2*Me_d, [0.2 0.8 0.2], .1); % Scaled matrix!
	end
	drawnow;
	
	% Updating joint position
	qt = qt + (dq_T1) * dt;
	it = it + 1; % Iterations++
end


%% Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robot and manipulability ellipsoids
figure('position',[10 10 900 900],'color',[1 1 1]);
hold on;
p = [];
for it = 1:2:nbIter-1
	colTmp = [1,1,1] - [.8,.8,.8] * (it)/nbIter;
	p = [p; plotArm(qt_track(:,it), ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp)];
end

z = get(p,'ZData'); % to put arm plot down compared to GMM plot
for i = 1:size(z,1)
	if isempty(z{i})
	set(p,'ZData',z{i}-10) 
	end
end

plotGMM(xt, 1E-2*Me_d, [0.2 0.8 0.2], .4);
plotGMM(x0, 1E-2*Me_track(:,:,1), [0.2 0.2 0.8], .4);
plotGMM(xt, 1E-2*Me_ct, [0.8 0.2 0.2], .4);
axis equal
set(gca,'xtick',[],'ytick',[])
xlabel('$x_1$','fontsize',40,'Interpreter','latex'); ylabel('$x_2$','fontsize',40,'Interpreter','latex');

% Initial manipulability ellipsoid
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, [0.2 0.8 0.2], .4);
plotGMM([0;0], 1E-2*Me_track(:,:,1), [0.2 0.2 0.8], .4);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1.8 1.8]);ylim([-1.8 1.8]);
set(gca,'xtick',[],'ytick',[]);
text(-.8,1,0,'Initial','FontSize',38,'Interpreter','latex')
axis equal;

% Final manipulability ellipsoid
figure('position',[10 10 450 450],'color',[1 1 1]);
hold on;
plotGMM([0;0], 1E-2*Me_d, [0.2 0.8 0.2], .4);
plotGMM([0;0], 1E-2*Me_ct, [0.8 0.2 0.2], .4);
xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
xlim([-1.8 1.8]);ylim([-1.8 1.8]);
text(-.7,1,0,'Final','FontSize',38,'Interpreter','latex')
set(gca,'xtick',[],'ytick',[])
axis equal;

% Cost
figure()
hold on;
for it = 1:nbIter-1
	cost(it) = norm(logm(Me_d^-.5*Me_track(:,:,it)*Me_d^-.5),'fro');
end
plot([1:nbIter-1].*dt, cost, '-','color',[0 0 .7],'Linewidth',3);
set(gca,'fontsize',14);
xlim([0 nbIter*dt])
xlabel('$t$','fontsize',22,'Interpreter','latex'); ylabel('$d$','fontsize',22,'Interpreter','latex');

end

%%%%%%%%%%%%%%%%%%
% Compute the force manipulability Jacobian (symbolic) in the form of a matrix using Mandel notation
function Jm_red = compute_red_manipulability_Jacobian(J, taskVar)
	if nargin < 2
		taskVar = 1:6;
	end
	Jm = compute_manipulability_Jacobian(J);
	Jm_red = [];
	for i=1:size(Jm,3)
		Jm_red = [Jm_red, symmat2vec(Jm(taskVar,taskVar,i))];
	end
end

%%%%%%%%%%%%%%%%%%
% Compute the force manipulability Jacobian (symbolic)
function Jm = compute_manipulability_Jacobian(J)
	J_grad = compute_joint_derivative_Jacobian(J);
	Jm = tmprod(J_grad,J,2) + tmprod(permute(J_grad,[2,1,3]),J,1);
	% mat_mult = kdlJacToArma(J)*reshape( arma::mat(permDerivJ.memptr(), permDerivJ.n_elem, 1, false), columns, columns*rows);
	% dJtdq_J = arma::cube(mat_mult.memptr(), rows, rows, columns);
end

%%%%%%%%%%%%%%%%%%
% Compute the Jacobian derivative w.r.t joint angles (hybrid Jacobian representation). Ref: H. Bruyninck and J. de Schutter, 1996
function J_grad = compute_joint_derivative_Jacobian(J)
	nb_rows = size(J,1); % task space dim.
	nb_cols = size(J,2); % joint space dim.
	J_grad = zeros(nb_rows, nb_cols, nb_cols);
	for i=1:nb_cols
		for j=1:nb_cols
			J_i = J(:,i);
			J_j = J(:,j);
			if j < i
				J_grad(1:3,i,j) = cross(J_j(4:6,:), J_i(1:3,:));
				J_grad(4:6,i,j) = cross(J_j(4:6,:), J_i(4:6,:));
			elseif j > i
				J_grad(1:3,i,j) = -cross(J_j(1:3,:), J_i(4:6,:));
			else
				J_grad(1:3,i,j) = cross(J_i(4:6,:), J_i(1:3,:));
			end
		end
	end
end

%%%%%%%%%%%%%%%%%%
% Logarithm map (SPD manifold)
function U = logmap(X, S)	
	N = size(X,3);
	for n = 1:N
	% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
	% 	tic
	% 	U(:,:,n) = S * logm(S \ X(:,:,n));
	% 	toc
	% 	tic
		[V, D] = eig(S \ X(:,:,n));
		U(:,:,n) = S * V * diag(log(diag(D))) / V;
	% 	toc
	end
end

%%%%%%%%%%%%%%%%%%
% Vectorization of a symmetric matrix
function v = symmat2vec(M)
	N = size(M,1);
	v = diag(M);
	for n = 1:N-1
		v = [v; sqrt(2) .* diag(M,n)]; % Mandel notation
	end
end

%%%%%%%%%%%%%%%%%%
% Mode-n tensor-matrix product
function [S,iperm] = tmprod(T, U, mode)
	size_tens = ones(1,mode);
	size_tens(1:ndims(T)) = size(T);
	N = length(size_tens);

	% Compute the complement of the set of modes.
	bits = ones(1,N);
	bits(mode) = 0;
	modec = 1:N;
	modec = modec(logical(bits(modec)));

	% Permutation of the tensor
	perm = [mode modec];
	size_tens = size_tens(perm);
	S = T; 
	if mode ~= 1
		S = permute(S,perm); 
	end

	% n-mode product
	size_tens(1) = size(U,1);
	S = reshape(U*reshape(S,size(S,1),[]),size_tens);

	% Inverse permutation
	iperm(perm(1:N)) = 1:N;
	S = permute(S,iperm); 
end
