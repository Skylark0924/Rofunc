function demo_manipulabilityTracking_mainTask02
% Matching of a desired manipulability ellipsoid as the main task (no desired position)
% using the formulation with the manipulability Jacobian (Mandel notation).
% The matrix gain used for the manipulability tracking controller is now defined as the inverse of
% a 2nd-order covariance matrix representing the variability information obtained from a learning algorithm.
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
nbIter = 100; % Number of iterations
clrmap = lines(4);

% Initialisation of the covariance transformation
cov_2D(:,:,1) = [0.3 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3];  % Unitary gain (baseline)
cov_2D(:,:,2) = [0.03 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.3]; % Priority on matching for 'x' axis
cov_2D(:,:,3) = [0.3 0.0 0.0; 0.0 0.03 0.0; 0.0 0.0 0.3]; % Priority on matching for 'y' axis
cov_2D(:,:,4) = [0.3 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.1]; % Correlation has highest priority for tracking

names = {'Baseline','Priority on x-axis','Priority on y-axis','Priority on xy-correltation'};

for n = 1:4
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
	q_Me_d = [0.0 ; pi/4 ; pi/2 ; pi/8];
	J_Me_d = robot.jacob0(q_Me_d); % Current Jacobian
	J_Me_d = J_Me_d(1:2,:);
	Me_d = (J_Me_d*J_Me_d');
	
	%% Testing Manipulability Transfer
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Initial conditions
	q0 =[pi/2 ; -pi/6; -pi/2 ; -pi/2]; % Initial robot configuration
	
	qt = q0;
	it = 1; % Iterations counter
	h1 = [];
	
	while( it < nbIter )
		delete(h1);
		
		Jt = robot.jacob0(qt); % Current Jacobian
		Jt_full = Jt;
		Jt = Jt(1:2,:);
		Htmp = robot.fkine(qt); % Forward Kinematics
		Me_ct = (Jt*Jt'); % Current manipulability
		
		Me_track(:,:,it,n)=Me_ct;
		qt_track(:,it) = qt;
		
		% Compatibility with 9.X and 10.X versions of robotics toolbox
		if isobject(Htmp) % SE3 object verification
			xt = Htmp.t(1:2);
		else
			xt = Htmp(1:2,end);
		end
		xt_track(:,it,n) = xt;
		
		% Compute manipulability Jacobian
		Jm_t = compute_red_manipulability_Jacobian(Jt_full, 1:2);
		
		% Desired joint velocities
		M_diff = logmap(Me_d,Me_ct);
		dq_T1 = pinv(Jm_t)*(cov_2D(:,:,n)\symmat2vec(M_diff));
		
		% Updating joint position
		qt = qt + (dq_T1)*dt;
		it = it + 1; % Iterations
	end
	
	%% Final plots
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Robot and manipulability ellipsoids
	figure('position',[10 10 900 900],'color',[1 1 1]); hold on;
	p = [];
	for it = 1:2:nbIter-1
		colTmp = [.95,.95,.95] - [.8,.8,.8] * (it)/nbIter;
		p = [p; plotArm(qt_track(:,it), ones(nbDOFs,1)*armLength, [0; 0; it*0.1], .2, colTmp)];
	end
	
	z = get(p,'ZData'); % to put arm plot down compared to GMM plot
	for i = 1:size(z,1)
		if isempty(z{i})
			set(p,'ZData',z{i}-10)
		end
	end
	
	plotGMM(xt(:,end), 1E-2*Me_d, [0.2 0.8 0.2], .4);
	plotGMM(xt_track(:,1,n), 1E-2*Me_track(:,:,1,n), [0.2 0.2 0.8], .4);
	plotGMM(xt(:,end), 1E-2*Me_ct, clrmap(n,:), .4);
	axis equal; xlim([-3,9]); ylim([-4,8]);
	set(gca,'fontsize',22,'xtick',[],'ytick',[]);
	xlabel('$x_1$','fontsize',38,'Interpreter','latex'); ylabel('$x_2$','fontsize',38,'Interpreter','latex');
	title(names{n});
	drawnow;
end


%% Final plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Manipulability ellipsoids
for n = 1:4
	figure('position',[10 10 1200 300],'color',[1 1 1]); hold on;
	for it = 1:10:nbIter-1
		plotGMM([it*dt;0], 2E-4*Me_d, [0.2 0.8 0.2], .1);
		plotGMM([it*dt;0], 2E-4*Me_track(:,:,it,n), clrmap(n,:), .3);
	end
	axis equal;
	set(gca,'fontsize',24,'ytick',[]);
	xlabel('$t$','fontsize',42,'Interpreter','latex');
	ylabel('$\mathbf{M}$','fontsize',42,'Interpreter','latex');
	title(names{n});
end

figure('position',[10 10 900 900],'color',[1 1 1]); hold on;
for it = 1:2:nbIter-1
	for n = 1:4
		plotGMM(xt_track(:,it,n), 1E-2*Me_track(:,:,it,n), clrmap(n,:), .15);
	end
end
plotGMM(xt_track(:,1,1), 1E-2*Me_track(:,:,1,1), [0.1 0.1 0.4], .6);
axis equal;
set(gca,'fontsize',18);
xlabel('$x_1$','fontsize',30,'Interpreter','latex'); ylabel('$x_2$','fontsize',30,'Interpreter','latex');
title('End-effector and manipulability ellipsoids trajectories');

end

%%%%%%%%%%%%%%%%%%
function Jm_red = compute_red_manipulability_Jacobian(J, taskVar)
	% Compute the force manipulability Jacobian (symbolic) in the form of a
	% matrix using Mandel notation.
	if nargin < 2
		taskVar = 1:6;
	end

	Jm = compute_manipulability_Jacobian(J);
	Jm_red = [];
	for i = 1:size(Jm,3)
		Jm_red = [Jm_red, symmat2vec(Jm(taskVar,taskVar,i))];
	end
end

%%%%%%%%%%%%%%%%%%
function Jm = compute_manipulability_Jacobian(J)
	% Compute the force manipulability Jacobian (symbolic).
	J_grad = compute_joint_derivative_Jacobian(J);
	Jm = tmprod(J_grad,J,2) + tmprod(permute(J_grad,[2,1,3]),J,1);
	% mat_mult = kdlJacToArma(J)*reshape( arma::mat(permDerivJ.memptr(), permDerivJ.n_elem, 1, false), columns, columns*rows);
	% dJtdq_J = arma::cube(mat_mult.memptr(), rows, rows, columns);
end

%%%%%%%%%%%%%%%%%%
function J_grad = compute_joint_derivative_Jacobian(J)
	% Compute the Jacobian derivative w.r.t joint angles (hybrid Jacobian
	% representation). Ref: H. Bruyninck and J. de Schutter, 1996
	nb_rows = size(J,1); % task space dim.
	nb_cols = size(J,2); % joint space dim.
	J_grad = zeros(nb_rows, nb_cols, nb_cols);
	for i = 1:nb_cols
		for j = 1:nb_cols
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
function U = logmap(X,S)
	% Logarithm map (SPD manifold)
	N = size(X,3);
	for n = 1:N
		% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
		% 	tic
		% 	U(:,:,n) = S * logm(S \ X(:,:,n));
		% 	toc
		% 	tic
		[V,D] = eig(S \ X(:,:,n));
		U(:,:,n) = S * V * diag(log(diag(d))) / V;
		% 	toc
	end
end

%%%%%%%%%%%%%%%%%%
function v = symmat2vec(M)
	% Vectorization of a symmetric matrix
	N = size(M,1);
	v = diag(M);
	for n = 1:N-1
		v = [v; sqrt(2) .* diag(M,n)]; % Mandel notation
	end
end

%%%%%%%%%%%%%%%%%%
function [S,iperm] = tmprod(T,U,mode)
	% Mode-n tensor-matrix product
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
