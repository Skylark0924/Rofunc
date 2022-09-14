function demo_IK02
% Inverse kinematics with nullspace projection operator (see also demo_IK_nullspace01.m).
%
% If this code is useful for your research, please cite the related publication:
% @article{Silverio19TRO,
% 	author="Silv\'erio, J. and Calinon, S. and Rozo, L. and Caldwell, D. G.",
% 	title="Learning Task Priorities from Demonstrations",
% 	journal="{IEEE} Trans. on Robotics",
% 	year="2019",
% 	volume="35",
% 	number="1",
% 	pages="78--94",
% 	doi="10.1109/TRO.2018.2878355"
% }
%
% The commented parts of this demo require the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox if you uncomment these parts.
%
% Copyright (c) 2020 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dt = 0.01; %Time step
nbData = 100;


%% Robot parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDOFs = 3; %Number of articulations
model.l = 0.6; %Length of each link
% L1 = Link('d', 0, 'a', model.l, 'alpha', 0);
% robot = SerialLink(repmat(L1,nbDOFs,1));

q(:,1) = [pi/2 pi/2 pi/3]; %Initial pose


%% IK with nullspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dxh = [0; 0];
%dxh = 0;

%dqh = ones(nbDOFs,1)*1;
%dqh = [-1; zeros(nbDOFs-1,1)];

for t=1:nbData
	r.q(:,t) = q; %Log data
% 	Htmp = robot.fkine(q);
% 	r.x(:,t) = Htmp.t(1:2);
	r.x(:,t) = fkine(q, model); %Log data
	
% 	J = robot.jacob0(q,'trans');
% 	J = J(1:2,:);
	J = jacob0(q, model);
	
% 	pinvJ = pinv(J);
% 	pinvJ = J' / ( J * J');
% 	pinvJ = (J' * J) \ J';
	[U,S,V] = svd(J);
	S(S>0) = S(S>0).^-1;
	pinvJ = V*S'*U';
	
	%Kx = J*Kq*J'; % Stiffness at the end-effector
	%robot.maniplty(q) %Manipulability index
	
% 	%Nullspace projection matrix
% 	N = eye(nbDOFs) - pinvJ*J;
% 	%N = eye(nbDOFs) - J'*pinv(J)'

	
% 	%Alternative way of computing the nullspace projection matrix, see e.g. wiki page on svd 
% 	%(https://en.wikipedia.org/wiki/Singular-value_decomposition) or
% 	%http://math.stackexchange.com/questions/421813/projection-matrix-onto-null-space
% 	%[U,S,V] = svd(J);
	sp = (sum(S,1)<1E-1); %Span of zero rows
	N = V(:,sp) * V(:,sp)';
	
% 	[U,S,V] = svd(pinvJ);
% 	sp = (sum(S,2)<1E-1); %Span of zero rows
% 	N = U(:,sp) * U(:,sp)'; %N = U * [0 0 0; 0 0 0; 0 0 1] * U'
% % 	d = zeros(nbDOFs,1);
% % 	d(sp) = 1;
% % 	N = U * diag(d) * U';
	
% 	dqh = [5*(pi/4-q(1)); zeros(nbDOFs-1,1)];
	dqh = ones(nbDOFs,1);
	
	dq = pinvJ * dxh + N * dqh;
	q = q + dq * dt;
end

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,1000,650],'color',[1 1 1]); hold on; axis off;
%plotopt = robot.plot({'noraise', 'nobase', 'noshadow', 'nowrist', 'nojaxes'});
for t=round(linspace(1,nbData,10))
	%robot.plot(r.q(:,t)', plotopt);
	colTmp = [1,1,1] - [.7,.7,.7] * t/nbData;
	plotArm(r.q(:,t), ones(nbDOFs,1)*model.l, [0;0;t*0.1], .02, colTmp);
end
axis equal; axis tight;

%print('-dpng','graphs/demoIK02.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics
function [x, Tf] = fkine(q, model)
	Tf = eye(4);
	T = repmat(Tf, [1,1,size(q,1)]);
	for n=1:size(q,1)
		c = cos(q(n));
		s = sin(q(n));
		T(:,:,n) = [c, -s, 0, model.l * c; ...
								s, c, 0, model.l * s; ...
								0, 0, 1, 0; ...
								0, 0, 0, 1]; %Homogeneous matrix 
		Tf = Tf * T(:,:,n);
	end
	x = Tf(1:2,end);
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with numerical computation
function J = jacob0(q, model)
	e = 1E-6;
	J = zeros(2,size(q,1));
	for n=1:size(q,1)
		qtmp = q;
		qtmp(n) = qtmp(n) + e;
		J(:,n) = (fkine(qtmp, model) - fkine(q, model)) / e;
	end
end