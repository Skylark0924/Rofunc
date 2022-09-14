function demo_IK_weighted01
% Inverse kinematics with nullspace control, by considering weights in both joint space and in task space.
% (see also demo_IK_nullspaceAsProduct01.m)
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% If this code is useful for your research, please cite the related publication:
% @article{Girgin19,
% 	author="Girgin, H. and Calinon, S.",
% 	title="Nullspace Structure in Model Predictive Control",
% 	journal="arXiv:1905.09679",
% 	year="2019",
% 	pages="1--16"
% }
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDOFs = 5; %Nb of degrees of freedom
dt = 0.1; %Time step
nbData = 200; %Number of datapoints
% Wx = eye(2);
% Wq = eye(nbDOFs);
Wx = diag([1, 1E-44]); %Weight in operational space (-> we don't care about tracking x2 precisely)
Wq = diag([1E-44, ones(1,nbDOFs-1)]); %Weight in configuration space (-> we don't want to move the first joint)
% invWq = diag([1, zeros(1,nbDOFs-1)]); 


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.l = 0.3;

L1 = Link('d', 0, 'a', model.l, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
robot2 = SerialLink(repmat(L1,3,1));

q0(:,1) = [pi/4; zeros(nbDOFs-1,1)]; %Initial pose


%% Weighted IK with nullspace as product of Gaussians (PoG)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vq = sqrtm(Wq);
% invVq = sqrtm(invWq);
invVq = inv(Vq + eye(nbDOFs)*1E-6);
Vx = sqrtm(Wx);
q = q0;
xh = [-0.2; 0.8];
xh2 = [0; 0.4];
for t=1:nbData
	r(1).q(:,t) = q; %Log data
	
	Htmp = robot.fkine(q);
	x = Htmp.t(1:2);
	Htmp2 = robot2.fkine(q(1:3));
	x2 = Htmp2.t(1:2);
	J = robot.jacob0(q,'trans');
	J = J(1:2,:);

% 	x = fkine(q, model);
% 	x2 = fkine(q(1:3), model);
% 	J = jacob0(q, model);
	
	J2 = [J(1:2,1:3), zeros(2, nbDOFs-3)];	
	Jw = Vx * J * Vq;
	pinvJw = pinv(Jw);
	N = eye(nbDOFs) - pinvJw * Jw;
	
% 	%Standard PoG computation
% 	Mu(:,1) = Vq * pinvJw * Vx * 2 * (xh - x);
% 	Mu(:,2) = Vq * pinv(J2) * 10 * (xh2 - x2);
% 	Q(:,:,1) = invVq * pinvJw * Jw * invVq;
% 	Q(:,:,2) = invVq * N * invVq;
% 	dq = (Q(:,:,1) + Q(:,:,2)) \ (Q(:,:,1) * Mu(:,1) + Q(:,:,2) * Mu(:,2)); %Fusion of controllers through product of Gaussians

	%Numerically robust PoG computation
	QMu(:,1) = invVq * pinvJw * (Jw * pinvJw) * Vx * 2 * (xh - x);
	QMu(:,2) = invVq * N * pinv(J2) * 10 * (xh2 - x2);
	Q(:,:,1) = invVq * pinvJw * Jw * invVq;
	Q(:,:,2) = invVq * N * invVq;
	dq = (Q(:,:,1) + Q(:,:,2)) \ (QMu(:,1) + QMu(:,2)); %Fusion of controllers through product of Gaussians
	
	q = q + dq * dt;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,2000,1300],'color',[1 1 1]); hold on; axis off;
h(1) = plot3(xh(1), xh(2), 11, '.','markersize',25,'color',[.4 1 .4]);
h(2) = plot3(xh2(1), xh2(2), 11, '.','markersize',25,'color',[.4 .4 1]);
htmp = plotArm(q0, ones(nbDOFs,1)*model.l, [0;0;1], .02, [.8,.8,.8]);
h(3) = htmp(1);
ii = 2;
for t=nbData:nbData %round(linspace(1,nbData,2))
	colTmp = [1,1,1] - [.7,.7,.7] * t/nbData;
	htmp = plotArm(r(1).q(:,t), ones(nbDOFs,1)*model.l, [0;0;ii], .02, colTmp);
	ii = ii + 1;
end
h(4) = htmp(1);
h(5) = plot3([xh(1),xh(1)], [xh(2)-.3,xh(2)+.3], [10,10], '-','linewidth',2,'color',[.8 0 0]);
plotArm(q0(1), model.l, [0;0;10], .02, [.8,0,0]);
axis equal; axis tight;
legend(h,{'Primary task (tracking with endeffector point)','Secondary task (tracking with other point on the kinematic chain)','Initial pose','Weighted IK solution','Weights (in joint space and task space)'},'location','southoutside','fontsize',12);

% figure; hold on;
% plot(r(2).q(1,:),'k-');
% plot(r(2).q(2,:),'r-');
% xlabel('t'); ylabel('q_1,q_2');

% print('-dpng','graphs/demoIK_weighted01.png');
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
								0, 0, 1, 0;
								0, 0, 0, 1]; %Homogeneous matrix 
		Tf = Tf * T(:,:,n);
	end
	x = Tf(1:2,end);
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with numerical computation
function J = jacob0(q, model)
	e = 1E-3;
	J = zeros(2,size(q,1));
	for n=1:size(q,1)
		qtmp = q;
		qtmp(n) = qtmp(n) + e;
		J(:,n) = (fkine(qtmp, model) - fkine(q, model)) / e;
	end
end