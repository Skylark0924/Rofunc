function demo_IK_nullspaceAsProduct01
% 3-level nullspace control formulated as product of Gaussians (PoG).
% (see also demo_IK_weighted01.m)
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
nbDOFs = 6; %Nb of degrees of freedom
dt = 0.1; %Time step
nbData = 200; %Number of datapoints


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
armLength = 0.3;
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot1 = SerialLink(repmat(L1,nbDOFs,1));
robot2 = SerialLink(repmat(L1,nbDOFs-2,1));
robot3 = SerialLink(repmat(L1,nbDOFs-4,1));
q0(:,1) = [pi/4; zeros(nbDOFs-1,1)]; %Initial pose


%% IK with nullspace as product of Gaussians
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = q0;
xh1 = [-.2; .8];
xh2 = [0; .4];
xh3 = [.2; .3];
for t=1:nbData
	r(1).q(:,t) = q; %Log data
	J = robot1.jacob0(q,'trans');
	J1 = J(1:2,:);
	J2 = [J(1:2,1:nbDOFs-2), zeros(2,2)];
	J3 = [J(1:2,1:nbDOFs-4), zeros(2,4)];
	Htmp = robot1.fkine(q);
	x1 = Htmp.t(1:2);
	Htmp2 = robot2.fkine(q(1:nbDOFs-2));
	x2 = Htmp2.t(1:2);
	Htmp3 = robot3.fkine(q(1:nbDOFs-4));
	x3 = Htmp3.t(1:2);
	
	pinvJ1 = pinv(J1);
	pinvJ2 = pinv(J2);
	pinvJ3 = pinv(J3);
	N1 = eye(nbDOFs) - pinvJ1 * J1; 
	N2 = eye(nbDOFs) - pinvJ2 * J2; 
	N3 = eye(nbDOFs) - pinvJ3 * J3; 
	
	dx1 = 10 * (xh1 - x1);
	dx2 = 10 * (xh2 - x2);
	dx3 = 10 * (xh3 - x3);
	
% 	%Standard nullspace control computation
% 	dq = pinvJ1 * dx1 + N1 * pinvJ2 * dx2 + N1 * N2 * pinvJ3 * dx3;
	
	%Nullspace control computation as product of Gaussians
	Mu(:,1) = pinvJ1 * dx1;
	Mu(:,2) = pinv(J2*N1) * dx2;
	Mu(:,3) = pinv(N2*N1) * pinvJ3 * dx3;
	Q(:,:,1) = pinvJ1 * J1;
	Q(:,:,2) = N1 * pinvJ2 * J2 * N1;
	Q(:,:,3) = N1 * N2 * N1;
	dq = (Q(:,:,1) + Q(:,:,2) + Q(:,:,3)) \ (Q(:,:,1) * Mu(:,1) + Q(:,:,2) * Mu(:,2) + Q(:,:,3) * Mu(:,3)); %Fusion of controllers with product of Gaussians
	
	q = q + dq * dt;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,1000,650],'color',[1 1 1]); hold on; axis off;
h(1) = plot3(xh1(1), xh1(2), 11, '.','markersize',25,'color',[.4 1 .4]);
h(2) = plot3(xh2(1), xh2(2), 11, '.','markersize',25,'color',[.4 .4 1]);
h(3) = plot3(xh3(1), xh3(2), 11, '.','markersize',25,'color',[.8 0 0]);
htmp = plotArm(q0, ones(nbDOFs,1)*armLength, [0;0;1], .02, [.8,.8,.8]);
h(4) = htmp(1);
ii = 2;
for t=nbData:nbData %round(linspace(1,nbData,2))
	colTmp = [1,1,1] - [.7,.7,.7] * t/nbData;
	htmp = plotArm(r(1).q(:,t), ones(nbDOFs,1)*armLength, [0;0;ii], .02, colTmp);
	ii = ii + 1;
end
h(5) = htmp(1);
axis equal; axis tight;
legend(h,{'Primary task','Secondary task','Third task','Initial pose','IK'},'location','southeast','fontsize',18);

% figure; hold on;
% plot(r(2).q(1,:),'k-');
% plot(r(2).q(2,:),'r-');
% xlabel('t'); ylabel('q_1,q_2');

% print('-dpng','graphs/demoIK_nullspaceAsProduct01.png');
pause;
close all;