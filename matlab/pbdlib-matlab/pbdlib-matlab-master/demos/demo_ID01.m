function demo_ID01
% Inverse dynamics with dynamically consistent nullspace (computed torque control). 
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
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
disp('This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kP = 0; %Stiffness in task space
kV = (2*kP)^.5; %Damping in task space
% kV = 10; %Damping in task space
kPq = 10; %Stiffness in joint space
kVq = 1; %Damping in joint space
dt = 1E-2; %Time step
nbData = 100;

%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDOFs = 2; %Number of articulations

model.l = 0.3; %Length of each link
model.m = 1.7; %Mass
model.g = 9.81; %Acceleration due to gravity

for k=1:nbDOFs
	lnk(k).L = Link([0 0 model.l 0], 'standard'); %Links with dh = [THETA D A ALPHA SIGMA OFFSET]
	lnk(k).L.m = model.m; %Mass
	lnk(k).L.r = [-model.l/2 0 0]; %Link COG (center of rod)
	lnk(k).L.I = [0; 0; model.m * model.l^2 / 12]; %Inertia at link COG (rod of length L and mass m, https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
	lnk(k).L.Jm = 0; %1E6;
	lnk(k).L.G = 0;
	lnk(k).L.B = 0;
	strTmp(k) = lnk(k).L;
	lnk(k).rob = SerialLink(strTmp);
	lnk(k).rob.gravity = [0 model.g 0]; %for 2D robot
end
robot = lnk(end).rob;

%Initial pose
q0(:,1) = ones(nbDOFs,1)*pi/4;
Htmp = robot.fkine(q0);
x0 = Htmp.t(1:2);


%% Computed torque control with nullspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = q0;
xh = x0 + [0;0];
dxh = [0;0];
ddxh = [0;0];
qh = q0 + ones(nbDOFs,1)*0.8;
dx = zeros(2,1);
dq = zeros(nbDOFs,1);
for t=1:nbData
	r.q(:,t) = q; %Log data
	
	Htmp = robot.fkine(q);
	x = Htmp.t(1:2);
	J = robot.jacob0(q,'trans');
	J = J(1:2,:);
	
	uG = robot.gravload(q')'; %Compute torque due to gravity 
% 	uG = cos(q) * model.l * model.m * model.g / 2; %Compute torque due to gravity 

	%Dynamically consistent nullspace (Jinv=pinv(J) would not work)
	M = robot.inertia(q') %Joint inertia
% 	M = model.m * model.l^2 / 3; %Inertia at link frame origin (rod of length L and mass m, https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
	
	Mx = inv(J / M * J' + eye(2)*1E-12); %Cartesian inertia
	W = inv(M);
	%W = eye(nbDOFs);
	
	Jinv = W * J' / (J*W*J' + eye(2)*1E-12); %Generalized inverse (weighted pseudoinverse)
	N = eye(nbDOFs) - J'*Jinv'; %Nullspace projection matrix (dynamically consistent)
	%N = (eye(nbDOFs) - Jinv*J)'; %Same as the line above but with Roy Featherston's paper notation
	%N = eye(nbDOFs) - pinv(J)*J; %Nullspace projection matrix (for standard least squares solution)
	%N = eye(nbDOFs) - J'*pinv(J)'; %Same as the line above (because of symmetry)
	
	%Computed torque control command
% 	u = J'* Mx * (ddxh + kP*(xh - x) + kV*(dxh - dx)) + N * M *(kPq * (qh - q) - kVq * dq) + uG;
% 	u = J'* Mx * (kP * (xh - x) - kV * dx) + N * (kPq * (qh - q) - kVq * dq) + uG;
% 	u = J' * (kP * (xh - x) - kV * dx) + N * (kPq * (qh - q) - kVq * dq) + uG;
% 	u = N * (kPq * (qh - q) - kVq * dq) + uG;
% 	u = uG; 
	u = rand(nbDOFs,1) * 1E-1 + uG;

% 	robot.inertia(q')
% 	uG = robot.gravload(q')'

% 	ddq = robot.accel(q', dq', u'); %Compute acceleration
	ddq = accel(q, u, model);

% 	ddq = (u - uG) / M; %Compute acceleration
	
	dq = dq + ddq*dt;
	q = q + dq*dt;
	
	dx = J * dq;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,1800,1200],'color',[1 1 1]); hold on; axis off;
for t=round(linspace(1,nbData,10))
	plotArm(r.q(:,t), ones(nbDOFs,1)*model.l, [0;0;t*0.1], .02, [.7 .7 .7]-.7*t/nbData);
end
plot(xh(1),xh(2),'r+');
axis equal; %axis([-.2 1 -.2 1]); 

%print('-dpng','demoID01.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%
%Rollout
function ddq = accel(q, u, model)
	for n=1:size(q,1)
		M = model.m * model.l^2 / 3 %Inertia at link frame origin (rod of length L and mass m, https://en.wikipedia.org/wiki/List_of_moments_of_inertia)
		uG = cos(q(n)) * model.l * model.m * model.g / 2; %Compute torque due to gravity 
		ddq(n) = (u(n) - uG) / M;
	end
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
