function demo_IK_quat02
% Inverse kinematics for orientation tracking with unit quaternions
% (kinematic model of WAM 7-DOF arm or an iCub 7-DOF arm)
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by João Silvério and Sylvain Calinon
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
dt = 1E-2; %Time step
nbData = 100; %Number of datapoints
Kp = 2E-1; %Tracking gain


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Robot parameters (built from function that contains WAM D-H parameters)
%[robot,L] = initWAMstructure('WAM');
[robot, link] = initiCubstructure('iCub');
nbDOFs = robot.n; %Number of articulations

%Initial pose
q0 = zeros(nbDOFs,1);

%Set desired orientation
% Qh = UnitQuaternion([0 0 1 0]); % Aligned with the basis frame
qh = ones(nbDOFs,1) * 2E-1;
Htmp = robot.fkine(qh);
Qh = UnitQuaternion(Htmp);


%% IK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = q0;
for t=1:nbData
	r.q(:,t) = q; %Log data
	Htmp = robot.fkine(q);
	Q = UnitQuaternion(Htmp);
	%Keep quaternion with closest difference
	if(Qh.inner(Q)<0)
		Q = Quaternion(-Q.double);
	end
	
	Jw = robot.jacob0(q,'rot');
	
% 	%The new Jacobian is built from the geometric Jacobian to allow the tracking of quaternion velocities in Cartesian space	
% 	J = 0.5 * QuatToMatrix(Q) * [zeros(1,nbDOFs); Jw];
% 	
% 	%Compute the quaternion derivative
% 	%-> The quaternion derivative at time t is given by:
% 	%dq(t) = (1/2) * q(t) * w(t), where w(t) is the angular velocity
% 	
% 	w = Kp * 2 * quaternionLog(Qh * Q.inv); % first compute angular velocity
% 	dQh = 0.5 * Q * UnitQuaternion([0 w]); % quaternion derivative
% 	
% 	r.Q(t,:) = Q.double; %Log the quaternion at each instant
% 	
% 	dq = pinv(J) * dQh.double'; % now with a quaternion derivative reference
% 	%N = eye(nbDOFs) - pinv(J)*J; %Nullspace
% 	%dq = pinv(J) * dQh.double' + N * [1; 0; 0; 0; 0];
% 	%dq = N * [-1; -1; 0; 0; 1];
	

	%The two functions below return the same dx
	dx = 4 * omegaConversionMatrix(Q.double') * logmap(Qh.double', Q.double') / dt;
% 	dx = angDiffFromQuat(Qh.double', Q.double') / dt;	
	
	dq = pinv(Jw) * dx * Kp;
	q = q + dq * dt;
end
Qh.double'
Q.double'


%% Plots
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,1000,650]); hold on; rotate3d on;
plot3Dframe(eye(3)*5E-2, zeros(3,1));

% %Plot animation of the movement (with robotics toolbox functionalities)
% plotopt = robot.plot({'noraise', 'nobase', 'noshadow', 'nowrist', 'nojaxes'});
% robot.plot(r.q'); %'floorlevel',-0.5
for i=1:nbDOFs
	T = link(i).fkine(qh(1:i));
	posTmp(:,i+1) = T.t(1:3,end);
end
plot3(posTmp(1,:),posTmp(2,:),posTmp(3,:), '-','linewidth',2,'color',[.8 0 0]);
plot3(posTmp(1,:),posTmp(2,:),posTmp(3,:), '.','markersize',18,'color',[.8 0 0]);
	

%Plot animation of the movement (with standard plot functions)
posTmp(:,1) = zeros(3,1);
for t=round(linspace(1,nbData,10))
	colTmp = [.9,.9,.9] - [.7,.7,.7] * t/nbData;
	for i=1:nbDOFs
		T = link(i).fkine(r.q(1:i,t));
		posTmp(:,i+1) = T.t(1:3,end);
	end
	plot3(posTmp(1,:),posTmp(2,:),posTmp(3,:), '-','linewidth',2,'color',colTmp);
	plot3(posTmp(1,:),posTmp(2,:),posTmp(3,:), '.','markersize',18,'color',colTmp);
% 	plot3Dframe(T.t(1:3,1:3)*1E-1, T.t(1:3,end), min(eye(3)+colTmp(1,1),1));
end
set(gca,'xtick',[],'ytick',[],'ztick',[]);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
view(3); axis equal; axis vis3d;

% %Plot quaternion trajectories
% figure; hold on;
% h1 = plot(dt*(1:nbData),repmat(Qh.double,nbData,1));
% h2 = plot(dt*(1:nbData),r.Q,'Linewidth',3);
% title('Quaternion elements (thin line = demonstration , bold line = reproduction)');
% xlabel('t');

pause;
close all;
end

function x = expmap(u, x0)
	theta = sqrt(sum(u.^2,1)); %norm(u,'fro')
	x = real(repmat(x0,[1,size(u,2)]) .* repmat(cos(theta), [size(u,1),1]) + u .* repmat(sin(theta)./theta, [size(u,1),1]));
	x(:,theta<1e-16) = repmat(x0, [1,sum(theta<1e-16)]);	
end

function u = logmap(x, x0)
	theta = acos(x0'*x); %acos(trace(x0'*x))
	u = (x - repmat(x0, [1,size(x,2)]) .* repmat(cos(theta), [size(x,1),1])) .* repmat(theta./sin(theta), [size(x,1),1]);
	u(:,theta<1e-16) = 0;
end

function G = omegaConversionMatrix(q)
%See Basile Graf (2017) "Quaternions And Dynamics" for definition of E and G
	G = [-q(2) q(1) -q(4) q(3); ...
			 -q(3) q(4) q(1) -q(2); ...
			 -q(4) -q(3) q(2) q(1)]; %E
% 	G = [-q(2) q(1) q(4) -q(3); ...
% 			 -q(3) -q(4) q(1) q(2); ...
% 			 -q(4) q(3) -q(2) q(1)]; 
end