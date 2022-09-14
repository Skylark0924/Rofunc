function demo_IK_quat01
% Inverse kinematics for orientation data represented as unit quaternions. 
% (see also demo_IK_orient01.m)
%
% Copyright (c) 2020 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
%
% The commented parts of this demo require the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox if you uncomment these parts.
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
nbData = 100; %Number of points in a trajectory
Kp = 1E-1; %Tracking gain

nbDOFs = 5; %Number of articulations
model.l = 0.2; %Length of links

q0 = rand(nbDOFs,1) * pi/2; %Initial pose
% q0 = [pi/5; pi/5; pi/5; pi/8; pi/8]; %Initial pose

% L1 = Link('d', 0, 'a', model.l, 'alpha', 0);
% robot = SerialLink(repmat(L1,nbDOFs,1));
% %Set desired pose (pointing vertically)
% Htmp = robot.fkine(ones(1,nbDOFs)*pi/5); 
% Qh = UnitQuaternion(Htmp) % Q denotes quaternion, q denotes joint position
% % Qh = Quaternion(-pi/4, [0 0 1]); % -> Quaternion(angle,vector)

[~, Qh] = fkine(ones(nbDOFs,1)*pi/5, model);


%% IK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q = q0;
for t=1:nbData
	r.q(:,t) = q; %Log data
	
% 	Htmp = robot.fkine(q);
% 	Q = UnitQuaternion(Htmp);
% 	J = robot.jacob0(q,'rot');
% 	dx = angDiffFromQuat(Qh, Q) / dt
% 	
% 	u = logmap(Qh.double', Q.double');
% 	dx = 4 * omegaConversionMatrix(Q.double') * u / dt

	[~, Q] = fkine(q, model);
	%Keep quaternion with closest difference
	if(Qh'*Q<0)
		Q = -Q;
	end

	dx = 4 * omegaConversionMatrix(Q) * logmap(Qh, Q) / dt;
	
	J = [zeros(2,nbDOFs); ones(1, nbDOFs)]; %Jacobian for orientation data (corresponds to a constant matrix for planar robots)
% 	[~, J] = jacob0(q, model);

	dq = pinv(J) * Kp * dx;
	q = q + dq * dt;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,1000,650]); hold on; axis off;
%plotopt = robot.plot({'noraise', 'nobase', 'noshadow', 'nowrist', 'nojaxes'});
for t=round(linspace(1,nbData,2))
	colTmp = [.8,.8,.8] - [.7,.7,.7] * t/nbData;
	plotArm(r.q(:,t), ones(nbDOFs,1)*model.l, [0;0;t*0.1], .02, colTmp);
	%robot.plot(r.q(:,t)', plotopt);
end
axis([-.2 1 -0.4 1]); axis equal;

%print('-dpng','graphs/demoIK_quat02.png');
pause;
close all;
end

function x = expmap(u, x0)
	theta = sqrt(sum(u.^2,1)); %norm(u,'fro')
	x = real(repmat(x0,[1,size(u,2)]) .* repmat(cos(theta),[size(u,1),1]) + u .* repmat(sin(theta)./theta,[size(u,1),1]));
	x(:,theta<1e-16) = repmat(x0,[1,sum(theta<1e-16)]);	
end

function u = logmap(x, x0)
	theta = acos(x0'*x); %acos(trace(x0'*x))
	u = (x - repmat(x0,[1,size(x,2)]) .* repmat(cos(theta),[size(x,1),1])) .* repmat(theta./sin(theta),[size(x,1),1]);
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

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics
function [xp, xo, Tf] = fkine(q, model)
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
	xp = Tf(1:2,end); %Position
	xo = atan2(Tf(2,1), Tf(1,1)); %Orientation (as single Euler angle for planar robot)
	xo = [cos(xo/2); [0;0;1] * sin(xo/2)]; %Orientation (as unit quaternion)
% 	xo = rotm2quat(Tf(1:3,1:3)); %Orientation (as unit quaternion)
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with numerical computation
function [Jp, Jo] = jacob0(q, model)
	e = 1E-4;
	Jp = zeros(2, size(q,1)); %Jacobian for position data
	Jo = [zeros(2, size(q,1)); ones(1, size(q,1))]; %Jacobian for orientation data (corresponds to a vector with 1 elements for planar robots)
	for n=1:size(q,1)
		qtmp = q;
		qtmp(n) = qtmp(n) + e;
		Jp(:,n) = (fkine(qtmp, model) - fkine(q, model)) / e;
% 		[xp2, xo2] = fkine(qtmp, model);
% 		[xp1, xo1] = fkine(q, model);
% 		Jp(:,n) = (xp2 - xp1) / e;
% 		Jo(:,n) = (xo2 - xo1) / e;
	end
end