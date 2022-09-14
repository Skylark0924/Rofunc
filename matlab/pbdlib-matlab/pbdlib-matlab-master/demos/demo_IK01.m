function demo_IK01
% Basic forward and inverse kinematics for a planar robot, with numerical Jacobian computation.
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
dt = 1E-2; %Time step
nbData = 100;


%% Robot parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbDOFs = 3; %Number of articulations
model.l = 0.6; %Length of each link
q(:,1) = [pi/2; pi/2; pi/3]; %Initial pose
% q = rand(model.nbDOFs,1);


%% IK 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dxh = [.1; 0];
for t=1:nbData-1
	J = jacob0(q(:,t), model); %Jacobian
	dq = pinv(J) * dxh;
% 	dq = J'/(J*J') * dxh;
	q(:,t+1) = q(:,t) + dq * dt;
	x(:,t+1) = fkine(q(:,t+1), model); %FK for a planar robot
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[20,10,800,650],'color',[1 1 1]); hold on; axis off;
for t=floor(linspace(1,nbData,2))
	colTmp = [.7,.7,.7] - [.7,.7,.7] * t/nbData;
	plotArm(q(:,t), ones(model.nbDOFs,1)*model.l, [0;0;-2+t/nbData], .03, colTmp);
% 	plot(x(1,t), x(2,t), '.','markersize',50,'color',[.8 0 0]);
end
plot(x(1,:), x(2,:), '.','markersize',20,'color',[0 0 0]);
axis equal; axis tight;

%print('-dpng','graphs/demoIK01.png');
pause;
close all;
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics
function [x, Tf] = fkine(q, model)
	Tf = eye(4);
	T = repmat(Tf, [1,1,model.nbDOFs]);
	for n=1:model.nbDOFs
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
	J = zeros(2,model.nbDOFs);
	for n=1:model.nbDOFs
		qtmp = q;
		qtmp(n) = qtmp(n) + e;
		J(:,n) = (fkine(qtmp, model) - fkine(q, model)) / e;
	end
end
