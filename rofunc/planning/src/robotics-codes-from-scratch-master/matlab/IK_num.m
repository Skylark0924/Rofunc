%%    Inverse kinematics with numerical computation for a planar manipulator
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Sylvain Calinon <https://calinon.ch>
%%
%%    This file is part of RCFS.
%%
%%    RCFS is free software: you can redistribute it and/or modify
%%    it under the terms of the GNU General Public License version 3 as
%%    published by the Free Software Foundation.
%%
%%    RCFS is distributed in the hope that it will be useful,
%%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%%    GNU General Public License for more details.
%%
%%    You should have received a copy of the GNU General Public License
%%    along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function IK_num

T = 50; %Number of datapoints
D = 3; %State space dimension (x1,x2,x3)
l = [2; 2; 1]; %Robot links lengths
fh = [-2; 1]; %Desired target for the end-effector
x = ones(D,1) * pi/D; %Initial robot pose

h = figure; hold on; axis off equal;
plot(fh(1,:), fh(2,:), 'r.','markersize',30); %Plot target
for t=1:T
	f = fkin0(x, l); %Forward kinematics
	J = Jkin(x, l); %Jacobian (for end-effector)
	x = x + J \ (fh - f(:,end)) * .1; %Update state 
	plot(f(1,:), f(2,:), 'k-'); %Plot robot
end

waitfor(h);
end


%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for end-effector 
function f = fkin(x, l)
	L = tril(ones(size(x,1))); %Transformation matrix
	f = [l' * cos(L * x); l' * sin(L * x)]; %Forward kinematics
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all articulations, including robot base and end-effector
function f = fkin0(x, l)
	L = tril(ones(size(x,1))); %Transformation matrix
	f = [L * diag(l) * cos(L * x), L * diag(l) * sin(L * x)]'; %Forward kinematics for links endpoints
	f = [zeros(2,1), f]; %Inclusion of robot base
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with numerical computation
function J = Jkin(x, l)
	e = 1E-6;
	D = size(x,1);
	
%	%Slow for-loop computation
%	J = zeros(2,D);
%	for n=1:D
%		xtmp = x;
%		xtmp(n) = xtmp(n) + e;
%		ftmp = fkin(xtmp, l); %Forward kinematics 
%		f = fkin(x, l); %Forward kinematics 
%		J(:,n) = (ftmp - f) / e; %Jacobian for end-effector
%	end
	
	%Fast matrix computation
	X = repmat(x, [1, D]);
	F1 = fkin(X, l);
	F2 = fkin(X + eye(D) * e, l);
	J = (F2 - F1) / e;
end
