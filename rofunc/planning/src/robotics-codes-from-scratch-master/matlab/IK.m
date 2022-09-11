%%    Inverse kinematics for a planar manipulator
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

T = 50; %Number of datapoints
D = 3; %State space dimension (x1,x2,x3)
l = [2; 2; 1]; %Robot links lengths
fh = [-2; 1]; %Desired target for the end-effector
x = ones(D,1) * pi/D; %Initial robot pose
L = tril(ones(D)); %Transformation matrix

h = figure; hold on; axis off equal;
plot(fh(1,:), fh(2,:), 'r.','markersize',30); %Plot target
for t=1:T
	f = [L * diag(l) * cos(L * x), L * diag(l) * sin(L * x)]'; %Forward kinematics (for all articulations, including end-effector)
	J = [-sin(L * x)' * diag(l) * L; cos(L * x)' * diag(l) * L]; %Jacobian (for end-effector)
	x = x + J \ (fh - f(:,end)) * .1; %Update state 
	plot([0,f(1,:)], [0,f(2,:)], 'k-'); %Plot robot
end
waitfor(h);
