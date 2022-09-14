function demo_Riemannian_SE2_interp01
% Interpolation on SE(2) manifold 
% (Implementation of exp and log maps based on "Lie Groups for 2D and 3D Transformations" by Ethan Eade)
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Sylvain Calinon, https://calinon.ch/
% 
% This file is part of PbDlib, https://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <https://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbVarPos = 2; %Number of variables [x1,x2]
model.nbVar = model.nbVarPos+1; %Number of variables [w,x1,x2]
nbData = 40; %Number of interpolation points


%% Generate random homogeneous matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(model.nbVar,model.nbVar,2);
for n=1:2
	v = rand(model.nbVarPos, 1) * 1E2;
	w = rand(1) * pi;
	R = [cos(w) -sin(w); sin(w) cos(w)];
	x(:,:,n) = [R, v; 0 0 1];
end


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = linspace(0,1,nbData); %standard interpolation
% w = linspace(-.5,1.5,nbData); %extrapolation (exaggeration)

% wi = [linspace(1,0,nbData); linspace(0,1,nbData)]; %standard interpolation
% % wi = [linspace(1.5,-.5,nbData); linspace(-.5,1.5,nbData)]; %extrapolation (exaggeration)
% S = x(:,:,1);

x2 = zeros(model.nbVar, model.nbVar, nbData);
x2(:,:,1) = x(:,:,1);
for t=2:nbData		
% 	%Interpolation between more than 2 points can be computed in an iterative form
% 	nbIter = 10; %Number of iterations for the convergence of Riemannian estimate
% 	for n=1:nbIter
% 		u = zeros(model.nbVar,1);
% 		for j=1:2
% 			u = u + wi(j,t) * logmap(x(:,:,j), S);
% 		end
% 		S = expmap(u,S);
% 	end
% 	x2(:,:,t) = S;
		
	%Interpolation between two points can be computed in closed form
	x2(:,:,t) = expmap(w(t)*logmap(x(:,:,2), x(:,:,1)), x(:,:,1));
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1200,800]); hold on; axis off;
plot2Dframe(x(1:2,1:2,:)*5E-2, x(1:2,end,:), eye(3)*.8, 4);
axis equal; 
% print('-dpng','graphs/demo_Riemannian_SE2_interp01a.png');

plot(squeeze(x2(1,end,:)), squeeze(x2(2,end,:)), '-','linewidth',2,'color',[0 0 0]);
plot2Dframe(x2(1:2,1:2,:)*5E-0, x2(1:2,end,:), min(eye(3)+.6, 1));
plot2Dframe(x(1:2,1:2,:)*5E-0, x(1:2,end,:), eye(3)*.8, 4);
% print('-dpng','graphs/demo_Riemannian_SE2_interp01b.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(u,S)
	for n=1:size(u,2)
		w = u(1,n);
		v = u(2:3,n);

		%Rotation part
		R = [cos(w) -sin(w); sin(w) cos(w)];

		%Translation part
		V = [sin(w), -(1-cos(w)); 1-cos(w), sin(w)] .* (1/w);
		t = V * v;

		X(:,:,n) = S * [R, t; 0 0 1];
	end
end

function u = logmap(X,S)
	for n=1:size(X,3)
		invS = [S(1:2,1:2)', -S(1:2,1:2)' * S(1:2,end); S(end,:)];
		H = invS * X(:,:,n);

		%Rotation part
		%Htmp = -logm(H(1:2,1:2)); w = Htmp(1,2);  %implementation more efficient?
		w = atan2(H(2,1), H(1,1));

		%Translation part
		a = sin(w)/w;
		b = (1-cos(w))/w;
		invV = [a, b; -b, a] .* 1./(a^2+b^2);
		v = invV * H(1:2,end);

		u(:,n) = [w; v];
	end
end