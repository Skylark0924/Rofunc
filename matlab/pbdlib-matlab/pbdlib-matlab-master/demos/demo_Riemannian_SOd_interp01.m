function demo_Riemannian_SOd_interp01
% Interpolation on SO(d) manifold 
% (implementation of exp and log maps based on "Newton method for Riemannian centroid computation in naturally reductive homogeneous spaces")
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
model.nbVar = 2; %Number of variables 
model.nbStates = 1; %Number of states in the GMM
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 30; %Number of iteration for the EM algorithm
nbData = 10; %Number of datapoints


%% Generate random rotation matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = zeros(model.nbVar, model.nbVar, 2);
for t=1:2
	w = rand(1) .* pi;
	x(:,:,t) = [cos(w) -sin(w); sin(w) cos(w)];
end
u = logmap(x, eye(model.nbVar));


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
figure('position',[10,10,1650,1250]); hold on; axis off;
plot2Dframe(x2, [linspace(0,10,nbData); zeros(1,nbData)], min(eye(3)+.6,1));
plot2Dframe(x, [linspace(0,10,2); zeros(1,2)], eye(3)*.8, 6);
axis equal; 
% print('-dpng','graphs/demo_Riemannian_S0d_interp01.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U, S)
	for n=1:size(U,3)
		X(:,:,n) = S * expm(S' * U(:,:,n));
	end
end

function U = logmap(X, S)
	for n=1:size(X,3)
		U(:,:,n) = S * logm(S' * X(:,:,n));
	end
end

function Y2 = transp(X, Y, S)
	Y2 = S * expm(S'*X/2) * S' * Y * expm(S'*X/2);
end