function demo_Riemannian_Sd_interp02
% Interpolation on a 3-sphere and comparison with SLERP
% (formulation with tangent space of the same dimension as the dimension of the manifold)
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
nbVar = 4; %Number of variables
nbStates = 2; %Number of states
nbData = 100; %Number of interpolation steps
% nbIter = 20; %Number of iteration for the Gauss Newton algorithm

x = rand(nbVar,nbStates) - 0.5;
for i=1:nbStates
	x(:,i) = x(:,i) / norm(x(:,i));
end


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];
xi = zeros(nbVar,nbData);
% xtmp = x(:,1);

for t=1:nbData
% 	%Interpolation between more than 2 points can be computed in an iterative form
% 	for n=1:nbIter
% 		utmp = zeros(nbVar,1);
% 		for i=1:nbStates
% 			utmp = utmp + w(i,t) * logmap(x(:,i), xtmp);
% 		end
% 		xtmp = expmap(utmp, xtmp);
% 	end
% 	xi(:,t) = xtmp;

	%Interpolation between two covariances can be computed in closed form
	xi(:,t) = expmap(w(2,t) * logmap(x(:,2),x(:,1)), x(:,1));
end


%% SLERP interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t=1:nbData
	xi2(:,t) = quatinterp(x(:,1)', x(:,2)', w(2,t), 'slerp');
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,900,1250]);
for i=1:nbVar
	subplot(nbVar,1,i); hold on;
	for n=1:nbStates
		plot([1,nbData],[x(i,n),x(i,n)],'-','linewidth',2,'color',[0 0 0]);
		plot([1,nbData],[-x(i,n),-x(i,n)],'--','linewidth',2,'color',[0 0 0]);
	end
	h(1) = plot(xi(i,:),'-','linewidth',2,'color',[.8 0 0]);
	h(2) = plot(xi2(i,:),':','linewidth',2,'color',[0 .7 0]);
	if i==1
		legend(h,'geodesic','SLERP');
	end
	ylabel(['q_' num2str(i)]);
	axis([1, nbData -1 1]);
end
xlabel('t');

%3D plot
figure('PaperPosition',[0 0 6 8],'position',[910,10,650,650]); hold on; axis off; rotate3d on;
colormap([.9 .9 .9]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
plot3Dframe(quat2rotm(x(:,end)'), zeros(3,1), eye(3)*.3);
view(3); axis equal; axis tight; axis vis3d;  
h=[];
for t=1:nbData
	delete(h);
	h = plot3Dframe(quat2rotm(xi(:,t)'), zeros(3,1));
	drawnow;
end

pause;
for t=1:nbData
	delete(h);
	h = plot3Dframe(quat2rotm(xi2(:,t)'), zeros(3,1));
	drawnow;
end

%print('-dpng','graphs/demo_Riemannian_Sd_interp02.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u,x0)
	theta = sqrt(sum(u.^2,1)); %norm(u,'fro')
	x = repmat(x0,[1,size(u,2)]) .* repmat(cos(theta),[size(u,1),1]) + u .* repmat(sin(theta)./theta,[size(u,1),1]);
	x(:,theta<1e-16) = repmat(x0,[1,sum(theta<1e-16)]);	
end

function u = logmap(x,x0)
% 	p = (x-x0) - trace(x0'*(x-x0)) * x0;
% 	if norm(p,'fro')<1e-16
% 		u = zeros(size(x));
% 	else
% 		u = acos(trace(x0'*x)) * p./(norm(p,'fro'));
% 	end
	%theta = acos(trace(x0'*x));	
		
	theta = acos(x0'*x);	
	u = (x - repmat(x0,[1,size(x,2)]) .* repmat(cos(theta),[size(x,1),1])) .* repmat(theta./sin(theta),[size(x,1),1]);
	u(:,theta<1e-16) = 0;
end

function Ac = transp(x1,x2,t)
	if nargin==2
		t=1;
	end
	u = logmap(x2,x1);
	e = norm(u,'fro');
	u = u ./ (e+realmin);
	Ac = -x1 * sin(e*t) * u' + u * cos(e*t) * u' + eye(size(u,1)) - u * u';
end

function x = geodesic(u,x0,t)
	normu = norm(u,'fro');
	x = x0 * cos(normu*t) + u./normu * sin(normu*t);
end