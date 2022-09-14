function demo_Riemannian_Sd_vecTransp01
% Parallel transport on a d-sphere
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
% Written by Noémie Jaquier, Sylvain Calinon
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


%% Generate data on a sphere
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 3;

Data = randn(3,nbData)*4E-1;
x = Data./repmat(sqrt(sum(Data.^2,1)),3,1);
t = 0:0.05:1;
nbp = 40;
[X,Y,Z] = sphere(nbp-1);


%% Follow the geodesic from x1 to x2 and x3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u12 = logmap(x(:,2),x(:,1));
u13 = logmap(x(:,3),x(:,1));
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; axis equal; rotate3d on;
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));
plot3(x(1,1), x(2,1), x(3,1),'.','markersize',20,'color', [0 1 0]);
for i=1:length(t)
	g = geodesic(u12,x(:,1),t(i));
	plot3(g(1,1), g(2,1), g(3,1),'.','markersize',12,'color', [1*(1-t(i)) 1*t(i) 0]);
end
plot3(x(1,2), x(2,2), x(3,2),'.','markersize',20,'color', [1 0 0]);
for i=1:length(t)
	g = geodesic(u13,x(:,1),t(i));
	plot3(g(1,1), g(2,1), g(3,1),'.','markersize',12,'color', [1*(1-t(i)) 0 1*t(i)]);
end
plot3(x(1,2), x(2,2), x(3,2),'.','markersize',20,'color', [0 0 1]);
axis vis3d;


%% Parallel transport of u12 from x1 to x3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p13 = transp(x(:,1),x(:,3)) * u12;

% Plot
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; axis equal; rotate3d on;
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));
% Plot x1 and its tangent space
plot3(x(1,1), x(2,1), x(3,1),'.','markersize',20,'color', [1 0 0]);
Rrot = rotM(x(:,1))';
msh = repmat(x(:,1),1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
patch(msh(1,:),msh(2,:),msh(3,:), [1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3)
% Plot x3 and its tangent space
plot3(x(1,3), x(2,3), x(3,3),'.','markersize',20,'color', [0 1 0]);
Rrot = rotM(x(:,3))';
msh = repmat(x(:,3),1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
patch(msh(1,:),msh(2,:),msh(3,:), [0 1 0],'edgecolor',[0 1 0],'facealpha',.3,'edgealpha',.3)
% Plot u12 on x1 tangent space and its parallel transported version on x3
% tangent space
plot3([x(1,1),x(1,1)+u12(1,1)],[x(2,1),x(2,1)+u12(2,1)],[x(3,1),x(3,1)+u12(3,1)], 'color', [1 0 0], 'Linewidth', 2);
plot3([x(1,3),x(1,3)+p13(1,1)],[x(2,3),x(2,3)+p13(2,1)],[x(3,3),x(3,3)+p13(3,1)], 'color', [0 1 0], 'Linewidth', 2);
axis vis3d;

%% Parallel transport along the geodesic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here the direction of the geodesic from x1 to x2 in the tangent space of
% x1 (u12) is parallel transported into the tangent space of x3, but any
% other vector in the tangent space of x1 can be transported the same way.
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; axis equal; rotate3d on;
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));
for i=1:length(t)
	g = geodesic(u13, x(:,1), t(i));
	plot3(g(1,1), g(2,1), g(3,1),'.','markersize',12,'color', [1*(1-t(i)) 1*t(i) 0]);
	Rrot = rotM(g)';
	msh = repmat(g,1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
	patch(msh(1,:),msh(2,:),msh(3,:), [1*(1-t(i)) 1*t(i) 0],'edgecolor',[1*(1-t(i)) 1*t(i) 0],'facealpha',.1,'edgealpha',.1)
	ptv = transp(x(:,1), x(:,3), t(i)) * u12;
	plot3([g(1,1),g(1,1)+ptv(1,1)],[g(2,1),g(2,1)+ptv(2,1)],[g(3,1),g(3,1)+ptv(3,1)], 'color', [1*(1-t(i)) 1*t(i) 0], 'Linewidth', 2);
	%Inner product between vector transported and direction of geodesic
	dir = logmap(x(:,3), g);
	if norm(dir) > 1E-5
		dir = dir ./ norm(dir);
		inprod(i) = dir' * ptv;
	end
end
axis vis3d;

%% Checks on parallel transportation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inner product conserved
inprod

% Those two vectors are the same :
% p1 is the direction of the geodesic from x3 to x1 in the tangent space of x3;
% p3 is the direction of the geodesic from x1 to x3 in the tangent space of x1 parallel transported to the tangent space of x3.
p1 = -logmap(x(:,1), x(:,3))
p3 = transp(x(:,1), x(:,3)) * logmap(x(:,3), x(:,1))

% Plot
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; axis equal; rotate3d on; 
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));
% Plot x1 and its tangent space
plot3(x(1,1), x(2,1), x(3,1),'.','markersize',20,'color', [1 0 0]);
Rrot = rotM(x(:,1))';
msh = repmat(x(:,1),1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
patch(msh(1,:),msh(2,:),msh(3,:), [1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3)
% Plot x3 and its tangent space
plot3(x(1,3), x(2,3), x(3,3),'.','markersize',20,'color', [0 1 0]);
Rrot = rotM(x(:,3))';
msh = repmat(x(:,3),1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
patch(msh(1,:),msh(2,:),msh(3,:), [0 1 0],'edgecolor',[0 1 0],'facealpha',.3,'edgealpha',.3)
% Plot u12 on x1 tangent space and its parallel transported version on x3
% tangent space
plot3([x(1,3),x(1,3)-p1(1,1)],[x(2,3),x(2,3)-p1(2,1)],[x(3,3),x(3,3)-p1(3,1)], 'color', [1 0 0], 'Linewidth', 2);
plot3([x(1,3),x(1,3)+p3(1,1)],[x(2,3),x(2,3)+p3(2,1)],[x(3,3),x(3,3)+p3(3,1)], 'color', [0 1 0], 'Linewidth', 2);
view(50,30); axis equal; axis tight; axis vis3d;      

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u,x0)
	normu = norm(u,'fro');
	x = x0 * cos(normu) + u./normu * sin(normu);
end

function u = logmap(x,x0)
	p = (x-x0) - trace(x0'*(x-x0)) * x0;
	u = acos(trace(x0'*x)) * p./norm(p,'fro');
end

function Ac = transp(x1,x2,t)
	if nargin==2
		t=1;
	end
	u = logmap(x2,x1);
	e = norm(u,'fro');
	u = u ./ e;
	Ac = -x1 * sin(e*t) * u' + u * cos(e*t) * u' + eye(size(u,1)) - u * u';
end

function x = geodesic(u,x0,t)
	normu = norm(u,'fro');
	x = x0 * cos(normu*t) + u./normu * sin(normu*t);
end