function demo_Riemannian_Sd_vecTransp02
% Vector transport on a d-sphere using Schild's ladder algorithm
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
nbDrawingSeg = 40; %Number of segments used to draw ellipsoids
nbIter_sl = 4; % Number of iterationsfor the Schild's ladder algorithm
sigma = 1; % Parameter for the Schild's ladder algorithm

Data = [0.7 0.8 0.9; -0.5 0.3 -0.1; -0.5 -0.5 -0.9];
x = Data./repmat(sqrt(sum(Data.^2,1)),3,1);
t = 0:0.05:1;
nbp = 40;
[X,Y,Z] = sphere(nbp-1);

x1 = x(:,1);
x2 = x(:,2);
a = logmap(x(:,3),x(:,1));


%% Transport of a from x1 to x2 using Schild's ladder algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot sphere
figure('PaperPosition',[0 0 8 8],'position',[10,10,1300,1300]); hold on; axis off; axis equal; rotate3d on; 
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));

% Plot datapoints
plot3(x1(1,:), x1(2,:), x1(3,:), '.','markersize',15,'color',[.5 .5 .5]);
plot3(x2(1,:), x2(2,:), x2(3,:), '.','markersize',15,'color',[.5 .5 .5]);

% Plot tangent space
h = [];
Rrot = rotM(x1)';
msh = repmat(x1,1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 6E-1;
h = [patch(msh(1,:),msh(2,:),msh(3,:), [1 1 1],'edgecolor',[.7 .7 .7],'facealpha',.4,'edgealpha',.8)];

% Plot vector to transport
a_gl = x1 + a;
plot3([x1(1,:) a_gl(1,:)], [x1(2,:) a_gl(2,:)], [x1(3,:) a_gl(3,:)],'linewidth',1,'color',[.8 0 0]);

view(80,-40); axis equal;

umsh_y = logmap(x2,x1);
msh_y = expmap(umsh_y*linspace(0,1,nbDrawingSeg*nbIter_sl), x1);
	
% Initialisation
t = 0:1/nbIter_sl:1;
y = geodesic(logmap(x2,x1),x1,t);
q(:,1) = a; % vector on the tangent space
z(:,1) = geodesic(q(:,1),y(:,1),sigma); % projection of this vector on the manifold

msh_q = expmap(q(:,1)*linspace(0,1,nbDrawingSeg), x1);
plot3(msh_q(1,:), msh_q(2,:), msh_q(3,:), '-','linewidth',1,'color',[0 0 .9]);

% Iterate
for i=1:nbIter_sl
	% Plot current point y_(i+1) on the geodesic between x0 and x1
	plot3(y(1,i+1), y(2,i+1), y(3,i+1), '.','markersize',15,'color',[.3 .3 .3]);
	plot3(msh_y(1,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg), msh_y(2,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg), msh_y(3,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg), '-','linewidth',1,'color',[.3 .3 .3]);
	
	% Compute the middle point between y_(i+1) and z_(i)
	m_dir = logmap(y(:,i+1),z(:,i));
	m(:,i) = geodesic(m_dir,z(:,i),.5);
	msh_m = expmap(m_dir*linspace(0,1,nbDrawingSeg), z(:,i));
	plot3(msh_m(1,:), msh_m(2,:), msh_m(3,:), '-','linewidth',1,'color',[0.9 0.4 0]);
	plot3(m(1,:), m(2,:), m(3,:), '.','markersize',15,'color',[.9 .4 0]);

	% Compute z_(i+1) on the geodesic starting at y_(i) passing through
	% m_(i)
	z_dir = logmap(m(:,i),y(:,i));
	z(:,i+1) = geodesic(z_dir,y(:,i),2*sigma);
	plot3(z(1,i+1), z(2,i+1), z(3,i+1), '.','markersize',15,'color',[0 .8 0]);
	msh_z = expmap(logmap(z(:,i+1),y(:,i))*linspace(0,1,nbDrawingSeg), y(:,i));
	plot3(msh_z(1,:), msh_z(2,:), msh_z(3,:), '-','linewidth',1,'color',[0 .8 0]);
	
	% Compute the current transported vector on the tangent space of
	% y_(i+1)
	q(:,i+1) = logmap(z(:,i+1),y(:,i+1));
	
	msh_q = expmap(q(:,i+1)*linspace(0,1,nbDrawingSeg), y(:,i+1));
	plot3(msh_q(1,:), msh_q(2,:), msh_q(3,:), '-','linewidth',1,'color',[0 0 .9]);
	Rrot = rotM(y(:,i+1))';
	msh = repmat(y(:,i+1),1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 6E-1;
	h = [patch(msh(1,:),msh(2,:),msh(3,:), [1 1 1],'edgecolor',[.7 .7 .7],'facealpha',.2,'edgealpha',.8)];
	q_gl = y(:,i+1) + q(:,i+1);
	plot3([y(1,i+1) q_gl(1,:)], [y(2,i+1) q_gl(2,:)], [y(3,i+1) q_gl(3,:)],'linewidth',1,'color',[.8 0 0]);
end

view(90,-65); axis equal; axis tight; axis vis3d; 

% print('-dpng','graphs/demo_Riemannian_sphere2_SchildLadder01.png');

% Final transported vector
a_schildsladderTransp = q(:,end);
pause();


%% Parallel transport of a from x1 to x2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_prlTransp = transp(x1,x2)*a;

% Plot sphere
figure('PaperPosition',[0 0 8 8],'position',[10,10,1300,1300]); hold on; axis off; axis equal; rotate3d on; 
colormap([.9 .9 .9]);
mesh(X,Y,Z,rand(size(Z)));

% Plot datapoints
plot3(x1(1,:), x1(2,:), x1(3,:), '.','markersize',15,'color',[.5 .5 .5]);
plot3(x2(1,:), x2(2,:), x2(3,:), '.','markersize',15,'color',[.5 .5 .5]);

% Plot starting tangent space
h = [];
Rrot = rotM(x1)';
msh = repmat(x1,1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 6E-1;
h = [patch(msh(1,:),msh(2,:),msh(3,:), [1 1 1],'edgecolor',[.7 .7 .7],'facealpha',.4,'edgealpha',.8)];

% Plot vector to transport
a_gl = x1 + a;
plot3([x1(1,:) a_gl(1,:)], [x1(2,:) a_gl(2,:)], [x1(3,:) a_gl(3,:)],'linewidth',1,'color',[0 0 0]);

% Plot final space
h = [];
Rrot = rotM(x2)';
msh = repmat(x2,1,5) + Rrot * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 6E-1;
h = [patch(msh(1,:),msh(2,:),msh(3,:), [1 1 1],'edgecolor',[.7 .7 .7],'facealpha',.4,'edgealpha',.8)];

% Plot transported vectors (with Schild's ladder and parallel transport)
plot3([x2(1,:) x2(1,:)+a_schildsladderTransp(1,:)], [x2(2,:) x2(2,:)+a_schildsladderTransp(2,:)], [x2(3,:) x2(3,:)+a_schildsladderTransp(3,:)],'linewidth',1,'color',[.8 0 0]);
plot3([x2(1,:) x2(1,:)+a_prlTransp(1,:)], [x2(2,:) x2(2,:)+a_prlTransp(2,:)], [x2(3,:) x2(3,:)+a_prlTransp(3,:)],'linewidth',1,'color',[0 0 0.8]);

view(90,-65); axis equal; axis tight; axis vis3d; 

pause;
close all;
end

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

function Ac = transp(x1,x2,t)
	if nargin==2
		t=1;
	end
	u = logmap(x2,x1);
	e = norm(u,'fro');
	u = u ./ e;
	Ac = -x1 * sin(e*t) * u' + u * cos(e*t) * u' + eye(size(u,1)) - u * u';
end

function x = geodesic(u, x0, t)
	normu = norm(u,'fro');
	x = repmat(x0,[1,size(t,2)]) .* repmat(cos(normu*t),[size(u,1),1]) + repmat(u./normu,[1,size(t,2)]) .* repmat(sin(normu*t),[size(u,1),1]);
end