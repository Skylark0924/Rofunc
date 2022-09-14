function demo_Riemannian_Hd_interp01
% Interpolation on n-hyperboloid manifold
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
model.nbVar = 3; %Number of variables (compatible with 2, 3 or 4 for the plots)
model.nbStates = 2; %Number of states (-> interpolation between two points)
nbData = 70; %Number of interpolation steps
% nbIter = 5; %Number of iteration for the Gauss Newton algorithm
nbRepros = 1; %Number of reproductions 

for n=1:nbRepros
% 	r(n).x = [[-2;5;3], [2;2;-2]];
	r(n).x = (rand(model.nbVar,model.nbStates) - 0.5) * 1E1;
	for i=1:model.nbStates
		r(n).x(end,i) = sqrt(r(n).x(1:end-1,i)' * r(n).x(1:end-1,i) + 1);
		%r(n).x(1:end-1,i)' * r(n).x(1:end-1,i) - r(n).x(end,i)^2
	end
end

p = (rand(model.nbVar,1) - 0.5) * 1E1;
p(end) = sqrt(p(1:end-1)' * p(1:end-1) + 1);
v = logmap(p, r(n).x(:,1)); %random vector in the tangent space of r(n).x(:,1) to be transported


%% Geodesic interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = [linspace(1,0,nbData); linspace(0,1,nbData)];
for n=1:nbRepros
	r(n).xi = zeros(model.nbVar,nbData);
	% xtmp = x(:,1);
	for t=1:nbData
	% 	%Interpolation between more than 2 points can be computed in an iterative form
	% 	for n=1:nbIter
	% 		utmp = zeros(model.nbVar,1);
	% 		for i=1:model.nbStates
	% 			utmp = utmp + w(i,t) * logmap(x(:,i), xtmp);
	% 		end
	% 		xtmp = expmap(utmp, xtmp);
	% 	end
	% 	r(n).xi(:,t) = xtmp;

		%Interpolation between two points can be computed in closed form
		r(n).xi(:,t) = expmap(w(2,t)*logmap(r(n).x(:,2), r(n).x(:,1)), r(n).x(:,1));	
		
		%Inner product between vector transported and direction of geodesic
		M = blkdiag(eye(size(r(n).x,1)-1), -1);
		ptv = transp(r(n).x(:,1), r(n).xi(:,t), v);
		dir = logmap(r(n).x(:,2), r(n).xi(:,t));
		if norm(dir) > 1E-5
			th = sqrt(dir' * M * dir);
			dir = dir ./ th; %Normalize (with Minkowski inner product)
			inprod(t) = dir' * M * ptv; %Minkowski inner product
		end		
	end
end
inprod


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if model.nbVar==3
	%Plot as Poincare disk model 
	figure('name','Poincare disk model','position',[10,10,650,650]); hold on; axis off; 
	plot(0,0,'k+','linewidth',1); %,'markersize',20
	t = linspace(0,2*pi,50);
	plot(cos(t), sin(t), '-','linewidth',1,'color',[0 0 0]);	
	for n=1:nbRepros
		%Conversion from hyperboloid model to Poincare disk model
		for i=1:model.nbStates
			r(n).x2(:,i) = r(n).x(1:end-1,i) / (1+r(n).x(end,i));
		end
		for t=1:nbData
			r(n).xi2(:,t) = r(n).xi(1:end-1,t) / (1+r(n).xi(end,t));
		end
		%u = logmap(x(:,2), x(:,1));
		%u2 = u(1:end-1) / (1+u(end));
		%plot2DArrow(x2(:,1), u2*.2, [0,0,0], 1, .05);
		plot(r(n).x2(1,:), r(n).x2(2,:), '.','markersize',20,'color',[.8 0 0]);
		axis equal;
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_PoincareDisk' num2str(n) 'a.png']);
		plot(r(n).xi2(1,:), r(n).xi2(2,:), '-','linewidth',1,'color',[0 0 0]);
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_PoincareDisk' num2str(n) 'b.png']);
	end
	
% 	%Plot also as 3D hyperboloid
% 	figure('name','3D hyperboloid model','position',[10,750,650,500]); hold on; axis off; rotate3d on;
% 	plot3(0,0,0,'k+');
% 	for n=1:nbRepros
% 		plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '.','markersize',20,'color',[.8 0 0]);
% 		plot3(r(n).xi(1,:), r(n).xi(2,:), r(n).xi(3,:), 'k-');
% 	end
% 	view(3); axis equal; axis tight; axis vis3d; 
	
	%Plot also as Poincare half-plane model
	figure('name','Poincare half-plane model','position',[680,750,650,500]); hold on; axis off;
	%plot(0,0,'k+');
	plot([-6,6],[0,0],'-','linewidth',1,'color',[0 0 0]);
	plot([0,0],[-5,5],'-','linewidth',1,'color',[0 0 0]);
	for n=1:nbRepros
		%Conversion from Poincare disk model to Poincare half-plane model (https://en.wikipedia.org/wiki/Poincar%C3%A9_disk_model)
		for i=1:model.nbStates
			r(n).x3(:,i) = [2*r(n).x2(1,i)/(r(n).x2(1,i)^2+(1-r(n).x2(2,i))^2); (1-r(n).x2(1,i)^2-r(n).x2(2,i)^2)/(r(n).x2(1,i)^2+(1-r(n).x2(2,i))^2)]; 
		end
		for t=1:nbData
			r(n).xi3(:,t) = [2*r(n).xi2(1,t)/(r(n).xi2(1,t)^2+(1-r(n).xi2(2,t))^2); (1-r(n).xi2(1,t)^2-r(n).xi2(2,t)^2)/(r(n).xi2(1,t)^2+(1-r(n).xi2(2,t))^2)]; 
		end	
		plot(r(n).x3(1,:), r(n).x3(2,:), '.','markersize',20,'color',[.8 0 0]);
		axis equal;
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_PoincareHalfPlane' num2str(n) 'a.png']);
		plot(r(n).xi3(1,:), r(n).xi3(2,:), '-','linewidth',1,'color',[0 0 0]);
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_PoincareHalfPlane' num2str(n) 'b.png']);
	end
	
	
	%Plot also as 1D normal distribution interpolations (corresponding to Fisher-Rao metric)
	figure('name','1D normal distribution interpolations (Fisher-Rao metric)','position',[1350,10,800,650]); 
	rg = [-9,6];
	for n=1:min(nbRepros,8)
		subplot(1,1,n); hold on; axis off;
% 		mtmp.Mu = reshape(r(n).xi3(1,:),[1,size(r(n).xi3,2)]);
% 		mtmp.Sigma = reshape(r(n).xi3(2,:),[1,1,size(r(n).xi3,2)]);
% 		plotGMM1D(mtmp,[-5,5,0,1],[.8 0 0],.2);
% 		mtmp.Mu = reshape(r(n).x3(1,:),[1,size(r(n).x3,2)]);
% 		mtmp.Sigma = reshape(r(n).x3(2,:),[1,1,size(r(n).x3,2)]);
% 		plotGMM1D(mtmp,[-5,5,0,1],[0 0 0],.2);

% 		for i=1:size(r(n).x3,2)
% 			mtmp.Mu = r(n).x3(1,i);
% 			mtmp.Sigma = r(n).x3(2,i);
% 			h = plotGMM1D(mtmp,[rg,0,gaussPDF(0,0,mtmp.Sigma)],[.8 0 0],.5);
% 		end
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_FisherRao' num2str(n) 'a.png']);
% 		delete(h)
		
		for i=1:5:size(r(n).xi3,2)
			mtmp.Mu = r(n).xi3(1,i);
			mtmp.Sigma = r(n).xi3(2,i);
			plotGMM1D(mtmp,[rg,0,gaussPDF(0,0,mtmp.Sigma)],[0 0 0],.1);
		end
		for i=1:size(r(n).x3,2)
			mtmp.Mu = r(n).x3(1,i);
			mtmp.Sigma = r(n).x3(2,i);
			plotGMM1D(mtmp,[rg,0,gaussPDF(0,0,mtmp.Sigma)],[.8 0 0],.5);
		end
% 		print('-dpng',['graphs/demo_Riemannian_Hd_interp_FisherRao' num2str(n) 'b.png']);
	end
	
elseif model.nbVar==4
	%Plot as Poincare ball model
	figure('name','Poincare ball model','position',[10,10,1300,1300]); hold on; axis off; 
	colormap([.8 .8 .8]);
	[X,Y,Z] = sphere(50);	
	mesh(X,Y,Z,'facealpha',.1);
	for n=1:nbRepros
		%Conversion from hyperboloid model to Poincare ball model
		for i=1:model.nbStates
			x2(:,i) = r(n).x(1:end-1,i) / (1+r(n).x(end,i));
		end
		for t=1:nbData
			xi2(:,t) = r(n).xi(1:end-1,t) / (1+r(n).xi(end,t));
		end
		%u = logmap(x(:,2), x(:,1));
		%u2 = u(1:end-1) / (1+u(end));
		%plot2DArrow(x2(:,1), u2*.2, [0,0,0], 1, .05);

		plot3(x2(1,:), x2(2,:), x2(3,:), '.','markersize',20,'color',[.8 0 0]);
		plot3(xi2(1,:), xi2(2,:), xi2(3,:), '-','linewidth',1,'color',[0 0 0]);
	end 
	view(3); axis equal; axis tight; axis vis3d; rotate3d on; 
% 	print('-dpng','graphs/demo_Riemannian_Hd_interp_PoincareBall01.png');
else
	%Plot as 2D hyperboloid model
	figure('name','2D hyperboloid model','position',[10,10,1300,650]); hold on; axis off; 
	for n=1:nbRepros
		plot(r(n).x(1,:), r(n).x(2,:), '.','markersize',20,'color',[.8 0 0]);
		u = logmap(r(n).x(:,2), r(n).x(:,1));
		plot2DArrow(r(n).x(:,1), u*.2, [0,0,0], 1, .05);
		plot(r(n).xi(1,:),r(n).xi(2,:),'-','linewidth',1,'color',[0 0 0]);
	end
	axis equal; 
end

% print('-dpng','graphs/demo_Riemannian_Hd_interp_hyperboloid2D01.png');
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u,x0)
	M = blkdiag(eye(size(x0,1)-1), -1); %Section 4.7.4 p.219 (in pdf, p.227) of Robbin and Salamon (2013), Introduction to differential geometry (https://www.math.wisc.edu/~robbin/Do_Carmo/diffgeo.pdf)
	for t=1:size(u,2)
		th = sqrt(u(:,t)' * M * u(:,t));
		if th<1e-16
			x(:,t) = x0;
		else
			%see p.224 (in pdf, p.232) of Robbin and Salamon (2013), Introduction to differential geometry (https://www.math.wisc.edu/~robbin/Do_Carmo/diffgeo.pdf) (or also https://ronnybergmann.net/mvirt/manifolds/Hn.html)
			x(:,t) = x0 .* cosh(th) + u(:,t) .* sinh(th) ./ th; 
		end
	end
end

function u = logmap(x,x0)
	M = blkdiag(eye(size(x0,1)-1), -1);
	for t=1:size(x,2)

% 		d = acosh(-x0' * M * x(:,t));
% 		if theta<1e-16
% 			u(:,t) = zeros(size(x,1),1);
% 		else
% 			u(:,t) = (x(:,t) - x0 .* cosh(d)) .* d ./ sinh(d);
% 		end

		e = x0' * M * x(:,t);
% 		u(:,t) = acosh(-e) .* (x(:,t) + e .* x0) ./ sqrt(e^2 - 1);
		d = acosh(-e);
		if d<1e-16
			u(:,t) = zeros(size(x,1),1);
		else
			u(:,t) = d .* (x(:,t) + e .* x0) ./ sqrt(e^2 - 1); %sqrt(e^2 - 1) is also equals to sqrt(a' * M * a) with a = (x(:,t) + e .* x0)
		end
		
	end
end

function v = transp(x,y,v)
	M = blkdiag(eye(size(x,1)-1), -1);
	d = acosh(-x' * M * y);
	v = v - (logmap(y,x) + logmap(x,y)) .* (logmap(y,x)' * M * v) ./ d.^2;
end