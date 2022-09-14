function demo_gradientDescent01
% Optimization with gradient descent for 1D input (Newton's method for roots finding)
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbIter = 20; %Number of iterations


%% Create manifold 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbp = 200;
xm(1,:) = linspace(-.3,.5,nbp);
ym = fct(xm);


%% Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x0 = -.25;
x0 = 0.45; 
[xg, yg, J, H] = findMin(x0, nbIter);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800],'color',[1 1 1]); hold on; axis off; 
plot(xm, ym, '-','linewidth',2,'color',[.6 .6 .6]);
plot(xm, zeros(1,nbp), '-','linewidth',2,'color',[.6 .6 .6]);
plot(xm, -ones(1,nbp)*.1, '-','linewidth',2,'color',[.6 .6 .6]);
axis equal; axis([min(xm), max(xm), -.12, max(ym)]);
% print('-dpng',['graphs/demo_1DgradientDescent' num2str(0,'%.2d') '.png']);
% pause(.2);
for t=1:nbIter-1
	plot(xg(1,t), yg(1,t), '.','markersize',30,'color',[.6 .6 .6]);	
	plot(xg(1,t), -.1, '.','markersize',30,'color',[.6 .6 .6]);
	%Plot tangent plane
	dx = xg(1,t+1) - xg(1,t);
	if dx<0
		xtmp = [dx, 4E-2];
	else
		xtmp = [-4E-2, dx];
	end
	msh = [xtmp; J(1,t)'*xtmp] + repmat([xg(:,t); yg(:,t)], 1, 2);
	h(1) = plot([xg(1,t), xg(1,t)], [-.1, yg(1,t)], '-','linewidth',1,'color',[.8 .8 .8]);
	h(2) = plot(msh(1,:),msh(2,:), '-','linewidth',2,'color',[0 0 .8]);
	h(3) = plot2DArrow([xg(1,t); -.1], [dx; 0], [0 0 .8], 3, 1E-2);
	h(4) = plot(xg(1,t), yg(1,t), '.','markersize',30,'color',[0 0 0]);	
	h(5) = plot(xg(1,t), -.1, '.','markersize',30,'color',[0 0 0]);
% 	print('-dpng',['graphs/demo_1DgradientDescent' num2str(t,'%.2d') '.png']);
% 	pause(.2);
% 	delete(h);
end
% 	plot(xg, yg, '.','markersize',20,'color',[0 0 0]);	
% 	plot(xg, zeros(1,nbIter), '.','markersize',20,'color',[0 0 0]);
plot(xg(1,end), yg(1,end), '.','markersize',40,'color',[.8 0 0]);
plot(xg(1,end), -.1, '.','markersize',40,'color',[.8 0 0]);
% print('-dpng','graphs/NewtonMethod_rootsFinding_final01.png');

% %Plot Jacobian and Hessian
% Jm = grad(xm);
% figure; hold on; %axis off;
% plot(xm, Jm, '-','linewidth',2,'color',[.6 .6 .6]);
% plot(xm, zeros(1,nbp),'-','linewidth',2,'color',[.6 .6 .6]);
% for t=1:nbIter-1
% 	plot(xg(1,t), J(1,t), '.','markersize',20,'color',[.3 .3 .3]);
% 	%Plot tangent plane
% 	xtmp = [-1, 1] * 4E-2;
% 	msh = [xtmp; H(1,t)'*xtmp] + repmat([xg(:,t); J(1,t)], 1, 2);
% 	h(1) = plot(msh(1,:),msh(2,:), '-','linewidth',3,'color',[.3 .3 1]);
% 	pause
% 	delete(h);
% end
% plot(xg(1,end), J(1,end), '.','markersize',30,'color',[.8 0 0]);
% %axis equal;

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = fct(x)
	y = -5.*exp(-x.^2) + 0.2.*sin(10.*x) + 5.118695338;
end

function g = grad(x)
	%syms x; f = -5.*exp(-x^2) + 0.2.*sin(10.*x); g = diff(f,x)
 	g = 10.*x.*exp(-x.^2) + 2.*cos(10.*x); 
end
		
function H = hess(x)
	%H = diff(g,x)
	%H = hessian(f,x)
 	H = 10.*exp(-x.^2) - 20.*x.^2.*exp(-x.^2) - 20.*sin(10.*x);
end

function [xg,yg,J,H] = findMin(x,nbIter)
	if nargin<2
		nbIter = 100;
	end
	xg = zeros(size(x,1),nbIter);
	J = zeros(size(x,1),nbIter);
	H = zeros(size(x,1),nbIter);
	for t=1:nbIter
		xg(:,t) = x; 
		J(:,t) = grad(x);
		
% 		%First order gradient method (fixed step size)
% 		x = x - 5E-2 .* J(:,t); 

% 		%Second order gradient method (Newton's method for optimization used to find minimum of a function (with null Jacobian), https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
% 		H(:,t) = hess(x);
% 		x = x - 8E-1 * (H(:,t) \ J(:,t)); %Use of a step size (optional)
		
		%Newton's method for finding the zeros of a function (https://en.wikipedia.org/wiki/Newton%27s_method) 
		x = x - J(:,t) \ fct(x);
		
	end
	yg = fct(xg);	
end