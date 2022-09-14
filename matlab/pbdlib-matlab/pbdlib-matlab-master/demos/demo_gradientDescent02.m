function demo_gradientDescent02
% Optimization with gradient descent for 2D input (Newton's method for roots finding)
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
nbIter = 10; %Number of iteration for the Gauss Newton algorithm


%% Create manifold 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbp = 40;
[xm,ym] = meshgrid(linspace(-.5,.5,nbp));
xm = [xm(:)'; ym(:)'];
ym = fct(xm);


%% Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [.47; .4]; %(rand(nbVarIn,1)-0.5) .* .8; 
[xg, yg, J] = findMin(x0,nbIter);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800],'color',[1 1 1]); hold on; axis off; rotate3d on;
colormap([.7 .7 .7]);
mesh(reshape(xm(1,:),nbp,nbp), reshape(xm(2,:),nbp,nbp), reshape(ym(1,:),nbp,nbp),'facealpha',.8,'edgealpha',.8);
msh = [1 1 -1 -1 1; 1 -1 -1 1 1; -.4*ones(1,5)] .* .5;
patch(msh(1,:), msh(2,:), msh(3,:), [.9 .9 .9],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
% patch(msh(1,:), msh(2,:), zeros(1,5), [.9 .9 .9],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
view(53,33); axis equal; axis vis3d; axis([-.55,.55,-.55,.55,-.22,.7]);
% print('-dpng',['graphs/demo_2DgradientDescent' num2str(0,'%.2d') '.png']);
% pause(.4);
for t=1:nbIter-1
	plot3(xg(1,t), xg(2,t), yg(1,t), '.','markersize',20,'color',[.6 .6 .6]);
	plot3(xg(1,t), xg(2,t), -.2, '.','markersize',20,'color',[.6 .6 .6]);
	%Plot tangent plane
	xtmp = [1 1 -1 -1 1; 1 -1 -1 1 1] * 1E-1;
	msh = [xtmp; J(:,t)'*xtmp] + repmat([xg(:,t); yg(1,t)], 1, 5);
	h = patch(msh(1,:),msh(2,:),msh(3,:), [.3 .3 1],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
	h = [h, mArrow3([xg(:,t); -.2], [xg(:,t+1); -.2],'tipWidth',8E-3,'stemWidth',2E-3,'color',[.3 .3 1])]; %*norm(J(:,t))
	h = [h, plot3(xg(1,t), xg(2,t), yg(1,t), '.','markersize',20,'color',[0 0 0])];
	h = [h, plot3(xg(1,t), xg(2,t), -.2, '.','markersize',20,'color',[0 0 0])];
% 		pause(.4);
% 		print('-dpng',['graphs/demo_2DgradientDescent' num2str(t,'%.2d') '.png']);
% 		delete(h);
end
% 		plot3(xg(1,:), xg(2,:), yg(1,:), '.','markersize',20,'color',[0 0 0]);
% 		plot3(xg(1,:), xg(2,:), zeros(1,nbIter), '.','markersize',20,'color',[0 0 0]);
plot3(xg(1,end), xg(2,end), yg(1,end), '.','markersize',40,'color',[.8 0 0]);
plot3(xg(1,end), xg(2,end), -.2, '.','markersize',40,'color',[.8 0 0]);

% print('-dpng',['graphs/demo_2DgradientDescent' num2str(nbIter,'%.2d') '.png']);
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function y = fct(x) 
% 	y = x(1,:).^2 - x(2,:).^4 .* x(1,:);
% end
% function g = grad(x)
% 	%syms x1 x2; f = x1^2 - x2^4 * x1; g = [diff(f,x1); diff(f,x2)]
%  	g = [-x(2,:).^4 + 2.*x(1,:); -4.*x(1,:).*x(2,:).^3];		
% end
% function H = hess(x)
% 	%H = [diff(g,x1), diff(g,x2)]
%  	H = [2, -4.*x(2,:).^3; -4.*x(2,:).^3, -12.*x(1,:).*x(2,:).^2];
% end

function y = fct(x)
	y = -exp(-x(1,:).^2 - x(2,:).^2) + 0.05.*sin(10.*x(1,:)) + 1.03252590037;
end
function g = grad(x)
	%syms x1 x2; f = -exp(-x1^2 -x2^2); g = [diff(f,x1); diff(f,x2)]
 	g = [2.*x(1,:)*exp(- x(1,:).^2 - x(2,:).^2) + 0.5.*cos(10.*x(1,:)); 2.*x(2,:).*exp(- x(1,:).^2 - x(2,:).^2)];
end	
function H = hess(x)
	%H = diff(g,x)
 	H = [-4.*x(1,:).^2*exp(- x(1,:).^2 - x(2,:).^2) + 2.*exp(- x(1,:).^2 - x(2,:).^2) - 5.*sin(10.*x(1,:)), ...
			-4.*x(1,:).*x(2,:).*exp(- x(1,:).^2 - x(2,:).^2); ...
			-4.*x(1,:).*x(2,:).*exp(- x(1,:).^2 - x(2,:).^2), ...
			-4.*x(2,:).^2.*exp(- x(1,:).^2 - x(2,:).^2) + 2.*exp(- x(1,:).^2 - x(2,:).^2)];
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
% 		x = x - 3E-1 .* J(:,t); 
		
% 		%Second order gradient method (Newton's method for optimization, https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
% 		H(:,t) = hess(x);
% 		x = x - (J(:,t) / H(:,t)); 

		%Newton's method for finding the zeros of a function (https://en.wikipedia.org/wiki/Newton%27s_method) 
		x = x - pinv(J(:,t)') * fct(x);
	end
	yg = fct(xg);	
end