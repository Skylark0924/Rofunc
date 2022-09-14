function demo_gradientDescent03
% Optimization with gradient descent for 1D input and 2D output (Newton's method for roots finding)
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
nbp = 200;
xm(1,:) = linspace(-.3,.5,nbp);
ym = fct(xm);


%% Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = .45; 
[xg, yg, J] = findMin(x0, nbIter);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800],'color',[1 1 1]); hold on; axis off; 
ofs = 1;
for k=1:2 
	plot(xm, ym(k,:)+(k-1)*ofs, '-','linewidth',2,'color',[.6 .6 .6]);
end
plot(xm, -.1*ones(1,nbp), '-','linewidth',2,'color',[.6 .6 .6]);
plot(xm, zeros(1,nbp), '-','linewidth',2,'color',[.6 .6 .6]);
plot(xm, ofs*ones(1,nbp), '-','linewidth',2,'color',[.6 .6 .6]);
axis([min(xm), max(xm), -.12, max(ym(k,:))+ofs]); 
% print('-dpng',['graphs/demo_multiOutputGradientDescent' num2str(00,'%.2d') '.png']);
% pause(.1);
% plot(xm, grad(xm),'g-');
for t=1:nbIter-1
	h=[];
	for k=1:2 
		plot(xg(1,t), yg(k,t)+(k-1)*ofs, '.','markersize',30,'color',[.6 .6 .6]);	
		%Plot tangent plane
		xtmp = [-1, 1] * 4E-2;
		msh = [xtmp; J(k,t)'*xtmp] + repmat([xg(:,t); yg(k,t)+(k-1)*ofs], 1, 2);
		h = [h, plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',[.3 .3 1])];
		h = [h, plot(xg(1,t), yg(k,t)+(k-1)*ofs, '.','markersize',30,'color',[0 0 0])];	
	end %k
	plot(xg(1,t), -.1, '.','markersize',30,'color',[.6 .6 .6]);
	h = [h, plot2DArrow([xg(1,t); -.1], [xg(1,t+1)-xg(1,t); 0], [.3 .3 1], 3, 1E-2)];
	h = [h, plot(xg(1,t), -.1, '.','markersize',30,'color',[0 0 0])];
% 		print('-dpng',['graphs/demo_multiOutputGradientDescent' num2str(t,'%.2d') '.png']);
% 		pause(.1);
% 		delete(h);
end %t
for k=1:2 
	plot(xg(1,end), yg(k,end)+(k-1)*ofs, '.','markersize',40,'color',[.8 0 0]);
end 
plot(xg(1,end), -.1, '.','markersize',40,'color',[.8 0 0]);
	
% print('-dpng',['graphs/demo_multiOutputGradientDescent' num2str(nbIter,'%.2d') '.png']);
pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = fct(x)
	y = [-5.*exp(-x.^2) + 0.2.*sin(10.*x) + 5.118695338; ...
		-5.*exp(-x.^2) + 5]; 
end

function g = grad(x)
	%syms x; f = -5.*exp(-x^2) + 0.2.*sin(10.*x); g = diff(f,x)
 	g = [10.*x.*exp(-x.^2) + 2.*cos(10.*x); ...
		10.*x.*exp(-x.^2)];
end

function [xg,yg,J] = findMin(x,nbIter)
	if nargin<2
		nbIter = 100;
	end
	xg = zeros(size(x,1),nbIter);
	J = zeros(2,nbIter);
	for t=1:nbIter
		xg(:,t) = x; 
		J(:,t) = grad(x);
		%Gauss-Newton method (generalization of Newton's method), https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
		%(note that if J' is used instead of pinv(J)=(J'*J)\J, we obtain a steepest gradient method) 
		x = x - pinv(J(:,t)) * fct(x);
	end
	yg = fct(xg);	
end