function demo_gradientDescent04
% Optimization with gradient descent for 2D input and 2D output depicting a planar robot IK problem (Newton's method for roots finding).
% First run 'startup_rvc' from the robotics toolbox.
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
nbIter = 7; %Number of iteration for the Gauss Newton algorithm


%% Robot parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDOFs = 2; %Nb of degrees of freedom
armLength = .4;
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
xd = [-.5; .5];


%% Create manifold 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbp = 40;
[xm,ym] = meshgrid(linspace(-pi, pi, nbp));
xm = [xm(:)'; ym(:)'];
ym = fct(xm, xd, robot);


%% Gradient descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x0 = [.2; 2]; %Initial pose
[xg, yg, J] = findMin(x0, xd, robot, nbIter);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1400,800],'color',[1 1 1]); 

subplot(1,2,1); hold on; axis off; rotate3d on;
colormap([.7 .7 .7]);
ofs = 5;
for k=1:2 
	mesh(reshape(xm(1,:),nbp,nbp), reshape(xm(2,:),nbp,nbp), reshape(ym(k,:),nbp,nbp)+(k-1)*ofs,'facealpha',.8,'edgealpha',.8);
	msh = [1 1 -1 -1 1; 1 -1 -1 1 1; zeros(1,5)+(k-1)*ofs/pi] .* pi;
	patch(msh(1,:),msh(2,:),msh(3,:), [0 .6 0],'edgecolor',[0 .3 0],'facealpha',.2,'edgealpha',.2);
end
msh = [1 1 -1 -1 1; 1 -1 -1 1 1; -3/pi*ones(1,5)] .* pi;
patch(msh(1,:),msh(2,:),msh(3,:), [.9 .9 .9],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
view(53,33); axis equal; axis vis3d; %axis([-.55,.55,-.55,.55,-.22,.7]);

%plot robot
subplot(1,2,2); hold on; axis off; 
plot3(xd(1), xd(2), 0, '.','markersize',50,'color',[.8 0 0]);
% plotArm(x0, ones(nbDOFs,1)*armLength, [0;0;-20], .02, [.6 .6 .6]);
axis equal; axis tight; axis([-.55,.45,-.1,.7]);
% pause(.4);
% print('-dpng',['graphs/demo_robotGradientDescent' num2str(0,'%.2d') '.png']);
% pause(.4);

for t=1:nbIter-1
	
	subplot(1,2,1); hold on;
	h=[];
	for k=1:2 
		plot3(xg(1,t), xg(2,t), yg(k,t)+(k-1)*ofs, '.','markersize',20,'color',[.6 .6 .6]);
		plot3(xg(1,t), xg(2,t), -3, '.','markersize',20,'color',[.6 .6 .6]);
% 			%Plot tangent plane
% 			xtmp = [1 1 -1 -1 1; 1 -1 -1 1 1] * 5E-1;
% 			msh = [xtmp; J(k,:,t)*xtmp] + repmat([xg(:,t); yg(k,t)+(k-1)*ofs], 1, 5);
% 			h = [h, patch(msh(1,:),msh(2,:),msh(3,:), [.3 .3 1],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3)];
% 		h = [h, plot3(xg(1,t), xg(2,t), yg(k,t)+(k-1)*ofs, '.','markersize',20,'color',[0 0 0])];
	end %k
	h = [h, mArrow3([xg(:,t); -3], [xg(:,t+1); -3],'tipWidth',6E-2,'stemWidth',2E-2,'color',[.3 .3 1])]; %*norm(J(:,t))
% 	h = [h, plot3(xg(1,t), xg(2,t), -3, '.','markersize',20,'color',[0 0 0])];
% 	h = [h, plot3([xg(1,t), xg(1,t)], [xg(2,t), xg(2,t)], [-3 yg(2,t)+ofs], '-','linewidth',1,'color',[.4 .4 .4])];
	
	%plot robot
	subplot(1,2,2); hold on; 
	plotArm(xg(:,t), ones(nbDOFs,1)*armLength, [0;0;-100+t], .02, [.7 .7 .7]);
	% 	h = [h, plotArm(xg(:,t), ones(nbDOFs,1)*armLength, [0;0;-100+t], .02, [.2 .2 .2])];
	
% 	pause(.4);
% 	print('-dpng',['graphs/demo_robotGradientDescent' num2str(t,'%.2d') '.png']);
% 	pause(.4);
% 	delete(h);
end %t

subplot(1,2,1); hold on;
for k=1:2 
	plot3(xg(1,end), xg(2,end), yg(k,end)+(k-1)*ofs, '.','markersize',40,'color',[.8 0 0]);
end
plot3([xg(1,end), xg(1,end)], [xg(2,end), xg(2,end)], [-3 ofs], '-','linewidth',1,'color',[1 .2 .2]);
plot3(xg(1,end), xg(2,end), -3, '.','markersize',40,'color',[.8 0 0]);
subplot(1,2,2); hold on;
plotArm(xg(:,t), ones(nbDOFs,1)*armLength, [0;0;-1], .02, [.2 .2 .2]);

% pause(.4);
% print('-dpng',['graphs/demo_robotGradientDescent' num2str(nbIter,'%.2d') '.png']);

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = fct(q, xd, robot)
	for t=1:size(q,2)
		Htmp = robot.fkine(q(:,t));
		y(:,t) = Htmp.t(1:2) - xd;
	end
end

function J = grad(q, robot)
	for t=1:size(q,2)
		Jtmp = robot.jacob0(q(:,t),'trans');
		J(:,:,t) = Jtmp(1:2,:);
	end
end	

function [qg, yg, J] = findMin(q, xd, robot, nbIter)
	if nargin<2
		nbIter = 100;
	end
	qg = zeros(size(q,1), nbIter);
	J = zeros(2, 2, nbIter);
	for t=1:nbIter
		qg(:,t) = q;
		J(:,:,t) = grad(q, robot);	
		%Gauss-Newton method (generalization of Newton's method), https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
		%(note that if J' is used instead of pinv(J)=(J'*J)\J, we obtain a steepest gradient method) 
		q = q - pinv(J(:,:,t)) * fct(q, xd, robot);
	end
	yg = fct(qg, xd, robot);	
end