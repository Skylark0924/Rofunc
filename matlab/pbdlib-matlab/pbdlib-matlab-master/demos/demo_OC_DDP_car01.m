function demo_OC_DDP_car01
% iLQR applied to a car parking problem.
%
% If this code is useful for your research, please cite the related publication:
% @article{Lembono21,
% 	author="Lembono, T. S. and Calinon, S.",
% 	title="Probabilistic Iterative {LQR} for Short Time Horizon {MPC}",
% 	year="2021",
% 	journal="arXiv:2012.06349",
% 	pages=""
% }
% 
% Copyright (c) 2021 Idiap Research Institute, https://idiap.ch/
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
model.dt = 1E-1; %Time step size
model.nbData = 100; %Number of datapoints
model.nbIter = 10; %Number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbVarX = 4; %Dimension of state (x1,x2,theta,phi)
model.nbVarU = 2; %Control space dimension (v,dphi)
model.l = 0.5; %Length of car
model.rfactor = 1E-6; %Control weight teR
model.Mu = [4; 3; pi/2; 0]; %Target
% model.Mu = [.2; 2; pi/2; 0]; %Target

R = speye((model.nbData-1)*model.nbVarU) * model.rfactor; %Control weight matrix 
Q = speye(model.nbVarX * model.nbPoints) * 1E3; %Precision matrix

%Time occurrence of viapoints
tl = linspace(1, model.nbData, model.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * model.nbVarX + [1:model.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(model.nbVarU*(model.nbData-1), 1);
x0 = zeros(model.nbVarX, 1);

for n=1:model.nbIter	
	%System evolution
	x = dynSysSimulation(x0, reshape(u, model.nbVarU, model.nbData-1), model);
	%Linearization
	[A, B] = linSys(x, reshape(u, model.nbVarU, model.nbData-1), model);
	Su0 = transferMatrices(A, B);
	Su = Su0(idx,:);
	%Gradient
	e = model.Mu - x(:,tl);
	du = (Su' * Q * Su + R) \ (Su' * Q * e(:) - R * u);
	%Estimate step size with line search method
	alpha = 1;
	cost0 = e(:)' * Q * e(:) + u' * R * u;
	while 1
		utmp = u + du * alpha;
		xtmp = dynSysSimulation(x0, reshape(utmp, model.nbVarU, model.nbData-1), model);		
		etmp = model.Mu - xtmp(:,tl);
		cost = etmp(:)' * Q * etmp(:) + utmp' * R * utmp; %Compute the cost
		if cost < cost0 || alpha < 1E-4
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha; %Update control by following gradient
end

%Log data
r.x = x;
r.u = reshape(u, model.nbVarU, model.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800]); hold on; axis off; 
for t=round(linspace(1, model.nbData, 20))
	R = [cos(r.x(3,t)) -sin(r.x(3,t)); sin(r.x(3,t)) cos(r.x(3,t))];
	msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,t),1,5);
	plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[.7 .7 .7]);
end
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[0 0 0]);
%Initial pose
R = [cos(r.x(3,1)) -sin(r.x(3,1)); sin(r.x(3,1)) cos(r.x(3,1))];
msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,1),1,5);
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 0 0]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',20,'color',[0 0 0]);
% %Final pose
% R = [cos(r.x(3,end)) -sin(r.x(3,end)); sin(r.x(3,end)) cos(r.x(3,end))];
% msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,end),1,5);
% plot(msh(1,:), msh(2,:), '-','linewidth',4,'color',[0 0 0]);
% h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',20,'color',[0 0 0]);
%Target pose
R = [cos(model.Mu(3)) -sin(model.Mu(3)); sin(model.Mu(3)) cos(model.Mu(3))];
msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(model.Mu(1:2),1,5);
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[.8 0 0]);
h(2) = plot(model.Mu(1), model.Mu(2), '.','markersize',20,'color',[.8 0 0]);
%legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; axis([-1 5 -1 4]);
%print('-dpng','graphs/DDP_car01.png');


% %% Plot control space
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[1030,10,1000,1000]);
% t = model.dt:model.dt:model.dt*model.nbData;
% for kk=1:size(r.u,1)
% 	subplot(size(r.u,1), 1, kk); grid on; hold on; box on; box on;
% 	plot(t(1:end-1), r.u(kk,:), '-','linewidth',2,'color',[0 0 0]);
% 	axis([min(t), max(t(1:end-1)), min(r.u(kk,:))-.2, max(r.u(kk,:))+.2]);
% 	xlabel('t'); ylabel(['u_' num2str(kk)]);
% end


% %% Plot robot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,1000,1000]); hold on; axis off;
% %Target pose
% R = [cos(model.Mu(3)) -sin(model.Mu(3)); sin(model.Mu(3)) cos(model.Mu(3))];
% msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(model.Mu(1:2),1,5);
% plot(msh(1,:), msh(2,:), '-','linewidth',4,'color',[.8 0 0]);
% plot(model.Mu(1), model.Mu(2), '.','markersize',40,'color',[.8 0 0]);
% axis equal; axis([-1 5 -1 4]);
% for t=1:model.nbData
% 	R = [cos(r.x(3,t)) -sin(r.x(3,t)); sin(r.x(3,t)) cos(r.x(3,t))];
% 	msh = R * [-.6 -.6 .6 .6 -.6; -.4 .4 .4 -.4 -.4] + repmat(r.x(1:2,t),1,5);
% 	h(1) = plot(msh(1,:), msh(2,:), '-','linewidth',4,'color',[0 0 0]);
% 	h(2) = plot(r.x(1,t),r.x(2,t), '.','markersize',40,'color',[0 0 0]);
% % 	print('-dpng',['graphs/anim/DDP_car_anim' num2str(t,'%.3d') '.png']);
% 	if t<model.nbData
% 		delete(h);
% 	end
% end
% % for t=model.nbData+1:model.nbData+10
% % 	print('-dpng',['graphs/anim/DDP_car_anim' num2str(t,'%.3d') '.png']);
% % end

pause;
close all;
end 

%%%%%%%%%%%%%%%%%%%%%%
function [Su, Sx] = transferMatrices(A, B)
	[nbVarX, nbVarU, nbData] = size(B);
	nbData = nbData+1;
	Sx = kron(ones(nbData,1), speye(nbVarX)); 
	Su = sparse(zeros(nbVarX*(nbData-1), nbVarU*(nbData-1)));
	for t=1:nbData-1
		id1 = (t-1)*nbVarX+1:t*nbVarX;
		id2 = t*nbVarX+1:(t+1)*nbVarX;
		id3 = (t-1)*nbVarU+1:t*nbVarU;
		Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);
		Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
		Su(id2,id3) = B(:,:,t);	
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Given the control trajectory u and initial state x0, compute the whole state trajectory
function x = dynSysSimulation(x0, u, model)	
	l = model.l;
	x = zeros(model.nbVarX, model.nbData);
	f = zeros(model.nbVarX, 1);
	x(:,1) = x0;
	for t=1:model.nbData-1
		f(1) = cos(x(3,t)) * u(1,t);
		f(2) = sin(x(3,t)) * u(1,t);
		f(3) = tan(x(4,t)) * u(1,t) / l;
		f(4) = u(2,t);
		x(:,t+1) = x(:,t) + f * model.dt;
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the system along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, model)	
	l = model.l;
	A = zeros(model.nbVarX, model.nbVarX, model.nbData-1);
	B = zeros(model.nbVarX, model.nbVarU, model.nbData-1);
	Ac = zeros(model.nbVarX);
	Bc = zeros(model.nbVarX, model.nbVarU);
	for t=1:model.nbData-1
		%Linearize the system
		Ac(1,3) = -u(1,t) * sin(x(3,t));
		Ac(2,3) = u(1,t) * cos(x(3,t));
		Ac(3,4) = u(1,t) * tan(x(4,t)^2 + 1) / l;
		Bc(1,1) = cos(x(3,t)); 
		Bc(2,1) = sin(x(3,t)); 
		Bc(3,1) = tan(x(4,t)) / l;
		Bc(4,2) = 1;
		%Discretize the linear system
		A(:,:,t) = eye(model.nbVarX) + Ac * model.dt;
		B(:,:,t) = Bc * model.dt;
	end

% 	%Symbolic expressions to find linearized system
% 	syms l 
% 	x = sym('x',[4 1]);
% 	u = sym('u',[2 1]);
% 	f(1) = cos(x(3)) * u(1);
% 	f(2) = sin(x(3)) * u(1);
% 	f(3) = tan(x(4)) * u(1) / l;
% 	f(4) = u(2);
% 	jacobian(f, x)
% 	jacobian(f, u)
end
