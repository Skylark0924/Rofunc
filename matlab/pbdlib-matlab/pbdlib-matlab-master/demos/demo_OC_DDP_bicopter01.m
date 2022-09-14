function demo_OC_DDP_bicopter01
% iLQR applied to a bicopter problem.
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
model.dt = 5E-2; %Time step size
model.nbData = 100; %Number of datapoints
model.nbIter = 20; %Number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbVarPos = 3; %Dimension of position (x1,x2,theta)
model.nbDeriv = 2; %Number of derivatives (nbDeriv=2 for [x; dx] state)
model.nbVarX = model.nbVarPos * model.nbDeriv; %State space dimension
model.nbVarU = 2; %Control space dimension (u1,u2)
model.m = 2.5; %Mass
model.l = 0.5; %Length
model.g = 9.81; %Acceleration due to gravity
model.I = 1.2; %Inertia
model.rfactor = 1E-6; %Control weight 
model.Mu = [4; 4; 0; zeros(model.nbVarPos,1)]; %Target

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
% x0 = [-1; 1; -pi/8; .3; .1; .2];

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
	%Estimate step size with backtracking line search method
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
figure('position',[10,10,800,800]); hold on; axis off; grid on; box on;
for t=floor(linspace(1,model.nbData,20))
	plot(r.x(1,t), r.x(2,t), '.','markersize',40,'color',[.6 .6 .6]);
	msh = [r.x(1:2,t), r.x(1:2,t)] + [cos(r.x(3,t)); sin(r.x(3,t))] * [-.4, .4];
	plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[.7 .7 .7]);
end
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[0 0 0]);
msh = [r.x(1:2,1), r.x(1:2,1)] + [cos(r.x(3,1)); sin(r.x(3,1))] * [-.4, .4];
plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[0 0 0]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',40,'color',[0 0 0]);
msh = [model.Mu(1:2), model.Mu(1:2)] + [cos(model.Mu(3)); sin(model.Mu(3))] * [-.4, .4];
plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[.8 0 0]);
h(2) = plot(model.Mu(1), model.Mu(2), '.','markersize',40,'color',[.8 0 0]);
%legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; axis([-.5 4.5 -.2 4.4]);
xlabel('x_1'); ylabel('x_2');
%print('-dpng','graphs/DDP_bicopter01.png');


% %% Plot control space
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[1030,10,1000,1000]);
% t = model.dt:model.dt:model.dt*model.nbData;
% for kk=1:size(r.u,1)
% 	subplot(size(r.u,1), 1, kk); grid on; hold on; box on; box on;
% 	plot(t(1:end-1), r.u(kk,:), '-','linewidth',2,'color',[0 0 0]);
% 	axis([min(t), max(t(1:end-1)), min(r.u(kk,:))-2, max(r.u(kk,:))+2]);
% 	xlabel('t'); ylabel(['u_' num2str(kk)]);
% end


% %% Plot robot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,800,800]); hold on; axis off;
% plot(model.Mu(1), model.Mu(2), '.','markersize',40,'color',[.8 0 0]);
% msh = [model.Mu(1:2), model.Mu(1:2)] + [cos(model.Mu(3)); sin(model.Mu(3))] * [-.4, .4];
% plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[.8 0 0]);
% axis equal; axis([-.5 4.5 -.2 4.2]);
% for t=1:model.nbData
% 	h(1) = plot(r.x(1,t), r.x(2,t), '.','markersize',40,'color',[0 0 0]);
% 	msh = [r.x(1:2,t), r.x(1:2,t)] + [cos(r.x(3,t)); sin(r.x(3,t))] * [-.4, .4];
% 	h(2) = plot(msh(1,:), msh(2,:), '-','linewidth',5,'color',[0 0 0]);
% 	drawnow;
% % 	print('-dpng',['graphs/anim/DDP_bicopter_anim' num2str(t,'%.3d') '.png']);
% 	if t<model.nbData
% 		delete(h);
% 	end
% end
% % for t=model.nbData+1:model.nbData+10
% % 	print('-dpng',['graphs/anim/DDP_bicopter_anim' num2str(t,'%.3d') '.png']);
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
	m = model.m;
	l = model.l;
	I = model.I;
	g = model.g;
	x = zeros(model.nbVarX, model.nbData);
	f = zeros(model.nbVarX, 1);
	x(:,1) = x0;
	for t=1:model.nbData-1
		f(1) = x(4,t);
		f(2) = x(5,t);
		f(3) = x(6,t);
		f(4) = -m^-1 * (u(1,t) + u(2,t)) * sin(x(3,t));
		f(5) =  m^-1 * (u(1,t) + u(2,t)) * cos(x(3,t)) - g;
		f(6) =  l/I  * (u(1,t) - u(2,t));
		x(:,t+1) = x(:,t) + f * model.dt;
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the system along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, model)	
	m = model.m;
	l = model.l;
	I = model.I;
	A = zeros(model.nbVarX, model.nbVarX, model.nbData-1);
	B = zeros(model.nbVarX, model.nbVarU, model.nbData-1);
	Ac = zeros(model.nbVarX);
	Ac(1:3,4:6) = eye(model.nbVarPos);
	Bc = zeros(model.nbVarX, model.nbVarU);
	for t=1:model.nbData-1
		%Linearize the system
		Ac(4,3) = -m^-1 * (u(1) + u(2)) * cos(x(3));
		Ac(5,3) = -m^-1 * (u(1) + u(2)) * sin(x(3));
		Bc(4,1) = -m^-1 * sin(x(3)); Bc(4,2) =  Bc(4,1);
		Bc(5,1) =  m^-1 * cos(x(3)); Bc(5,2) =  Bc(5,1);
		Bc(6,1) =  l / I;            Bc(6,2) = -Bc(6,1);
		%Discretize the linear system
		A(:,:,t) = eye(model.nbVarX) + Ac * model.dt;
		B(:,:,t) = Bc * model.dt;
	end

% 	%Symbolic expressions to find linearized system
% 	syms m l g I
% 	x = sym('x',[6 1]);
% 	u = sym('u',[2 1]);
% 	f(1) = x(4);
% 	f(2) = x(5);
% 	f(3) = x(6);
% 	f(4) = -m^-1 * (u(1) + u(2)) * sin(x(3));
% 	f(5) =  m^-1 * (u(1) + u(2)) * cos(x(3)) - g;
% 	f(6) =  l/I  * (u(1) - u(2));
% 	jacobian(f, x)
% 	jacobian(f, u)
end
