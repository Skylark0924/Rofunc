function demo_OC_DDP_cartpole01
% iLQR applied to a cartpole problem.
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
model.nbIter = 30; %Number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbVarPos = 2; %Dimension of position
model.nbDeriv = 2; %Number of derivatives (nbDeriv=2 for [x; dx] state)
model.nbVarX = model.nbVarPos * model.nbDeriv; %State space dimension
model.nbVarU = 1; %Control space dimension
model.m = 1; %Pendulum mass
model.M = 5; %Cart mass
model.L = 1; %Pendulum length
model.g = 9.81; %Acceleration due to gravity
model.d = 1; %Cart damping
% model.b = 0.01; %Friction
model.rfactor = 1E-5; %Control weight teR
model.Mu = [2; pi; 0; 0]; %Target

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
% x0 = [0; pi/4; .4; 0];

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
figure('position',[10,10,800,800]); hold on; axis on; grid on; box on;
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[.6 .6 .6]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',40,'color',[0 0 0]);
h(2) = plot(model.Mu(1), model.Mu(2), '.','markersize',40,'color',[.8 0 0]);
legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; %axis([-5 5 -4 6]);
xlabel('x'); ylabel('\theta');
% print('-dpng','graphs/DDP_cartpole01.png');


% %% Plot control space
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[1020,10,1000,1000]);
% t = model.dt:model.dt:model.dt*model.nbData;
% for kk=1:size(r.u,1)
% 	subplot(size(r.u,1), 1, kk); grid on; hold on; box on; box on;
% 	for k=1:length(r)
% 		plot(t(1:end-1), r(k).u(kk,:), '-','linewidth',1,'color',(ones(1,3)-k/length(r))*.9);
% 	end
% 	plot(t(1:end-1), r.u(kk,:), '-','linewidth',2,'color',[0 0 0]);
% 	axis([min(t), max(t(1:end-1)), min(r.u(kk,:))-2, max(r.u(kk,:))+2]);
% 	xlabel('t'); ylabel(['u_' num2str(kk)]);
% end

%% Plot robot (static)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[820 10 800 400],'color',[1 1 1]); hold on; axis off;
plotArm(model.Mu(2)-pi/2, model.L, [0;0;-1], .1, [.8 0 0]);
for t=floor(linspace(1, model.nbData, 30))
	plotArm(r.x(2,t)-pi/2, model.L, [r.x(1,t); 0; t*.1], .1, [.7 .7 .7]-.7*t/model.nbData);
end
axis equal; %axis([-4.2 4.2 -1.2 1.2]*2);


% %% Plot robot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 2200 700],'color',[1 1 1]); hold on; axis off;
% plotArm(model.Mu(2)-pi/2, model.L, [0;0;-1], .1, [.8 0 0]);
% axis equal; axis([-4.2 4.2 -1.2 1.2]*2);
% h=[];
% for t=1:model.nbData
% 	delete(h);
% 	h = plotArm(r.x(2,t)-pi/2, model.L, [r.x(1,t);0;0], .1, [0 0 0]);
% 	drawnow;
% end

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
	M = model.M;
	L = model.L;
	g = model.g;
	d = model.d;
	x = zeros(model.nbVarX, model.nbData);
	f = zeros(model.nbVarX, 1);
	x(:,1) = x0;
	for t=1:model.nbData-1
		sx = sin(x(2,t));
		cx = cos(x(2,t));
		D = m*L^2*(M+m*(1-cx^2));
		f(1) = x(3,t);
		f(2) = x(4,t);
% 		f(3) = (1/D)*(-m^2*L^2*g*cx*sx + m*L^2*(m*L*x(4,t)^2*sx - d*x(3,t))) + m*L^2*(1/D) * u(1,t);
% 		f(4) = (1/D)*((m+M)*m*g*L*sx   - m*L*cx*(m*L*x(4,t)^2*sx - d*x(3,t))) - m*L*cx*(1/D) * u(1,t);
		f(3) = (u(1,t) + m*sx*(L*x(4,t)^2+g*cx)) / (M+m*sx^2); %From RLSC Homework (Part 2: Optimal Control)
		f(4) = (-u(1,t) - m*L*x(4,t)*cx*sx - (M+m)*g*sx) / (L*(M+m*sx^2)); %From RLSC Homework (Part 2: Optimal Control)
		x(:,t+1) = x(:,t) + f * model.dt;
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the sxstem along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, model)
	m = model.m;
	M = model.M;
	L = model.L;
	g = model.g;
	d = model.d;

	A = zeros(model.nbVarX, model.nbVarX, model.nbData-1);
	B = zeros(model.nbVarX, model.nbVarU, model.nbData-1);
	
% 	%Linearization at target (pendulum up corresponds to s=1)
% 	s = 1; 
% 	Ac = [0 0 1 0;
% 				 0 0 0 1;
% 				 0 -m*g/M -d/M  0;
% 				 0 -s*(m+M)*g/(M*L) -s*d/(M*L) 0];
% 	Bc = [0; 0; 1/M; s*1/(M*L)];

	for t=1:model.nbData-1
		%Linearization
		Ac = [0,                                                                                                                                                                                0, 1,                                      0; ...
					0,                                                                                                                                                                                0, 0,                                      1; ...
					0,                              ((m*cos(x(2,t))*(L*x(4,t)^2 + g*cos(x(2,t))) - g*m*sin(x(2,t))^2)/(m*sin(x(2,t))^2 + M) - (2*m*cos(x(2,t))*sin(x(2,t))*(u(1,t) + m*sin(x(2,t))*(L*x(4,t)^2 + g*cos(x(2,t)))))/(m*sin(x(2,t))^2 + M)^2),  0,   ((2*L*m*x(4,t)*sin(x(2,t)))/(m*sin(x(2,t))^2 + M)); ...
					0, (2*m*cos(x(2,t))*sin(x(2,t))*(u(1,t) + g*sin(x(2,t))*(M + m) + L*m*x(4,t)*cos(x(2,t))*sin(x(2,t))))/(L*(m*sin(x(2,t))^2 + M)^2) - (g*cos(x(2,t))*(M + m) + L*m*x(4,t)*cos(x(2,t))^2 - L*m*x(4,t)*sin(x(2,t))^2)/(L*(m*sin(x(2,t))^2 + M)),  0, (-(m*cos(x(2,t))*sin(x(2,t)))/(m*sin(x(2,t))^2 + M))];
		Bc = [0; 0; 1/(m*sin(x(2,t))^2 + M); -1/(L*(m*sin(x(2,t))^2 + M))];
		%Discretize the linear sxstem
		A(:,:,t) = eye(model.nbVarX) + Ac * model.dt;
		B(:,:,t) = Bc * model.dt;
	end
	
% 	%Symbolic expressions to find linearized system	
% 	syms m M L g d u
% 	x = sym('x',[4 1]);
% 	sx = sin(x(2));
% 	cx = cos(x(2));
% 	D = m*L^2*(M+m*(1-cx^2));
% 	f(1) = x(3);
% 	f(2) = x(4);
% % 	f(3) = (1/D)*(-m^2*L^2*g*cx*sx + m*L^2*(m*L*x(4)^2*sx - d*x(3))) + m*L^2*(1/D) * u;
% % 	f(4) = (1/D)*((m+M)*m*g*L*sx   - m*L*cx*(m*L*x(4)^2*sx - d*x(3))) - m*L*cx*(1/D) * u;
% 	f(3) = (u + m*sx*(L*x(4)^2+g*cx)) / (M+m*sx^2);
% 	f(4) = (-u - m*L*x(4)*cx*sx - (M+m)*g*sx) / (L*(M+m*sx^2));
% 	jacobian(f, x)
% 	jacobian(f, u)
end