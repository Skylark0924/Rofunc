function demo_OC_DDP_pendulum01
% iLQR applied to an inverted pendulum problem.
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
model.dt = 2E-2; %Time step size
model.nbData = 100; %Number of datapoints
model.nbIter = 300; %Maximum number of iterations for iLQR
model.nbPoints = 1; %Number of viapoints
model.nbVarPos = 1; %Dimension of position
model.nbDeriv = 2; %Number of derivatives (nbDeriv=2 for [x; dx] state)
model.nbVarX = model.nbVarPos * model.nbDeriv; %State space dimension
model.nbVarU = 1; %Control space dimension
model.l = 1; %Link length
model.m = 1; %Mass
model.g = 9.81; %Acceleration due to gravity
model.b = 0.01; %Friction (optional)
model.wq = 1E3; %Tracking weight
model.wr = 1E-2; %Control weight 
model.Mu = [pi/2; 0]; %Target

Q = speye(model.nbVarX * model.nbPoints) * model.wq; %Precision matrix
R = speye(model.nbVarU * (model.nbData-1)) * model.wr; %Control weight matrix 

%Time occurrence of viapoints
tl = linspace(1, model.nbData, model.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * model.nbVarX + [1:model.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(model.nbVarU*(model.nbData-1), 1);
% x0 = zeros(model.nbVarX, 1);
x0 = [-pi/2; 0];

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
	
	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);

%Log data
r.x = x;
r.u = reshape(u, model.nbVarU, model.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,800]); hold on; axis on; grid on; box on;
plot(r.x(1,:),r.x(2,:), '-','linewidth',2,'color',[.6 .6 .6]);
h(1) = plot(r.x(1,1),r.x(2,1), '.','markersize',30,'color',[0 0 0]);
h(2) = plot(model.Mu(1), model.Mu(2), '.','markersize',30,'color',[.8 0 0]);
legend(h,{'Initial pose','Target pose'},'location','northwest','fontsize',30);
axis equal; axis([-pi pi -2 5]);
set(gca,'xtick',[-pi, -pi/2, 0, pi/2, pi],'xticklabel',{'-\pi','-\pi/2','0','\pi/2','\pi'},'ytick',0,'fontsize',22);
xlabel('$q$','interpreter','latex','fontsize',30); 
ylabel('$\dot{q}$','interpreter','latex','fontsize',30);
% print('-dpng','graphs/DDP_pendulum_q_dq01.png');


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


% %% Plot robot (static)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[820 10 800 800],'color',[1 1 1]); hold on; axis off;
% plotArm(model.Mu(1), model.l, [0;0;20], .05, [.8 0 0]);
% for t=floor(linspace(1, model.nbData, 30))
% 	plotArm(r.x(1,t), model.l, [0; 0; t*.1], .05, [.7 .7 .7]-.7*t/model.nbData);
% end
% axis equal; axis([-1.2 1.2 -1.2 1.2]);
% % print('-dpng','graphs/DDP_pendulum01.png');


% %% Plot robot (animated)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10 10 700 700],'color',[1 1 1]); hold on; axis off;
% plotArm(model.Mu(1), model.l, [0;0;-1], .05, [.8 0 0]);
% axis equal; axis([-1.2 1.2 -1.2 1.2]);
% h=[];
% for t=1:model.nbData
% % 	delete(h);
% 	h = plotArm(r.x(1,t), model.l, zeros(3,1), .05, [0 0 0]);
% 	drawnow;
% % 	print('-dpng',['graphs/anim/DDP_pendulum_anim' num2str(t,'%.3d') '.png']);
% end
% % for t=model.nbData+1:model.nbData+10
% % 	print('-dpng',['graphs/anim/DDP_pendulum_anim' num2str(t,'%.3d') '.png']);
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
	b = model.b;
	l = model.l;
	m = model.m;
	g = model.g;
	x = zeros(model.nbVarX, model.nbData);
	f = zeros(model.nbVarX, 1);
	x(:,1) = x0;
	for t=1:model.nbData-1
		f(1) = x(2,t);
% 		f(2) = u(t) / (m * l^2) - cos(x(1,t)) * g / l - (b / (m * l^2)) * x(2,t); %With friction
		f(2) = u(t) / (m * l^2) - cos(x(1,t)) * g / l; %Without friction
		x(:,t+1) = x(:,t) + f * model.dt;
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Linearize the system along the trajectory computing the matrices A and B
function [A, B] = linSys(x, u, model)
	b = model.b;
	g = model.g;
	l = model.l;
	m = model.m;	
	A = zeros(model.nbVarX, model.nbVarX, model.nbData-1);
	B = zeros(model.nbVarX, model.nbVarU, model.nbData-1);
	Bc = [0; 1/(m * l^2)];
	for t=1:model.nbData-1
		%Linearize the system
% 		Ac = [0, 1; sin(x(1,t)) * g / l, -b / (m * l^2)]; %With friction
		Ac = [0, 1; sin(x(1,t)) * g / l, 0]; %Without friction

		%Discretize the linear system
		A(:,:,t) = eye(model.nbVarX) + Ac * model.dt;
		B(:,:,t) = Bc * model.dt;
	end

% 	%Symbolic expressions to find linearized system
% 	syms b m l g x u 
% 	x = sym('x',[2 1]);
% 	f(1) = x(2);
% 	f(2) = u / (m * l^2) - (g / l) * sin(x(1)) - (b / (m * l^2)) * x(2);
% 	jacobian(f, x)
% 	jacobian(f, u)
end