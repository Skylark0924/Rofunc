function demo_OC_DDP_humanoid01
% iLQR applied to a planar 5-link humanoid problem with constraints between joints.
% (see also demo_OC_DDP_CoM01.m and demo_OC_DDP_bimanual01.m)
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
model.dt = 1E-2; %Time step size
model.nbData = 100; %Number of datapoints
model.nbPoints = 2; %Number of viapoints
model.nbIter = 20; %Number of iterations for iLQR
model.nbVarX = 5; %State space dimension (q1,q2,q3,q4,q5)
model.nbVarU = 3; %Control space dimension (dq1,dq2,dq3)
% model.nbVarU = 5; %Control space dimension (dq1,dq2,dq3,dq4,dq5)
model.nbVarF = 3; %Objective function dimension (x1,x2,o)

model.l = ones(1, model.nbVarX) * 2; %Links lengths
model.sz = [.2, .2]; %Size of objects
model.r = 1E-5; %Control weight term
model.Mu = [[2.5; 1; 0], [3; 4; 0]]; %Viapoints 

R = speye((model.nbData-1)*model.nbVarU) * model.r; %Control weight matrix (at trajectory level)
% Q = speye(model.nbVarF * model.nbPoints) * 1E3; %Precision matrix
Q = kron(eye(model.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (orientation does not matter)

%Time occurrence of viapoints
tl = linspace(1, model.nbData, model.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * model.nbVarX + [1:model.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(model.nbVarU*(model.nbData-1), 1);
a = .7;
x0 = [pi/2-a; 2*a; -a; pi-.2; pi/2];

A = repmat(eye(model.nbVarX), [1 1 model.nbData-1]);
% B = repmat(eye(model.nbVarX, model.nbVarU) * model.dt, [1 1 model.nbData-1]);
phi = [-1,0,0; 2,0,0; -1,0,0; 0,1,0; 0,0,1]; %Joint coordination matrix
B = repmat(phi * model.dt, [1 1 model.nbData-1]); 

[Su0, Sx0] = transferMatrices(A, B); %Constant Su and Sx for the proposed system
Su = Su0(idx,:);

for n=1:model.nbIter	
	x = reshape(Su0 * u + Sx0 * x0, model.nbVarX, model.nbData); %System evolution
	f = fkine(x(:,tl), model);
	J = jacob(x(:,tl), f, model);
	du = (Su' * J' * Q * J * Su + R) \ (-Su' * J' * Q * f(:) - u * model.r); %Gradient
	
	%Estimate step size with line search method
	alpha = 1;
	cost0 = f(:)' * Q * f(:) + norm(u)^2 * model.r; %u' * R * u
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, model.nbVarX, model.nbData);
		ftmp = fkine(xtmp(:,tl), model);
		cost = ftmp(:)' * Q * ftmp(:) + norm(utmp)^2 * model.r; %utmp' * R * utmp
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end
	u = u + du * alpha; %Update control by following gradient
end

%Log data
r.x = x;
r.f = fkine0(x, model); 
r.u = reshape(u, model.nbVarU, model.nbData-1);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
colMat = lines(model.nbPoints);
msh0 = diag(model.sz) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:model.nbPoints
	msh(:,:,t) = msh0 + repmat(model.Mu(1:2,t), 1, size(msh0,2));
end

figure('position',[10,10,1000,800],'color',[1,1,1]); hold on; axis off;
plotArm(r.x(:,1), model.l, [0; 0; -13], .2, [.8 .8 .8]);
plotArm(r.x(:,model.nbData/2+1), model.l, [0; 0; -12], .2, [.6 .6 .6]);
plotArm(r.x(:,model.nbData), model.l, [0; 0; -11], .2, [.4 .4 .4]);
for t=1:model.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:)); %,'facealpha',.2
	plot2Dframe(eye(2)*4E-1, model.Mu(1:2,t), repmat(colMat(t,:),3,1), 6);
end
plot(r.f(1,:), r.f(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(r.f(1,1), r.f(2,1), '.','markersize',40,'color',[0 0 0]);
plot(r.f(1,tl), r.f(2,tl), '.','markersize',30,'color',[0 0 0]);
axis equal; 
% print('-dpng','graphs/DDP_humanoid01.png');


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
% 	axis([min(t), max(t(1:end-1)), min(r.u(kk,:))-.1, max(r.u(kk,:))+.1]);
% 	xlabel('t'); ylabel(['u_' num2str(kk)]);
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
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
	e(1:2,:) = f(1:2,:) - f0(1:2,:);
	e(3,:) = imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'));
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics (in robot coordinate system)
function f = fkine0(x, model)
	f = model.l * exp(1i * tril(ones(model.nbVarX)) * x); 
	f = [real(f); imag(f); mod(sum(x)+pi, 2*pi) - pi]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics (in object coordinate system)
function f = fkine(x, model)
% 	f = fkine0(x, model) - model.Mu; %Error by ignoring manifold
	f = logmap(fkine0(x, model), model.Mu); %Error by considering manifold
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = jacob0(x, model)
	J = 1i * exp(1i * tril(ones(model.nbVarX)) * x).' * tril(ones(model.nbVarX)) * diag(model.l); 
	J = [real(J); imag(J); ones(1, model.nbVarX)]; %x1,x2,o
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation
function J = jacob(x, f, model)
	J = []; 
	for t=1:size(x,2)
		Jtmp = jacob0(x(:,t), model);
		J = blkdiag(J, Jtmp);
	end
end