%%    iLQR with control primitives applied to a planar manipulator for a viapoints task (batch formulation)
%%
%%    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
%%    Written by Sylvain Calinon <https://calinon.ch>
%%
%%    This file is part of RCFS.
%%
%%    RCFS is free software: you can redistribute it and/or modify
%%    it under the terms of the GNU General Public License version 3 as
%%    published by the Free Software Foundation.
%%
%%    RCFS is distributed in the hope that it will be useful,
%%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%%    GNU General Public License for more details.
%%
%%    You should have received a copy of the GNU General Public License
%%    along with RCFS. If not, see <http://www.gnu.org/licenses/>.

function iLQR_manipulator_CP

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 100; %Number of datapoints
param.nbIter = 300; %Maximum number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarU = 3; %Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3; %Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2; 2; 1]; %Links length
param.sz = [.2, .3]; %Size of objects
param.r = 1E-6; %Control weight term
param.nbFct = 4; %Number of basis functions (odd number for Fourier basis functions)
param.basisName = 'RBF'; %PIECEWISE, RBF, BERNSTEIN, FOURIER

param.Mu = [[2; 1; -pi/6], [3; 2; -pi/3]]; %Viapoints 
%param.Mu = [[2; 1; -pi/2], [3; 1; -pi/2]]; %Viapoints
for t=1:param.nbPoints
    param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Orientation
end

R = speye((param.nbData-1)*param.nbVarU) * param.r; %Control weight matrix (at trajectory level)
Q = speye(param.nbVarF * param.nbPoints) * 1E0; %Precision matrix
% Q = kron(eye(param.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Basis functions Psi
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isequal(param.basisName,'PIECEWISE')
    phi = buildPhiPiecewise(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'RBF')
    phi = buildPhiRBF(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'BERNSTEIN')
    phi = buildPhiBernstein(param.nbData-1, param.nbFct);
elseif isequal(param.basisName,'FOURIER')
    phi = buildPhiFourier(param.nbData-1, param.nbFct);
end

Psi = kron(phi, eye(param.nbVarU)); %Application of basis functions to multidimensional control commands


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial commands
x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); tril(kron(ones(param.nbData-1), eye(param.nbVarX)*param.dt))];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter	
    x = real(reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData)); %System evolution
    [f, J] = f_reach(x(:,tl), param);
% 	du = (Su' * J' * Q * J * Su + R) \ (-Su' * J' * Q * f(:) - R * u(:));
    w = (Psi' * (Su' * J' * Q * J * Su + R) * Psi) \ (-Psi' * (Su' * J' * Q * f(:) + R * u(:)));
    du = Psi * w;
    
    %Estimate step size with line search method
    alpha = 1;
    cost0 = f(:)' * Q * f(:) + norm(u)^2 * param.r; %u' * R * u
    while 1
        utmp = u + du * alpha;
        xtmp = real(reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData));
        ftmp = f_reach(xtmp(:,tl), param);
        cost = ftmp(:)' * Q * ftmp(:) + norm(utmp)^2 * param.r; %utmp' * R * utmp
        if cost < cost0 || alpha < 1E-3
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


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
msh0 = diag(param.sz) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:param.nbPoints
    msh(:,:,t) = param.A(:,:,t) * msh0 + repmat(param.Mu(1:2,t), 1, size(msh0,2));
end

h(1) = figure('position',[10,10,1200,1200],'color',[1,1,1]); hold on; axis off;
ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(1)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
ftmp = fkin0(x(:,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.2 .2 .2]);
colMat = lines(param.nbPoints);
for t=1:param.nbPoints
    patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:));
end
ftmp = fkin(x, param); 
plot(ftmp(1,:), ftmp(2,:), 'k-','linewidth',2);
plot(ftmp(1,1), ftmp(2,1), 'k.','markersize',40);
plot(ftmp(1,tl), ftmp(2,tl), 'k.','markersize',30);
axis equal; 

    
%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
h(2) = figure('position',[1350 10 1200 1200],'color',[1 1 1]); 
%States plot
for j=1:param.nbVarF
    subplot(param.nbVarF+param.nbVarU+1, 1, j); hold on;
    plot(tl, param.Mu(j,:), 'r.','markersize',35);
    plot(ftmp(j,:), 'k-','linewidth',2);
    ylabel(['f_' num2str(j)],'fontsize',26);
end
%Commands plot
u = real(reshape(u, param.nbVarU, param.nbData-1));
for j=1:param.nbVarU
    subplot(param.nbVarF+param.nbVarU+1, 1, param.nbVarF+j); hold on;
    plot(u(j,:), 'k-','linewidth',2);
    ylabel(['u_' num2str(j)],'fontsize',26);
end
%Basis functions plot (display only the real part for Fourier basis functions)
subplot(param.nbVarF+param.nbVarU+1, 1, param.nbVarF+param.nbVarU+1); hold on; 
clrmap = lines(param.nbFct);
for i=1:param.nbFct
    plot(real(phi(:,i)), '-','linewidth',3,'color',clrmap(i,:)); 
end
% set(gca,'xtick',[],'ytick',[]); %axis tight; 
xlabel('t','fontsize',26); 
ylabel('\phi_k','fontsize',26);

waitfor(h);
end 


%Building piecewise constant basis functions
function phi = buildPhiPiecewise(nbData, nbFct) 
	phi = kron(eye(nbFct), ones(ceil(nbData/nbFct),1));
	phi = phi(1:nbData,:);
end

%Building radial basis functions (RBFs)
function phi = buildPhiRBF(nbData, nbFct) 
	t = linspace(0, 1, nbData);
	tMu = linspace(t(1)-1/(nbFct-3), t(end)+1/(nbFct-3), nbFct); %Repartition of centers to limit border effects
	sigma = 1 / (nbFct-2); %Standard deviation
	phi = exp(-(t' - tMu).^2 / sigma^2);
	
	%Optional rescaling
	%phi = phi ./ repmat(sum(phi,2), 1, nbFct); 
end

%Building Bernstein basis functions
function phi = buildPhiBernstein(nbData, nbFct)
	t = linspace(0, 1, nbData);
	phi = zeros(nbData, nbFct);
	for i=1:nbFct
		phi(:,i) = factorial(nbFct-1) ./ (factorial(i-1) .* factorial(nbFct-i)) .* (1-t).^(nbFct-i) .* t.^(i-1);
	end
end

%Building Fourier basis functions
function phi = buildPhiFourier(nbData, nbFct)
	t = linspace(0, 1, nbData);
	
	%Computation for general signals (incl. complex numbers)
	d = ceil((nbFct-1)/2);
	k = -d:d;
	phi = exp(t' * k * 2 * pi * 1i); 
	%phi = cos(t' * k * 2 * pi); %Alternative computation for real signal
		
%	%Alternative computation for real and even signal
%	k = 0:nbFct-1;
%	phi = cos(t' * k * 2 * pi);
%	%phi(:,2:end) = phi(:,2:end) * 2;
%	%invPhi = cos(k' * t * 2 * pi) / nbData;
%	%invPsi = kron(invPhi, eye(param.nbVar));
end

%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
    e(1:2,:) = f(1:2,:) - f0(1:2,:);
    e(3,:) = imag(log(exp(f0(3,:)*1i)' .* exp(f(3,:)*1i).'));
end

%Forward kinematics for end-effector (in robot coordinate system)
function f = fkin(x, param)
	L = tril(ones(size(x,1)));
	f = [param.l' * cos(L * x); ...
		param.l' * sin(L * x); ...
		mod(sum(x,1)+pi, 2*pi) - pi]; %x1,x2,o (orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics for all robot articulations (in robot coordinate system)
function f = fkin0(x, param)
	L = tril(ones(size(x,1)));
	f = [L * diag(param.l) * cos(L * x), ...
	     L * diag(param.l) * sin(L * x)]'; 
	f = [zeros(2,1), f];
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian with analytical computation (for single time step)
function J = Jkin(x, param)
	L = tril(ones(size(x,1)));
	J = [-sin(L * x)' * diag(param.l) * L; ...
	      cos(L * x)' * diag(param.l) * L; ...
	      ones(1, size(x,1))]; %x1,x2,o
end

%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
% 	f = fkin(x, param) - param.Mu; %Error by ignoring manifold
	f = logmap(fkin(x, param), param.Mu); %Error by considering manifold
	
	J = []; 
	for t=1:size(x,2)
		f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered FK
			
		Jtmp = Jkin(x(:,t), param);
		Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
%		%Bounding boxes (optional)
%		for i=1:2
%			if abs(f(i,t)) < param.sz(i)
%				f(i,t) = 0;
%				Jtmp(i,:) = 0;
%			else
%				f(i,t) = f(i,t) - sign(f(i,t)) * param.sz(i);
%			end
%		end
		
		J = blkdiag(J, Jtmp);
	end
end
