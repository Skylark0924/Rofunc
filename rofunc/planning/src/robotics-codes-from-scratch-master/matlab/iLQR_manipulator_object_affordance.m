%%	  Batch iLQR applied to an object affordance planning problem with a planar manipulator, by considering object boundaries.
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

function iLQR_manipulator_object_affordance

%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.dt = 1E-2; %Time step size
param.nbData = 50; %Number of datapoints
param.nbIter = 300; %Maximum number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarU = 3; %Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3; %Task space dimension (f1,f2,f3, with f3 as orientation)
param.l = [3; 2; 1]; %Robot links lengths
param.sz = [.2, .3]; %Size of objects
param.q = 1E0; %Tracking weighting term
param.r = 1E-5; %Control weighting term

param.Mu = [[2; 1; -pi/6], [3; 2; -pi/3]]; %Viapoints
for t=1:param.nbPoints
    param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Orientation
end

R = speye(param.nbVarU * (param.nbData-1)) * param.r; %Control weight matrix (at trajectory level)

% Q = kron(eye(param.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)
Q = speye(param.nbVarF * param.nbPoints) * param.q; %Precision matrix
Qc = Q; %Object affordance constraint matrix

%Constraining the offset of the first and second viapoint to be correlated
Qc(1:2, 4:5) = -eye(2) * 1E0;
Qc(4:5, 1:2) = -eye(2) * 1E0;

% %Constraining the offset and orientation of the first and second viapoint to be correlated
% Q(1:3, 7:9) = -eye(3) * 1E0;
% Q(7:9, 1:3) = -eye(3) * 1E0;

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial control commands
x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose

%Transfer matrices (for linear system as single integrator)
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); kron(tril(ones(param.nbData-1)), eye(param.nbVarX)*param.dt)];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter	
    x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
    [f, J] = f_reach(x(:,tl), param);
    [fc, Jc] = f_reach(x(:,tl), param, 0); %Object affordance
    du = (Su' * J' * Q * J * Su + Su' * Jc' * Qc * Jc * Su + R) \ (-Su' * J' * Q * f(:) -Su' * Jc' * Qc * fc(:) - u * param.r); %Gradient
    
    %Estimate step size with backtracking line search method
    alpha = 1;
    cost0 = f(:)' * Q * f(:) + fc(:)' * Qc * fc(:) + norm(u)^2 * param.r; %u' * R * u
    while 1
        utmp = u + du * alpha;
        xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
        ftmp = f_reach(xtmp(:,tl), param);
        fctmp = f_reach(xtmp(:,tl), param, 0); %Object affordance
        cost = ftmp(:)' * Q * ftmp(:) + fctmp(:)' * Qc * fctmp(:) + norm(utmp)^2 * param.r; %utmp' * R * utmp
        if cost < cost0 || alpha < 1E-3
            break;
        end
        alpha = alpha * 0.5;
    end
    u = u + du * alpha;
    
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

h = figure('position',[10,10,800,800],'color',[1,1,1]); hold on; axis off;
ftmp = fkin0(x(:,1), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.8 .8 .8]);
ftmp = fkin0(x(:,tl(1)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.4 .4 .4]);
ftmp = fkin0(x(:,tl(end)), param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',4,'color',[.0 .0 .0]);

colMat = lines(param.nbPoints);
for t=1:param.nbPoints
    patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:)); 
end
ftmp = fkin(x, param); 
plot(ftmp(1,:), ftmp(2,:), 'k-','linewidth',2);
plot(ftmp(1,1), ftmp(2,1), 'k.','markersize',40);
plot(ftmp(1,tl), ftmp(2,tl), 'k.','markersize',30);
axis equal; 

waitfor(h);
end 

%%%%%%%%%%%%%%%%%%%%%%
%Logarithmic map for R^2 x S^1 manifold
function e = logmap(f, f0)
    e(1:2,:) = f0(1:2,:) - f(1:2,:);
    e(3,:) = imag(log(exp(f0(3,:)*1i).' .* exp(f(3,:)*1i)'));
end

%%%%%%%%%%%%%%%%%%%%%%
%Forward kinematics (in robot coordinate system)
function f = fkin(x, param)
    L = tril(ones(size(x,1)));
    f = [param.l' * cos(L * x); ...
         param.l' * sin(L * x); ...
         mod(sum(x,1)+pi, 2*pi) - pi]; %f1,f2,f3 (f3 represents orientation as single Euler angle for planar robot)
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
%Jacobian of forward kinematics function with analytical computation
function J = Jkin(x, param)
    L = tril(ones(size(x,1)));
    J = [-sin(L * x)' * diag(param.l) * L; ...
          cos(L * x)' * diag(param.l) * L; ...
          ones(1, size(x,1))]; %f1,f2,f3 (f3 represents orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param, bb)
    if nargin<3
        bb = 1; %Bounding boxes considered by default
    end
% 	f = fkin(x, param) - param.Mu; %Error by ignoring manifold
    f = logmap(param.Mu, fkin(x, param)); %Error by considering manifold
    
    J = []; 
    for t=1:size(x,2)
        f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered FK
        
        Jtmp = Jkin(x(:,t), param);
        %Jtmp = Jkin_num(x(:,t), param);
        
        Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
        
        %Bounding boxes 
        if bb==1
            for i=1:2
                if abs(f(i,t)) < param.sz(i)
                    f(i,t) = 0;
                    Jtmp(i,:) = 0;
                else
                    f(i,t) = f(i,t) - sign(f(i,t)) * param.sz(i);
                end
            end
        end
        
        J = blkdiag(J, Jtmp);
    end
end
