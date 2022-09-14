function demo_OC_DDP_manipulator01
% iLQR applied to a planar manipulator  
% (viapoints task with position+orientation, including bounding boxes on f(x))
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
param.dt = 1E-2; %Time step size
param.nbData = 50; %Number of datapoints
param.nbIter = 300; %Maximum number of iterations for iLQR
param.nbPoints = 2; %Number of viapoints
param.nbVarX = 3; %State space dimension (x1,x2,x3)
param.nbVarU = 3; %Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3; %Task space dimension (f1,f2,f3, with f3 as orientation)
param.l = [3; 2; 1]; %Robot links lengths
param.fmax = [.2, .3, .01]; %Bounding boxes parameters
param.q = 1E0; %Tracking weighting term
param.r = 1E-5; %Control weighting term

% param.Mu = [[2; 1; -pi/2], [3; 1; -pi/2]]; %Viapoints
param.Mu = [[2; 1; -pi/6], [3; 2; -pi/3]]; %Viapoints
% param.Mu = [3; 0; -pi/2]; %Viapoints 
for t=1:param.nbPoints
	param.A(:,:,t) = [cos(param.Mu(3,t)), -sin(param.Mu(3,t)); sin(param.Mu(3,t)), cos(param.Mu(3,t))]; %Orientation
end

Q = speye(param.nbVarF * param.nbPoints) * param.q; %Precision matrix
% Q = kron(eye(param.nbPoints), diag([1E0, 1E0, 0])); %Precision matrix (by removing orientation constraint)
R = speye(param.nbVarU * (param.nbData-1)) * param.r; %Control weight matrix (at trajectory level)

%Time occurrence of viapoints
tl = linspace(1, param.nbData, param.nbPoints+1);
tl = round(tl(2:end));
idx = (tl - 1) * param.nbVarX + [1:param.nbVarX]';


%% Iterative LQR (iLQR)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
u = zeros(param.nbVarU*(param.nbData-1), 1); %Initial control commands
x0 = [3*pi/4; -pi/2; -pi/4]; %Initial robot pose

%Transfer matrices (for linear system as single integrator)
% A = repmat(eye(param.nbVarX), [1 1 param.nbData-1]);
% B = repmat(eye(param.nbVarX, param.nbVarU) * param.dt, [1 1 param.nbData-1]); 
% [Su0, Sx0] = transferMatrices(A, B); %Constant Su and Sx for the proposed system
Su0 = [zeros(param.nbVarX, param.nbVarX*(param.nbData-1)); kron(tril(ones(param.nbData-1)), eye(param.nbVarX)*param.dt)];
Sx0 = kron(ones(param.nbData,1), eye(param.nbVarX));
Su = Su0(idx,:);

for n=1:param.nbIter	
	x = reshape(Su0 * u + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
	[f, J] = f_reach(x(:,tl), param); %Residuals and Jacobian
	du = (Su' * J' * Q * J * Su + R) \ (-Su' * J' * Q * f(:) - u * param.r); %Gauss-Newton update
	
	%Estimate step size with backtracking line search method
	alpha = 1;
	cost0 = f(:)' * Q * f(:) + norm(u)^2 * param.r; %u' * R * u
	while 1
		utmp = u + du * alpha;
		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData);
		ftmp = f_reach(xtmp(:,tl), param);
		cost = ftmp(:)' * Q * ftmp(:) + norm(utmp)^2 * param.r; %utmp' * R * utmp
		if cost < cost0 || alpha < 1E-3
			break;
		end
		alpha = alpha * 0.5;
	end

%	%Estimate step size with backtracking line search method
%	cost0 = f(:)' * Q * f(:) + norm(u)^2 * param.r; %u' * R * u
%	cost = 1E10;
%	alpha = 2;
%	while (cost > cost0 && alpha > 1E-3)
%		alpha = alpha * 0.5;
%		utmp = u + du * alpha;
%		xtmp = reshape(Su0 * utmp + Sx0 * x0, param.nbVarX, param.nbData); %System evolution
%		ftmp = f_reach(xtmp(:,tl), param); %Residuals
%		cost = ftmp(:)' * Q * ftmp(:) + norm(utmp)^2 * param.r; %Cost
%	end
	
	u = u + du * alpha;
	
	if norm(du * alpha) < 1E-2
		break; %Stop iLQR when solution is reached
	end
end
disp(['iLQR converged in ' num2str(n) ' iterations.']);


%% Plot state space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
msh0 = diag(param.fmax(1:2)) * [-1 -1 1 1 -1; -1 1 1 -1 -1];
for t=1:param.nbPoints
	msh(:,:,t) = param.A(:,:,t) * msh0 + repmat(param.Mu(1:2,t), 1, size(msh0,2));
end

h = figure('position',[10,10,1600,1200],'color',[1,1,1]); hold on; axis off;
plotArm(x(:,1), param.l, [0; 0; -3], .2, [.8 .8 .8]);
plotArm(x(:,tl(1)), param.l, [0; 0; -2], .2, [.6 .6 .6]);
plotArm(x(:,tl(2)), param.l, [0; 0; -1], .2, [.4 .4 .4]);
colMat = lines(param.nbPoints);
for t=1:param.nbPoints
	patch(msh(1,:,t), msh(2,:,t), min(colMat(t,:)*1.7,1),'linewidth',2,'edgecolor',colMat(t,:)); %,'facealpha',.2
	plot2Dframe(param.A(:,:,t)*5E-1, param.Mu(1:2,t), repmat(colMat(t,:),3,1), 2);
end
%plot(param.Mu(1,:), param.Mu(2,:), '.','markersize',40,'color',[.8 0 0]);
ftmp = fkin(x, param);
plot(ftmp(1,:), ftmp(2,:), '-','linewidth',2,'color',[0 0 0]);
plot(ftmp(1,1), ftmp(2,1), '.','markersize',40,'color',[0 0 0]);
axis equal; %axis([-2.4 3.5 -0.6 3.8]);
%print('-dpng','graphs/DDP_manipulator01.png');


%%% Timeline plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%figure('position',[820 10 800 800],'color',[1 1 1]); 
%%States plot
%for j=1:param.nbVarF
%	subplot(param.nbVarF+param.nbVarU, 1, j); hold on;
%	plot(tl, param.Mu(j,:), '.','markersize',35,'color',[.8 0 0]);
%	plot(r.f(j,:), '-','linewidth',2,'color',[0 0 0]);
%% 	set(gca,'xtick',[1,nbData],'xticklabel',{'0','T'},'ytick',[0:2],'fontsize',20);
%	ylabel(['$f_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
%% 	axis([1, nbData, 0, 2.1]);
%end
%%Commands plot
%r.u = reshape(u, param.nbVarU, param.nbData-1);
%for j=1:param.nbVarU
%	subplot(param.nbVarF+param.nbVarU, 1, param.nbVarF+j); hold on;
%	plot(r.u(j,:), '-','linewidth',2,'color',[0 0 0]);
%% 	set(gca,'xtick',[1,nbData],'xticklabel',{'0','T'},'ytick',[0:2],'fontsize',20);
%	ylabel(['$u_' num2str(j) '$'], 'interpreter','latex','fontsize',26);
%% 	axis([1, nbData, 0, 2.1]);
%end
%xlabel('$t$','interpreter','latex','fontsize',26); 


%%% Plot value function for reaching costs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nbRes = 100;
%limAxes = [-2.4 3.5 -0.6 3.8];
%[xx, yy] = ndgrid(linspace(limAxes(1),limAxes(2),nbRes), linspace(limAxes(3),limAxes(4),nbRes));
%x = [xx(:)'; yy(:)'];
%msh = [min(x(1,:)), min(x(1,:)), max(x(1,:)), max(x(1,:)), min(x(1,:)); ...
%	   min(x(2,:)), max(x(2,:)), max(x(2,:)), min(x(2,:)), min(x(2,:))];
%%Reaching at T/2
%f = f_reach2(x, param.Mu(1:2,1), param.A(:,:,1), param.sz);
%z = sum(f.^2, 1);
%zz = reshape(z, nbRes, nbRes);
%%Reaching at T
%f2 = f_reach2(x, param.Mu(1:2,2), param.A(:,:,2), param.sz);
%z2 = sum(f2.^2, 1);
%zz2 = reshape(z2, nbRes, nbRes);

%colMat = lines(param.nbPoints);
%h = figure('position',[820,10,860,400],'color',[1,1,1]); 
%colormap(repmat(linspace(1,.4,64),3,1)');
%%Reaching at T/2
%subplot(1,2,1); hold on; axis off; title('Reaching cost (for t=T/2)');
%surface(xx, yy, zz-max(zz(:)), 'EdgeColor','interp','FaceColor','interp'); 
%contour(xx, yy, zz, [0:.4:max(zz(:))].^2,'color',[.3,.3,.3]); 
%plot(param.Mu(1,1), param.Mu(2,1),'.','markersize',30,'color',colMat(1,:));
%plot(msh(1,:), msh(2,:),'-','linewidth',1,'color',[0 0 0]); %border
%%plotArm(x(:,tl(1)), param.l, [0; 0; -2], .2, [.6 .6 .6]);
%axis equal; axis(limAxes);
%%Reaching at T
%subplot(1,2,2); hold on; axis off; title('Reaching cost (for t=T)');
%surface(xx, yy, zz2-max(zz2(:)), 'EdgeColor','interp','FaceColor','interp'); 
%contour(xx, yy, zz2, [0:.4:max(zz2(:))].^2,'color',[.3,.3,.3]); 
%plot(param.Mu(1,2), param.Mu(2,2),'.','markersize',30,'color',colMat(2,:));
%plot(msh(1,:), msh(2,:),'-','linewidth',1,'color',[0 0 0]); %border
%axis equal; axis(limAxes);
%%print('-dpng','graphs/DDP_manipulator01b.png');

waitfor(h);
end 

% %%%%%%%%%%%%%%%%%%%%%%
% function [Su, Sx] = transferMatrices(A, B)
% 	[nbVarX, nbVarU, nbData] = size(B);
% 	nbData = nbData+1;
% % 	Sx = kron(ones(nbData,1), speye(nbVarX)); 
% 	Sx = speye(nbData*nbVarX, nbVarX); 
% 	Su = sparse(nbVarX*(nbData-1), nbVarU*(nbData-1));
% 	for t=1:nbData-1
% 		id1 = (t-1)*nbVarX+1:t*nbVarX;
% 		id2 = t*nbVarX+1:(t+1)*nbVarX;
% 		id3 = (t-1)*nbVarU+1:t*nbVarU;
% 		Sx(id2,:) = squeeze(A(:,:,t)) * Sx(id1,:);
% 		Su(id2,:) = squeeze(A(:,:,t)) * Su(id1,:);	
% 		Su(id2,id3) = B(:,:,t);	
% 	end
% end

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
%Jacobian of forward kinematics function with analytical computation
function J = Jkin(x, param)
	L = tril(ones(size(x,1)));
	J = [-sin(L * x)' * diag(param.l) * L; ...
	      cos(L * x)' * diag(param.l) * L; ...
	      ones(1, size(x,1))]; %f1,f2,f3 (f3 represents orientation as single Euler angle for planar robot)
end

%%%%%%%%%%%%%%%%%%%%%%
%Jacobian of forward kinematics function with numerical computation
function J = Jkin_num(x, param)
	e = 1E-6;
	
%	%Slow for-loop computation
%	J = zeros(param.nbVarF, param.nbVarX);
%	for n=1:size(x,1)
%		xtmp = x;
%		xtmp(n) = xtmp(n) + e;
%		J(:,n) = (fkine0(xtmp, param) - fkine0(x, param)) / e;
%	end
	
	%Fast matrix computation
	X = repmat(x, [1, param.nbVarX]);
	F1 = fkin(X, param);
	F2 = fkin(X + eye(param.nbVarX) * e, param);
	J = (F2 - F1) / e;
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost and gradient for a viapoints reaching task (in object coordinate system)
function [f, J] = f_reach(x, param)
% 	f = fkin(x, param) - param.Mu; %Error by ignoring manifold
	f = logmap(param.Mu, fkin(x, param)); %Error by considering manifold
	
	J = []; 
	for t=1:size(x,2)
		f(1:2,t) = param.A(:,:,t)' * f(1:2,t); %Object-centered FK
		
		Jtmp = Jkin(x(:,t), param);
		%Jtmp = Jkin_num(x(:,t), param);
		
		Jtmp(1:2,:) = param.A(:,:,t)' * Jtmp(1:2,:); %Object-centered Jacobian
		
		%Bounding boxes (optional)
		for i=1:length(param.fmax)
			if abs(f(i,t)) < param.fmax(i)
				f(i,t) = 0;
				Jtmp(i,:) = 0;
			else
				f(i,t) = f(i,t) - sign(f(i,t)) * param.fmax(i);
			end
		end
		
		J = blkdiag(J, Jtmp);
	end
end

%%%%%%%%%%%%%%%%%%%%%%
%Cost in task space for a viapoints reaching task (for visualization)
function f = f_reach2(f, Mu, A, sz)
	f = A' * (f - repmat(Mu,1,size(f,2))); %Object-centered error
	%Bounding boxes (optional)
	for i=1:2
		id = abs(f(i,:)) < sz(i);
		f(i,id) = 0;
		f(i,~id) = f(i,~id) - sign(f(i,~id)) * sz(i);
	end
end
