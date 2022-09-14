function demo_Riemannian_SPD_vecTransp02
% Vector transport on the symmetric positive definite (SPD) manifold S²+ using Schild's ladder algorithm
% (covariance matrices of dimension 2x2)
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon20RAM,
% 	author="Calinon, S.",
% 	title="Gaussians on {R}iemannian Manifolds: Applications for Robot Learning and Adaptive Control",
% 	journal="{IEEE} Robotics and Automation Magazine ({RAM})",
% 	year="2020",
% 	month="June",
% 	volume="27",
% 	number="2",
% 	pages="33--45",
% 	doi="10.1109/MRA.2020.2980548"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, https://idiap.ch/
% Written by Noémie Jaquier and Sylvain Calinon
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


%% Generate data on SPD manifold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDrawingSeg = 40; %Number of segments used to draw ellipsoids
nbIter_sl = 4; % Number of iterations for the Schild's ladder algorithm
sigma = 1; % Parameter for the Schild's ladder algorithm

% nbData = 3;
% Data = 3.*randn(2,2,nbData);
% Data = (Data + permute(Data,[2,1,3]))./2;
% Mu0 = [3,1;1,5];
% X = expmap(Data,Mu0);

X(:,:,1) = [2, -0.14; -0.14, 7];
X(:,:,2) = [14.5, -6; -6 7.5];
X(:,:,3) = [3, 0.5; 0.5, 2.7];

x1 = X(:,:,1);
x2 = X(:,:,2);
a = logmap(X(:,:,3),X(:,:,1));


%% Transport of a from x1 to x2 using Schild's ladder algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the SPD convex cone
figure('PaperPosition',[0 0 10 5],'position',[10,10,1300,650]); hold on;% rotate3d on;
axis equal;

r = 20;
phi = 0:0.1:2*pi+0.1;
alpha = [zeros(size(phi)); r.*ones(size(phi))];
beta = [zeros(size(phi));r.*sin(phi)];
gamma = [zeros(size(phi));r/sqrt(2).*cos(phi)];

h = mesh(alpha,beta,gamma,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])

h = plot3(alpha(2,:),beta(2,:),gamma(2,:),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(alpha(:,63),beta(:,63),gamma(:,63),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(alpha(:,40),beta(:,40),gamma(:,40),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])

set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gca,'ZTick',[]);
xlabel('\alpha'); ylabel('\gamma'); zlabel('\beta');
view(70,12);

% Plot datapoints on the manifold
plot3(x1(1,1,1), x1(2,2,1), x1(2,1,1), '.','markersize',12,'color',[.5 .5 .5]);
plot3(x2(1,1,1), x2(2,2,1), x2(2,1,1), '.','markersize',12,'color',[.5 .5 .5]);

% Plot vector to transport
plot3([x1(1,1,:) a(1,1,:)], [x1(2,2,:) a(2,2,:)], [x1(2,1,:) a(2,1,:)],'linewidth',1,'color',[.8 0 0]);

msh_y = geodesic(logmap(x2,x1),x1,linspace(0,1,nbDrawingSeg*nbIter_sl));

% Initialisation
t = 0:1/nbIter_sl:1;
y = geodesic(logmap(x2,x1),x1,t);
q(:,:,1) = a; % vector on the tangent space
z(:,:,1) = geodesic(q(:,:,1),y(:,:,1),sigma); % projection of this vector on the manifold

msh_q = expmap(q(:,:,1).*repmat(reshape(linspace(0,1,nbDrawingSeg),[1,1,nbDrawingSeg]),2,2,1), x1);
plot3(permute(msh_q(1,1,:),[3,1,2]), permute(msh_q(2,2,:),[3,1,2]), permute(msh_q(2,1,:),[3,1,2]), '-','linewidth',1,'color',[0 0 .9]);

% Iterate
for i=1:nbIter_sl
	% Plot current point y_(i+1) on the geodesic between x0 and x1
	plot3(y(1,1,i+1), y(2,2,i+1), y(2,1,i+1), '.','markersize',15,'color',[.3 .3 .3]);
	plot3(permute(msh_y(1,1,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg),[3,1,2]), permute(msh_y(2,2,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg),[3,1,2]), permute(msh_y(2,1,(i-1)*nbDrawingSeg+1:i*nbDrawingSeg),[3,1,2]), '-','linewidth',1,'color',[.3 .3 .3]);

	% Compute the middle point between y_(i+1) and z_(i)
	m_dir = logmap(y(:,:,i+1),z(:,:,i));
	m(:,:,i) = geodesic(m_dir,z(:,:,i),.5);
	msh_m = expmap(m_dir.*repmat(reshape(linspace(0,1,nbDrawingSeg),[1,1,nbDrawingSeg]),2,2,1), z(:,:,i));
	plot3(permute(msh_m(1,1,:),[3,1,2]), permute(msh_m(2,2,:),[3,1,2]), permute(msh_m(2,1,:),[3,1,2]), '-','linewidth',1,'color',[0.9 0.4 0]);
	plot3(m(1,1,i), m(2,2,i), m(2,1,i), '.','markersize',15,'color',[.9 .4 0]);

	% Compute z_(i+1) on the geodesic starting at y_(i) passing through
	% m_(i)
	z_dir = logmap(m(:,:,i),y(:,:,i));
	z(:,:,i+1) = geodesic(z_dir,y(:,:,i),2*sigma);
	plot3(z(1,1,i+1), z(2,2,i+1), z(2,1,i+1), '.','markersize',15,'color',[0 .8 0]);
	msh_z = expmap(logmap(z(:,:,i+1),y(:,:,i)).*repmat(reshape(linspace(0,1,nbDrawingSeg),[1,1,nbDrawingSeg]),2,2,1), y(:,:,i));
	plot3(permute(msh_z(1,1,:),[3,1,2]), permute(msh_z(2,2,:),[3,1,2]), permute(msh_z(2,1,:),[3,1,2]), '-','linewidth',1,'color',[0 .8 0]);
	
	% Compute the current transported vector on the tangent space of
	% y_(i+1)
	q(:,:,i+1) = logmap(z(:,:,i+1),y(:,:,i+1));
	
	msh_q = expmap(q(:,:,i+1).*repmat(reshape(linspace(0,1,nbDrawingSeg),[1,1,nbDrawingSeg]),2,2,1), y(:,:,i+1));
	plot3(permute(msh_q(1,1,:),[3,1,2]), permute(msh_q(2,2,:),[3,1,2]), permute(msh_q(2,1,:),[3,1,2]), '-','linewidth',1,'color',[0 0 .9]);
	plot3([y(1,1,i+1) q(1,1,i+1)], [y(2,2,i+1) q(2,2,i+1)], [y(2,1,i+1) q(2,1,i+1)],'linewidth',1,'color',[.8 0 0]);
end
% Final transported vector
a_schildsladderTransp = q(:,:,end);
pause();


%% Parallel transport of a from x1 to x2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_prlTransp = transp(x1,x2)*a*transp(x1,x2)';

% Plot the SPD convex cone
figure('PaperPosition',[0 0 10 5],'position',[10,10,1300,650]); hold on;% rotate3d on;
axis equal;

r = 20;
phi = 0:0.1:2*pi+0.1;
alpha = [zeros(size(phi)); r.*ones(size(phi))];
beta = [zeros(size(phi));r.*sin(phi)];
gamma = [zeros(size(phi));r/sqrt(2).*cos(phi)];

h = mesh(alpha,beta,gamma,'linestyle','none','facecolor',[.95 .95 .95],'facealpha',.5);
direction = cross([1 0 0],[1/sqrt(2),1/sqrt(2),0]);
rotate(h,direction,45,[0,0,0])

h = plot3(alpha(2,:),beta(2,:),gamma(2,:),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(alpha(:,63),beta(:,63),gamma(:,63),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])
h = plot3(alpha(:,40),beta(:,40),gamma(:,40),'linewidth',2,'color',[0 0 0]);
rotate(h,direction,45,[0,0,0])

set(gca,'XTick',[]);
set(gca,'YTick',[]);
set(gca,'ZTick',[]);
xlabel('\alpha'); ylabel('\gamma'); zlabel('\beta');
view(70,12);

% Plot datapoints on the manifold
plot3(x1(1,1,1), x1(2,2,1), x1(2,1,1), '.','markersize',12,'color',[.5 .5 .5]);
plot3(x2(1,1,1), x2(2,2,1), x2(2,1,1), '.','markersize',12,'color',[.5 .5 .5]);

% Plot vector to transport
plot3([x1(1,1,:) a(1,1,:)], [x1(2,2,:) a(2,2,:)], [x1(2,1,:) a(2,1,:)],'linewidth',1,'color',[.8 0 0]);

% Plot transported vector
plot3([x2(1,1,:) a_schildsladderTransp(1,1,:)], [x2(2,2,:) a_schildsladderTransp(2,2,:)], [x2(2,1,:) a_schildsladderTransp(2,1,:)],'linewidth',1,'color',[.8 0 0]);
plot3([x2(1,1,:) a_prlTransp(1,1,:)], [x2(2,2,:) a_prlTransp(2,2,:)], [x2(2,1,:) a_prlTransp(2,1,:)],'linewidth',1,'color',[0 0 .8]);

pause();
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(U,S)
	% Exponential map (SPD manifold)
	N = size(U,3);
	for n = 1:N
		X(:,:,n) = S^.5 * expm(S^-.5 * U(:,:,n) * S^-.5) * S^.5;
	end
end

function U = logmap(X,S)
	% Logarithm map 
	N = size(X,3);
	for n = 1:N
	% 	U(:,:,n) = S^.5 * logm(S^-.5 * X(:,:,n) * S^-.5) * S^.5;
	% 	U(:,:,n) = S * logm(S\X(:,:,n));
		[v,d] = eig(S\X(:,:,n));
		U(:,:,n) = S * v*diag(log(diag(d)))*v^-1;
	end
end

function Ac = transp(S1,S2)
	% Parallel transport (SPD manifold)
	% t = 1;
	% U = logmap(S2,S1);
	% Ac = S1^.5 * expm(0.5 .* t .* S1^-.5 * U * S1^-.5) * S1^-.5;
	Ac = (S2/S1)^.5;
end

function g = geodesic(U,S,t)
	for i=1:length(t)
		g(:,:,i) = S^.5 * expm(t(i) .* S^-.5 * U * S^-.5) * S^.5;
	end
end