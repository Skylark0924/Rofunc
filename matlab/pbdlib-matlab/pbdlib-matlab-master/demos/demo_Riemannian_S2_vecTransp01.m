function demo_Riemannian_S2_vecTransp01
% Parallel transport on a sphere.
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
model.nbStates = 2; %Number of states in the GMM
model.nbVar = 2; %Dimension of the tangent space
model.nbVarMan = 3; %Dimension of the manifold


%% Setting GMM parameters manually
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Priors = ones(1,model.nbStates) / model.nbStates;

%model.MuMan = randn(model.nbVarMan,model.nbStates);
model.MuMan(:,1) = [1; -.5; 0];
model.MuMan(:,2) = [0; -1; .5];
for i=1:model.nbStates
	model.MuMan(:,i) = model.MuMan(:,i) / norm(model.MuMan(:,i));
end

model.Mu = zeros(model.nbVar,model.nbStates);

model.Sigma(:,:,1) = diag([2,4]) * 5E-2;
model.Sigma(:,:,2) = diag([2,4]) * 5E-2;

%Eigendecomposition of Sigma
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	U0(:,:,i) = V * D.^.5;
end


%% Transportation of covariance 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = model.MuMan(:,1);
h = model.MuMan(:,2);

% vm = rotM(g)' * [logmap(h,g); 0];
% m = norm(vm);
% u = vm/m;

tl = linspace(0,1,20);

for n=1:20
	t = tl(n);
	
% 	%With Freifeld parametrization (eq. (5.17), p.169 of pdf)
% 	hist(n).MuMan = p*cos(m*t) + u*sin(m*t); 
% 	Rpar = -g*sin(m*t)*u' + u*cos(m*t)*u' + eye(model.nbVarMan) - u*u'; 
% 	hist(n).U = E' * Rpar * E * U0(:,:,1); 
	
	hist(n).MuMan = expmap(logmap(h,g)*t, g);
	
	Ac = transp(g, hist(n).MuMan);
	hist(n).U = Ac * U0(:,:,1);
	hist(n).Sigma = hist(n).U * hist(n).U';
    
	% Direction of the geodesic
	hist(n).dirG = logmap(h, hist(n).MuMan);
	if norm(hist(n).dirG) > 1E-5
		% Normalise the direction
		hist(n).dirG = hist(n).dirG ./ norm(hist(n).dirG);
		% Compute the inner product with the first eigenvector
		inprod(n) = hist(n).dirG' * hist(n).U(:,1);
	end
    
end

% %Check that the two vectors below are the same
% p1 = -logmap(g,h);
% p2 = E' * rotM(h) * Rpar * rotM(g)' * E * logmap(h,g);
% norm(p1-p2)
% 
% %Check that the transported eigenvectors remain orthogonal
% hist(end).U(:,1)' * hist(end).U(:,2)

% Check that the inner product between the direction of the geodesic and
% parallel transported vectors (e.g. eigenvectors) is conserved.
inprod


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);
MuMan0 = [0; 0; 1];
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids


%Display of covariance contours on the sphere
t = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(t); sin(t)], model.MuMan(:,i));
end
for n=1:20
	[V,D] = eig(hist(n).Sigma);
	hist(n).Gdisp = expmap(V*D.^.5*[cos(t); sin(t)], hist(n).MuMan);
end

%Manifold plot
figure('position',[10,10,1200,1200]); hold on; axis off; grid off; rotate3d on; 
colormap([.9 .9 .9]);
nbp = 40;
[X,Y,Z] = sphere(nbp-1);
mesh(X,Y,Z);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',12,'color',clrmap(i,:));
	plot3Dframe(rotM(model.MuMan(:,i))'*0.08, model.MuMan(:,i));
end
%Plot transported covariance
for n=1:20
	plot3(hist(n).MuMan(1), hist(n).MuMan(2), hist(n).MuMan(3), '.','markersize',12,'color',clrmap(1,:)*n/20);
	plot3(hist(n).Gdisp(1,:), hist(n).Gdisp(2,:), hist(n).Gdisp(3,:), '-','linewidth',.2,'color',clrmap(1,:)*n/20);
	for j=1:model.nbVar
		msh = expmap(squeeze(hist(n).U(:,j))*linspace(0,1,nbDrawingSeg), hist(n).MuMan);
		plot3(msh(1,:), msh(2,:), msh(3,:), '-','linewidth',1,'color',clrmap(1,:)*n/20);
	end
end
plot3Dframe(rotM(MuMan0)'*0.08, MuMan0);
view(30,12); axis equal; axis vis3d;  

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, mu)
	x = rotM(mu)' * expfct(u);
end

function u = logmap(x, mu)
	if norm(mu-[0;0;-1])<1e-6
		R = [1 0 0; 0 -1 0; 0 0 -1];
	else
		R = rotM(mu);
	end
	u = logfct(R*x);
end

function Exp = expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2);
	Exp = real([u(1,:).*sin(normv)./normv; u(2,:).*sin(normv)./normv; cos(normv)]);
	Exp(:,normv < 1e-16) = repmat([0;0;1],1,sum(normv < 1e-16));	
end

function Log = logfct(x)
	scale = acos(x(3,:)) ./ sqrt(1-x(3,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(1,:).*scale; x(2,:).*scale];	
end

function Ac = transp(g,h)
	E = [eye(2); zeros(1,2)];
	vm = rotM(g)' * [logmap(h,g); 0];
	mn = norm(vm);
	uv = vm / (mn+realmin);
	Rpar = eye(3) - sin(mn)*(g*uv') - (1-cos(mn))*(uv*uv');	
	Ac = E' * rotM(h) * Rpar * rotM(g)' * E; %Transportation operator from g to h 
end