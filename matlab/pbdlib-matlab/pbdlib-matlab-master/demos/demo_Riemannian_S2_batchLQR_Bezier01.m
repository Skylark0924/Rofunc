function demo_Riemannian_S2_batchLQR_Bezier01
% Bezier interpolation on a sphere by relying on batch LQR with Riemannian manifold 
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
nbData = 100; %Number of datapoints
nbRepros = 1; %Number of reproductions

model.nbStates = 2; %Number of states in the GMM
model.nbVarPos = 2; %Dimension of data in tangent space (here: v1,v2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.nbVarMan = 2*model.nbVarPos+1; %Dimension of the manifold (here: x1,x2,x3,v1,v2)
model.dt = 1E-1; % %Time step duration
model.rfactor = 1E-10;	%Control cost in LQR 
e0 = [0; 0; 1]; %Origin on manifold

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);


%% Discrete dynamical System settings (in tangent space)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarPos)); %Discrete nD
B = kron(B1d, eye(model.nbVarPos)); %Discrete nD

%Build Sx and Su matrices for batch LQR
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar));
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Generate cubic Bezier curve parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.P = rand(model.nbVarPos, model.nbStates*2);
model.P(:,1) = model.P(:,2) + randn(model.nbVarPos,1) * 1E-1;
model.P(:,3) = model.P(:,4) + randn(model.nbVarPos,1) * 1E-1;

model.Mu = [model.P(:,[1,end]); zeros(model.nbVarPos, model.nbStates)];
model.Mu(3:4,2) = (model.P(1:2,4) - model.P(1:2,3)) * 3;
model.Mu(3:4,1) = (model.P(1:2,2) - model.P(1:2,1)) * 3;
model.invSigma = repmat(eye(model.nbVar)*1E0, [1,1,model.nbStates]);

model.MuMan = [expmap(model.Mu(1:model.nbVarPos,:), e0); model.Mu(model.nbVarPos+1:end,:)]; %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent space at point MuMan of the manifold

% %Cubic Bezier curve plot from Bernstein polynomials
% %See e.g. http://blogs.mathworks.com/graphics/2014/10/13/bezier-curves/ or 
% %http://www.ams.org/samplings/feature-column/fcarc-bezier#2
% t = linspace(0,1,nbData);
% x = kron((1-t).^3, model.P(:,1)) + kron(3*(1-t).^2.*t, model.P(:,2)) + kron(3*(1-t).*t.^2, model.P(:,3)) + kron(t.^3, model.P(:,4));
% %dx = kron(3*(1-t).^2, model.P(:,2)-model.P(:,1)) + kron(6*(1-t).*t, model.P(:,3)-model.P(:,2)) + kron(3*t.^2, model.P(:,4)-model.P(:,3));
% % model.Mu(3:4,1) = dx(1:2,1);
% % model.Mu(3:4,2) = dx(1:2,end);


%% Batch LQR (computation centered on x, by creating single Gaussian N(MuQ,Q^-1))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbRepros
	x = model.MuMan(1:3,1);
	U = [zeros(model.nbVarPos,1); model.MuMan(4:5,1)]; 
	
	Ac = transp(model.MuMan(1:3,2), x);
	dxTmp = Ac * model.MuMan(4:5,2);

	MuQ = [zeros(model.nbVarPos,1); model.MuMan(4:5,1); ...
		zeros(model.nbVar*(nbData-2),1); ...
		logmap(model.MuMan(1:3,2),x); dxTmp]; %Only the first and last values will be used
	
	%Set cost for two viapoints at the beginning and at the end
% 	Q = blkdiag(model.invSigma(:,:,1), zeros(model.nbVar*(nbData-2)), blkdiag(Ac,Ac) * model.invSigma(:,:,2) * blkdiag(Ac,Ac)');
% 	Q = blkdiag(model.invSigma(:,:,1), zeros(model.nbVar*(nbData-2)), blkdiag(Ac,eye(model.nbVarPos)) * model.invSigma(:,:,2) * blkdiag(Ac,eye(model.nbVarPos))');
	Q = blkdiag(model.invSigma(:,:,1), zeros(model.nbVar*(nbData-2)), model.invSigma(:,:,2)); %No need to transport since covariance is isotropic

	%Compute acceleration commands
	SuInvSigmaQ = Su' * Q;
	Rq = SuInvSigmaQ * Su + R;
	rq = SuInvSigmaQ * (MuQ-Sx*U);
	ddu = Rq \ rq;

	U = reshape(Sx*U+Su*ddu, model.nbVar, nbData);
	r(n).x = expmap(U,x);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1650,1250]); hold on; axis off; grid off; rotate3d on; 
clrmap = lines(model.nbStates);
colormap([.7 .7 .7]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);
for i=1:model.nbStates
	%Plot tangent plane
	msh = repmat(model.MuMan(1:3,i),1,5) + rotM(model.MuMan(1:3,i))' * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
	patch(msh(1,:),msh(2,:),msh(3,:), [.8 .8 .8],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
	%Plot Bezier control points
	msh = [model.MuMan(1:3,i), rotM(model.MuMan(1:3,i))' * [model.MuMan(4:5,i); 0] + model.MuMan(1:3,i)];
	plot3(msh(1,:),msh(2,:),msh(3,:),'-','linewidth',2,'color',clrmap(i,:));
	plot3(msh(1,:),msh(2,:),msh(3,:),'.','markersize',24,'color',clrmap(i,:));
end

%Plot Bezier curve in tangent space of first control point
msh = rotM(model.MuMan(1:3,1))' * [U(1:2,:); zeros(1,nbData)] + repmat(model.MuMan(1:3,1),1,nbData);
plot3(msh(1,:),msh(2,:),msh(3,:),'-','linewidth',2,'color',[.7 .7 .7]);

%Plot Bezier control points in tangent space of first control point
Ac = transp(model.MuMan(1:3,2), x);
dxTmp = Ac * model.MuMan(4:5,2);
msh = rotM(model.MuMan(1:3,1))' * [logmap(model.MuMan(1:3,2),x); 0] + model.MuMan(1:3,1);
msh(:,2) = rotM(model.MuMan(1:3,1))' * [logmap(model.MuMan(1:3,2),x)+dxTmp; 0] + model.MuMan(1:3,1);
plot3(msh(1,:),msh(2,:),msh(3,:),'-','linewidth',2,'color',min(clrmap(2,:)+.2,1));
plot3(msh(1,:),msh(2,:),msh(3,:),'.','markersize',24,'color',min(clrmap(2,:)+.2,1));
	
%Plot Bezier curve on manifold
for n=1:nbRepros
	plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '-','linewidth',2,'color',[0 0 0]);
end
view(0,70); axis equal; axis vis3d;  

%print('-dpng','graphs/demo_Riemannian_S2_batchLQR_Bezier01.png');
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