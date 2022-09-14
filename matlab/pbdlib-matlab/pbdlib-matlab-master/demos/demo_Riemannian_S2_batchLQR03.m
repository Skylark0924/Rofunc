function demo_Riemannian_S2_batchLQR03
% LQT on a sphere by relying on Riemannian manifold (recomputed in an online manner), 
% by using only position data (-> velocity commands)
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
nbSamples = 5; %Number of demonstrations
nbRepros = 1; %Number of reproductions
nbIter = 20; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 20; %Number of iteration for the EM algorithm
nbData = 100; %Number of datapoints
nbD = 40; %Time window for LQR computation
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 6; %Number of states in the GMM
model.nbVarPos = 2; %Dimension of data in tangent space (here: v1,v2)
model.nbDeriv = 1; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.nbVarMan = 2*model.nbVarPos+1; %Dimension of the manifold (here: x1,x2,x3,v1,v2)
model.dt = 1E-2; %Time step duration
model.params_diagRegFact = 1E-4; %Regularization of covariance
model.rfactor = 5E-1;	%Control cost in LQR 
e0 = [0; -1; 0]; %Origin on manifold

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbD-1),R);


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
Su = zeros(model.nbVar*nbD, model.nbVarPos*(nbD-1));
Sx = kron(ones(nbD,1),eye(model.nbVar));
M = B;
for n=2:nbD
	id1 = (n-1)*model.nbVar+1:nbD*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Generate data on a sphere from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/S.mat');
u = [];
for n=1:nbSamples
 	s(n).Data = []; %Resampling
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)) * .9E-1; %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).Data = [s(n).Data; dTmp];
	end
	%s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)) * .9E-1; %Resampling 
	u = [u, s(n).Data]; 
end
x0 = expmap(u(1:model.nbVarPos,:), e0); %x is on the manifold 


%% GMM parameters estimation (encoding of x on sphere)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu(1:model.nbVarPos,:), e0); %Center on the manifold
model.Mu = zeros(model.nbVarPos,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x0,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x0(1:3,:), model.MuMan(1:3,i)), model.Mu(1:model.nbVarPos,i), model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp = logmap(x0(1:3,:), model.MuMan(1:3,i));		
			model.MuMan(:,i) = expmap(uTmp*H(i,:)', model.MuMan(1:3,i));
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp * diag(H(i,:)) * uTmp' + eye(size(uTmp,1)) * model.params_diagRegFact;
	end
end

%Precomputation of inverses
for i=1:model.nbStates
	model.Q(:,:,i) = inv(model.Sigma(:,:,i));
end


%% Batch LQR recomputed in an online manner (computation centered on x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbRepros
	x = x0(1:3,1) + randn(3,1)*2E-1;
	x = x / norm(x);
	for t=1:nbData
		%Log data
		r(n).x(:,t) = x; 
		
		%Version with stepwise tracking
		%Set list of states for the next nbD time steps according to first demonstration (alternatively, an HSMM can be used)
		id = [t:min(t+nbD-1,nbData), repmat(nbData,1,t-nbData+nbD-1)];
		[~,q] = max(H(:,id),[],1); %works also for nbStates=1
		%Create single Gaussian N(MuQ,Q^-1) based on optimal state sequence q
		MuQ = zeros(model.nbVar*nbD,1);
		Q = zeros(model.nbVar*nbD);
		for s=1:nbD
			id = (s-1)*model.nbVar+1:s*model.nbVar;
			MuQ(id) = logmap(model.MuMan(:,q(s)), x);
			%Transportation of Sigma from model.MuMan to x
			Ac = transp(model.MuMan(:,q(s)), x);
			Q(id,id) = Ac * model.Q(:,:,q(s)) * Ac';
		end
		
		%Compute velocity commands
		ddu = (Su' * Q * Su + R) \ Su' * Q * MuQ; %(Sx*U=0 since we compute the command in the tangent space at x, with U=0)
		U = B * ddu(1:model.nbVarPos); %Update U with first control command (A*U=0 since we compute the command in the tangent space at x, with U=0)
		%Log data (for plots)
		r(n).s(t).u = reshape(Su*ddu, model.nbVar, nbD);
		%Update x
		x = expmap(U, x); 
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(3, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(1:3,i));
end

%Manifold plot
figure('position',[10,10,800,800]); hold on; axis off; grid off; rotate3d on; 
colormap([.8 .8 .8]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z); %,'facealpha',.5
plot3(x0(1,:), x0(2,:), x0(3,:), '.','markersize',10,'color',[.5 .5 .5]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',24,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
end
for n=1:nbRepros
	plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '-','linewidth',2,'color',[0 0 0]);
	plot3(r(n).x(1,1), r(n).x(2,1), r(n).x(3,1), '.','markersize',24,'color',[0 0 0]);
end
view(-20,15); axis equal; axis vis3d;  

%Draw tangent plane
tt = 1;
e0 = r(n).x(:,tt);
msh = repmat(e0,1,5) + rotM(e0)' * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
patch(msh(1,:),msh(2,:),msh(3,:), [.8 .8 .8],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);

% %Draw referential
% msh = repmat(e0,1,2) + rotM(e0)' * [1 -1; 0 0; 0 0] * 2E-1;
% plot3(msh(1,:),msh(2,:),msh(3,:), '-','linewidth',2,'color',[.6 .6 .6]);
% msh = repmat(e0,1,2) + rotM(e0)' * [0 0; 1 -1; 0 0] * 2E-1;
% plot3(msh(1,:),msh(2,:),msh(3,:), '-','linewidth',2,'color',[.6 .6 .6]);
% plot3(0,0,1, '.','markersize',20,'color',[0 0 0]);


%% Tangent space plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[820,10,650,650]); hold on; axis off; box on;
plot(0,0,'+','markersize',40,'linewidth',2,'color',[.7 .7 .7]);
plot(0,0,'.','markersize',20,'color',[0 0 0]);
% % plot(u0(1,:), u0(2,:), '.','markersize',12,'color',[.5 .5 .5]);
% for t=1:nbData*nbSamples
% 	plot(u0(1,t), u0(2,t), '.','markersize',12,'color',GAMMA(:,t)'*clrmap);
% end
% for i=1:model.nbStates
% 	plotGMM(model.Mu(:,i), model.Sigma(:,:,i)*3, clrmap(i,:), .3);
% end
plot(r(1).s(tt).u(1,:), r(1).s(tt).u(2,:), '-','color',[0 0 0]);
axis equal;

% print('-dpng','graphs/demo_Riemannian_S2_batchLQR02.png');
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