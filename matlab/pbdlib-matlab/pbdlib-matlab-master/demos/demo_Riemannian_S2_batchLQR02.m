function demo_Riemannian_S2_batchLQR02
% LQT on a sphere by relying on Riemannian manifold (recomputed in an online manner),
% by using position and velocity data (-> acceleration commands)
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
nbRepros = 5; %Number of reproductions
nbIter = 20; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 20; %Number of iteration for the EM algorithm
nbData = 100; %Number of datapoints
nbD = 40; %Time window for LQR computation
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 6; %Number of states in the GMM
model.nbVarPos = 2; %Dimension of data in tangent space (here: v1,v2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.nbVarMan = 2*model.nbVarPos+1; %Dimension of the manifold (here: x1,x2,x3,v1,v2)
model.dt = 1E-5; %Time step duration
model.params_diagRegFact = 1E-4; %Regularization of covariance
model.rfactor = 1E-15;	%Control cost in LQR 
e0 = [0; -1; 0]; %Point on the sphere to project handwriting data

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
	%s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)) * 1.3E-1;
	u = [u, s(n).Data]; 
end
x0 = [expmap(u(1:model.nbVarPos,:), e0); u(model.nbVarPos+1:end,:)]; %x is on the manifold and dx is in the tangent space of e0


%% GMM parameters estimation (encoding of [x;dx] with x on sphere and dx in Euclidean space)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [expmap(model.Mu(1:model.nbVarPos,:), e0); model.Mu(model.nbVarPos+1:end,:)]; %Center on the manifold
model.Mu = zeros(model.nbVarPos,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
u = [];
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
			xTar = model.MuMan(1:3,i); %Position target
			uTmp = logmap(x0(1:3,:), xTar);
			
% 			dTmp = gradient(uTmp) / model.dt;
			dTmp = x0(4:5,:); %dx in the e0 tangent space
			
			%Transportation of dTmp from e0 to xTar
			Ac = transp(e0, xTar);
			dTmp = Ac * dTmp;
			
			u(:,:,i) = [uTmp; dTmp];
			model.MuMan(:,i) = [expmap(uTmp*H(i,:)', xTar); dTmp*H(i,:)'];

% 			%Debug
% 			dTmp = x0(4:5,2);
% 			%Transportation of dTmp from e0 to xTar
% 			Ac = transp(e0, xTar);
% 			dTmp = Ac * dTmp;
% 			model.MuMan(:,i) = [expmap(uTmp*H(i,:)', xTar); dTmp];

		end
		%Update Sigma
		model.Sigma(:,:,i) = u(:,:,i) * diag(H(i,:)) * u(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
% 		model.Sigma(:,:,i) = blkdiag(eye(model.nbVarPos)*1E-1, eye(model.nbVarPos)*1E-5);
	end
end

%Precomputation of inverses
for i=1:model.nbStates
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end


%% Batch LQR recomputed in an online manner (computation centered on x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbRepros
	x = x0(1:3,1) + randn(3,1)*3E-1;
	x = x / norm(x);
	x_old = x;
	U = zeros(model.nbVar,1);
	for t=1:nbData
		r(n).x(:,t) = x; %Log data
		U(1:model.nbVarPos) = zeros(model.nbVarPos,1); %Set tangent space at x
		
		%Transportation of velocity vectors from x_old to x
		Ac = transp(x_old, x);
		U(model.nbVarPos+1:end) = Ac * U(model.nbVarPos+1:end); 

		%Set list of states for the next nbD time steps according to first demonstration (alternatively, an HSMM can be used)
		id = [t:min(t+nbD-1,nbData), repmat(nbData,1,t-nbData+nbD-1)];
		[~,q] = max(H(:,id),[],1); %works also for nbStates=1
		
		%Create single Gaussian N(MuQ,Q^-1) based on optimal state sequence q
		MuQ = zeros(model.nbVar*nbD,1);
		Q = zeros(model.nbVar*nbD);
		for s=1:nbD
			id = (s-1)*model.nbVar+1:s*model.nbVar;
			%Transportation of Sigma and duCov from model.MuMan to x
			Ac = transp(model.MuMan(1:3,q(s)), x);
			Q(id,id) = blkdiag(Ac,Ac) * model.invSigma(:,:,q(s)) * blkdiag(Ac,Ac)';
% 			Q(id,id) = blkdiag(Ac,eye(model.nbVarPos)) * model.invSigma(:,:,q(s)) * blkdiag(Ac,eye(model.nbVarPos))';
			
			%Transportation of du from model.MuMan to x
			dxTmp = Ac * model.MuMan(4:5,q(s));
			MuQ(id) = [logmap(model.MuMan(1:3,q(s)), x); dxTmp];
% 			MuQ(id) = [logmap(model.MuMan(1:3,q(s)), x); model.MuMan(4:5,q(s))]; 
		end
		
		%Compute acceleration commands
		SuInvSigmaQ = Su' * Q;
		Rq = SuInvSigmaQ * Su + R;
		rq = SuInvSigmaQ * (MuQ-Sx*U);
		ddu = Rq \ rq;
		
		U = A * U + B * ddu(1:model.nbVarPos); %Update U with first control command
		x_old = x; %Keep x for next iteration
		x = expmap(U(1:model.nbVarPos), x); %Update x
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(3, nbDrawingSeg, model.nbStates);
Gvel = zeros(3, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(1:3,i));
	Gvel(:,i) = expmap(model.MuMan(4:5,i)*model.dt*5, model.MuMan(1:3,i));
end

%Plots
figure('position',[10,10,800,800]); hold on; axis off; grid off; rotate3d on; 
colormap([.8 .8 .8]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);
plot3(x0(1,:), x0(2,:), x0(3,:), '.','markersize',10,'color',[.5 .5 .5]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',24,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
	plot3(Gvel(1,i), Gvel(2,i), Gvel(3,i), 'o','markersize',6,'linewidth',2,'color',clrmap(i,:));
end
for n=1:nbRepros
	h(1) = plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '-','linewidth',2,'color',[0 0 0]);
	plot3(r(n).x(1,1), r(n).x(2,1), r(n).x(3,1), '.','markersize',24,'color',[0 0 0]);
end
view(-20,15); axis equal; axis vis3d;  

% print('-dpng','graphs/demo_Riemannian_S2_batchLQR01.png');
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