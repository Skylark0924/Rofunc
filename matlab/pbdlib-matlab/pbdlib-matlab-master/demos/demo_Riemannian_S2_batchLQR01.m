function demo_Riemannian_S2_batchLQR01
% LQT on a sphere (version with viapoints) by relying on Riemannian manifold (recomputed in an online manner), 
% by using position and velocity data (-> acceleration commands)
% (see also demo_Riemannian_Sd_MPC01.m)
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
nbD = 20; %Time window for LQR computation
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 6; %Number of states in the GMM
model.nbVarPos = 2; %Dimension of data in tangent space (here: v1,v2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.nbVarMan = 2*model.nbVarPos+1; %Dimension of the manifold (here: x1,x2,x3,v1,v2)
model.dt = 1E-2; %Time step duration
model.params_diagRegFact = 1E-4; %Regularization of covariance
model.rfactor = 1E-10; %Control cost in LQR 
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
u0 = [];
for n=1:nbSamples
 	s(n).u0 = []; 
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)) .* 9E-2; %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives expressed in single tangent space e0
		end
		s(n).u0 = [s(n).u0; dTmp];
	end
	u0 = [u0, s(n).u0]; 
end
x0 = [expmap(u0(1:model.nbVarPos,:), e0); u0(model.nbVarPos+1:end,:)]; %x0 is on the manifold and the derivatives are expressed in the tangent space of e0
	

%% GMM parameters estimation (encoding of [x;dx] with x on sphere and dx in Euclidean space)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u0, model, nbSamples);
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
			
			dTmp = x0(4:5,:); %Derivatives are expressed in the tangent space of e0
			
			%Transportation of derivatives from e0 to xTar (MuMan)
			Ac = transp(e0, xTar);
			dTmp = Ac * dTmp; %Derivatives are now expressed in xTar (MuMan)
			
			u(:,:,i) = [uTmp; dTmp];
			model.MuMan(:,i) = [expmap(uTmp*H(i,:)', xTar); dTmp*H(i,:)']; 
		end
		%Update Sigma
		model.Sigma(:,:,i) = u(:,:,i) * diag(H(i,:)) * u(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
% 		model.Sigma(:,:,i) = blkdiag(eye(model.nbVarPos)*1E-1, eye(model.nbVarPos)*1E-5);
	end
end

%Precomputation of inverses
for i=1:model.nbStates
	model.Q(:,:,i) = inv(model.Sigma(:,:,i));
end

%List of states with time stamps
[~,q] = max(H(:,1:nbData),[],1); 
qList = [];
qCurr = q(1);
t_old = 1; 
for t=1:nbData
	if qCurr~=q(t) || t==nbData
		tm = t_old + floor((t-t_old)/2);
		qList = [qList, [tm; qCurr]];	
		qCurr = q(t);
		t_old = t;
	end
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

		%Version with viapoints
		id = t:t+nbD-1; %Time window
		qid = qList(1,:) > id(1) & qList(1,:) < id(end); 
		qTmp = qList(:,qid); %List only the states appearing within the time window
		MuQ = zeros(model.nbVar*nbD,1);
		Q = zeros(model.nbVar*nbD);
		for i=1:size(qTmp,2)
			s = qTmp(1,i) - t; %Time step
			id = (s-1)*model.nbVar+1:s*model.nbVar;
			
			%Transportation of Q from MuMan to x
			Ac = transp(model.MuMan(1:3,qTmp(2,i)), x);
			Q(id,id) = blkdiag(Ac,Ac) * model.Q(:,:,qTmp(2,i)) * blkdiag(Ac,Ac)';
% 			Q(id,id) = blkdiag(Ac,eye(model.nbVarPos)) * model.Q(:,:,qTmp(2,i)) * blkdiag(Ac,eye(model.nbVarPos))';

			%Transportation of du from MuMan to x
			dxTmp = Ac * model.MuMan(4:5,qTmp(2,i));
		
% 			%Transportation of du from e0 to x
% 			Ac = transp(e0, x);
% 			dxTmp = Ac * model.MuMan(4:5,qTmp(2,i));
			
			MuQ(id) = [logmap(model.MuMan(1:3,qTmp(2,i)), x); dxTmp];
% 			MuQ(id) = [logmap(model.MuMan(1:3,qTmp(2,i)), x); model.MuMan(4:5,qTmp(2,i))]; 

			%Log data (for plots)
			r(n).s(t).Mu(:,i) = logmap(model.MuMan(1:3,qTmp(2,i)), x);
			r(n).s(t).Sigma(:,:,i) = Ac * model.Sigma(1:model.nbVarPos,1:model.nbVarPos,qTmp(2,i)) * Ac';
		end
		
		%Compute acceleration commands
		ddu = (Su' * Q * Su + R) \ Su' * Q * (MuQ - Sx * U);
		
		%Log data (for plots)
		r(n).s(t).u = reshape(Sx*U+Su*ddu, model.nbVar, nbD);
		r(n).s(t).q = qTmp;
		
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
% Gvel = zeros(3, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(1:model.nbVarPos,1:model.nbVarPos,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(1:3,i));
% 	Gvel(:,i) = expmap(model.MuMan(4:5,i)*model.dt*5, model.MuMan(1:3,i));
end

%S2 manifold plot
figure('position',[10,10,1600,500]); 
subplot(2,3,[1,4]); hold on; axis off; grid off; rotate3d on; 
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
set(gca,'LooseInset',[0,0,0,0]);
colormap([.8 .8 .8]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);
% plot3(x0(1,:), x0(2,:), x0(3,:), '.','markersize',10,'color',[.5 .5 .5]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',24,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
% 	plot3(Gvel(1,i), Gvel(2,i), Gvel(3,i), 'o','markersize',6,'linewidth',2,'color',clrmap(i,:));
end
for n=1:nbRepros
	plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '-','linewidth',2,'color',[.6 .6 .6]);
	plot3(r(n).x(1,1), r(n).x(2,1), r(n).x(3,1), '.','markersize',24,'color',[.6 .6 .6]);
end
view(-20,15); axis equal; axis([-1 1 -1 1 -1 1].*1.4); axis vis3d;

%Tangent plane plot
subplot(2,3,[2,5]); hold on; axis off; 
plot(0,0,'+','markersize',40,'linewidth',2,'color',[.3 .3 .3]);
plot(0,0,'.','markersize',20,'color',[0 0 0]);
axis equal; axis([-1 1 -1 1].*1.1);
	
%Timeline plot
labList = {'$x_1$','$x_2$','$\dot{x}_1$','$\dot{x}_2$','$\ddot{x}_1$','$\ddot{x}_2$'}; 
for j=1:2
	v(j).limAxes = [1, nbData, min(r(1).x(j,:))-3E-1, max(r(1).x(j,:))+3E-1];
	subplot(2,3,j*3); hold on;
	%Plot viapoints reference
	for i=1:size(qList,2)
		errorbar(qList(1,i), model.MuMan(j,qList(2,i)), model.Sigma(j,j,qList(2,i)).^.5, 'linewidth',2,'color',clrmap(qList(2,i),:));
		plot(qList(1,i), model.MuMan(j,qList(2,i)), '.','markersize',20,'color',clrmap(qList(2,i),:));
	end
	for n=1:nbRepros
		plot(r(n).x(j,:), '-','linewidth',2,'color',[.6 .6 .6]);
	end
	if j<7
		ylabel(labList{j},'fontsize',24,'interpreter','latex');
	end
	axis(v(j).limAxes);
	set(gca,'xtick',[],'ytick',[],'linewidth',2);
	xlabel('$t$','fontsize',24,'interpreter','latex');
end


%% Anim plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for tt=floor(linspace(1,nbData,nbData)) %58:58
	e0 = r(n).x(:,tt);
	xTmp = expmap(r(1).s(tt).u(1:model.nbVarPos,:), e0);
	%Tangent plane on S2 anim
	subplot(2,3,[1,4]); hold on;
	msh = repmat(e0,1,5) + rotM(e0)' * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 5E-1;
% 	h = patch(msh(1,:),msh(2,:),msh(3,:), [.8 .8 .8],'edgecolor',[.6 .6 .6]),'facealpha',.3,'edgealpha',.3);
	h = plot3(msh(1,:),msh(2,:),msh(3,:), '-','color',[.6 .6 .6]);
	h = [h, plot3(e0(1), e0(2), e0(3), '.','markersize',20,'color',[0 0 0])];
	%Referential on S2 anim
	msh = repmat(e0,1,2) + rotM(e0)' * [1 -1; 0 0; 0 0] * 2E-1;
	h = [h, plot3(msh(1,:),msh(2,:),msh(3,:), '-','linewidth',2,'color',[.3 .3 .3])];
	msh = repmat(e0,1,2) + rotM(e0)' * [0 0; 1 -1; 0 0] * 2E-1;
	h = [h, plot3(msh(1,:),msh(2,:),msh(3,:), '-','linewidth',2,'color',[.3 .3 .3])];
	%Path on S2 anim
	h = [h, plot3(xTmp(1,:),xTmp(2,:),xTmp(3,:), '-','linewidth',2,'color',[0 0 0])];
	
	%Tangent plane anim
	subplot(2,3,[2,5]); hold on; axis off; 
	for i=1:size(r(1).s(tt).Mu,2)
		h = [h, plotGMM(r(1).s(tt).Mu(:,i), r(1).s(tt).Sigma(:,:,i), clrmap(r(1).s(tt).q(2,i),:), .3)];
	end
	h = [h, plot(r(1).s(tt).u(1,:), r(1).s(tt).u(2,:), '-','linewidth',2,'color',[0 0 0])];
	
	%Time window anim
	for j=1:2
		subplot(2,3,j*3); hold on;
		msh = [tt tt+nbD-1 tt+nbD-1 tt tt; v(j).limAxes([3,3]) v(j).limAxes([4,4]) v(j).limAxes(3)];
		h = [h, plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.6 .6 .6])];
		h = [h, plot(tt:tt+nbD-1, xTmp(j,:), '-','linewidth',2,'color',[0 0 0])];
		h = [h, plot(tt, xTmp(j,1), '.','markersize',20,'color',[0 0 0])];
	end

	drawnow;
% 	print('-dpng',['graphs/anim/S2_batchLQR_anim' num2str(tt,'%.3d') '.png']);

	delete(h);
end

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