function demo_Riemannian_S2_infHorLQR01
% Infinite-horizon linear quadratic regulation on a sphere by relying on Riemannian manifold.
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
nbDrawingSeg = 50; %Number of segments used to draw ellipsoids
nbRepros = 20; %Number of reproductions

nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
nbVar = nbVarPos * nbDeriv; %Dimension of state vector in the tangent space
nbVarMan = nbVarPos+1; %Dimension of the manifold
dt = 1E-3; %Time step duration
rfactor = 1E-6;	%Control cost in LQR 

%Control cost matrix
R = eye(nbVarPos) * rfactor;

%Target and desired covariance
xTar = [-1; .2; .6];
xTar = xTar / norm(xTar);

[Ar,~] = qr(randn(nbVarPos));
uCov = Ar*diag([.1,1.4])*Ar' * 1E-1;
%uCov = diag([.1,.1]) * 1E-1;

% %Eigendecomposition 
% [V,D] = eig(uCov);
% U0 = V * D.^.5;


%% Discrete dynamical System settings (in tangent space)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(nbDeriv);
for i=0:nbDeriv-1
	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(nbDeriv,1); 
for i=1:nbDeriv
	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(nbVarPos)); %Discrete nD
B = kron(B1d, eye(nbVarPos)); %Discrete nD

%Generate initial conditions
for n=1:nbRepros
	x0(:,n) = [-1; -1; 0] + randn(nbVarMan,1)*9E-1;
	x0(:,n) = x0(:,n) / norm(x0(:,n));
end	
	

%% Discrete LQR with infinite horizon (computation centered on xTar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
duTar = zeros(nbVarPos*(nbDeriv-1),1);
Q = blkdiag(inv(uCov), zeros(nbVarPos*(nbDeriv-1))); %Precision matrix
P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
L = (B'*P*B + R) \ B'*P*A; %Feedback gain (discrete version)
for n=1:nbRepros
	x = x0(:,n);
	U = [logmap(x,xTar); zeros(nbVarPos*(nbDeriv-1),1)];
	for t=1:nbData
		r(n).x(:,t) = x; %Log data
		%U = [-logmap(x,xTar); U(nbVarPos+1:end)];
		ddu = L * [-logmap(x,xTar); duTar-U(nbVarPos+1:end)]; %Compute acceleration (with only feedback terms)
		U = A*U + B*ddu; %Update U
		x = expmap(U(1:nbVarPos), xTar);
	end
end


%% Discrete LQR with infinite horizon (computation centered on x, for comparison)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
duTar = zeros(nbVarPos*(nbDeriv-1),1);
for n=1:nbRepros
	x = x0(:,n);
	x_old = x;
	U = zeros(nbVar,1);
	for t=1:nbData
		r2(n).x(:,t) = x; %Log data
		U(1:nbVarPos) = zeros(nbVarPos,1); %Set tangent space at x
		
		%Transportation of velocity vector from x_old to x
		Ac = transp(x_old, x);
		U(nbVarPos+1:end) = Ac * U(nbVarPos+1:end); %Transport of velocity
		
		%Transportation of uCov from xTar to x
		Ac = transp(xTar, x);
		uCovTmp = Ac * uCov * Ac';
				
		Q = blkdiag(inv(uCovTmp), zeros(nbVarPos*(nbDeriv-1))); %Precision matrix
		P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
		L = (B'*P*B + R) \ B'*P*A; %Feedback gain (discrete version)
		
		ddu = L * [logmap(xTar,x); duTar-U(nbVarPos+1:end)]; %Compute acceleration 
		%ddu = L * [logmap(x,xTar); duTar-U(nbVarPos+1:end)]; %Compute acceleration 
		U = A*U + B*ddu; %Update U
		x_old = x;
		x = expmap(U(1:nbVarPos), x);
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
[V,D] = eig(uCov(1:2,1:2));
Gdisp = expmap(V*D.^.5*[cos(tl); sin(tl)], xTar);

%Plots
figure('position',[10,10,1300,650]); hold on; axis off; grid off; rotate3d on; 
colormap(repmat(linspace(.95,.2,64),3,1)');
nbp = 40;
[X,Y,Z] = sphere(nbp-1);
p = [reshape(X,1,nbp^2); reshape(Y,1,nbp^2); reshape(Z,1,nbp^2)];
dtmp = logmap(p,xTar)';
c = sum((dtmp/uCov(1:2,1:2)).*dtmp, 2);
surf(X,Y,Z,reshape(c,nbp,nbp),'linestyle','none');
plot3(xTar(1), xTar(2), xTar(3), '.','markersize',12,'color',[.8 0 0]);
plot3(Gdisp(1,:), Gdisp(2,:), Gdisp(3,:), '-','linewidth',3,'color',[.8 0 0]);
for n=1:nbRepros
	h(1) = plot3(r(n).x(1,:), r(n).x(2,:), r(n).x(3,:), '-','linewidth',1,'color',[0 0 0]);
	h(2) = plot3(r2(n).x(1,:), r2(n).x(2,:), r2(n).x(3,:), '-','linewidth',1,'color',[.7 .7 .7]);
end
%legend(h,'Centered on xhat','Centered on x');
view(-75,8); axis equal; axis vis3d;
%print('-dpng','-r300','graphs/demo_Riemannian_S2_infHorLQR01.png');

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