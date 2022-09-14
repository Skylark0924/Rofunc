function demo_Riemannian_S3_infHorLQR01
% Linear quadratic regulation of unit quaternions (orientation) by relying on Riemannian manifold and infinite-horizon LQR
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
nbData = 120; %Number of datapoints
nbRepros = 1; %Number of reproductions

model.nbVarPos = 3; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.nbVarMan = model.nbVarPos+1; %Dimension of the manifold
model.params_diagRegFact = 1E-4; %Regularization of covariance
model.dt = 5E-3; %Time step duration
model.rfactor = 1E-2;	%Control cost in LQR 

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;

%Target and desired covariance
xTar = (rand(model.nbVarMan,1)-0.5) * 2;
xTar = xTar / norm(xTar);
%xTar = [1; 0; 0; 0];

%[Ar,~] = qr(randn(model.nbVarPos));
% uCov = Ar * diag([1E0,1E0,1E6]) * Ar' * 1E-3;

% S0 = diag([1E0,1E0,1E6]) * 1E-3;
% uCov = quat2rotm(xTar') * S0 * quat2rotm(xTar')'
uCov = diag([1,1,1]) * 1E-3;

% %Eigendecomposition 
% [V,D] = eig(S0);
% U0 = V * D.^.5;

% Ac = transp([1;0;0;0], xTar);
% Ac2 = transp(xTar, [1;0;0;0]);
% U = Ac * U0; %Transport of velocity
% U2 = Ac2 * U0; %Transport of velocity

% %Eigendecomposition 
% [V,D] = eig(uCov);
% U0 = V * D.^.5;


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


%% Discrete LQR with infinite horizon (computation centered on xTar)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dxTar = zeros(model.nbVarPos*(model.nbDeriv-1),1);
Q = blkdiag(inv(uCov), zeros(model.nbVarPos*(model.nbDeriv-1))); %Precision matrix
P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
L = (B'*P*B + R) \ B'*P*A; %Feedback gain (discrete version)
for n=1:nbRepros
	x = [-1; -1; 1; 0] + randn(model.nbVarMan,1)*9E-1;
	x = x / norm(x);
	U = [-logmap(x,xTar); zeros(model.nbVarPos*(model.nbDeriv-1),1)];
	for t=1:nbData
		r(n).x(:,t) = x; %Log data
		%U = [-logmap(x,xTar); U(model.nbVarPos+1:end)];
		ddu = L * [logmap(x,xTar); dxTar-U(model.nbVarPos+1:end)]; %Compute acceleration (with only feedback terms)
		U = A*U + B*ddu; %Update U
		x = expmap(-U(1:model.nbVarPos), xTar);
	end
end


% %% Discrete LQR with infinite horizon (computation centered on x)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dxTar = zeros(model.nbVarPos*(model.nbDeriv-1),1);
% for n=1:nbRepros
% 	x = [-1; -1; 1; 0] + randn(model.nbVarMan,1)*9E-1;
% 	%x = xTar + randn(model.nbVarMan,1)*1E-1;
% 	x = x / norm(x);
% 	x_old = x;
% 	U = zeros(model.nbVar,1);
% 	for t=1:nbData
% 		r(n).x(:,t) = x; %Log data
% 		U(1:model.nbVarPos) = zeros(model.nbVarPos,1); %Set tangent space at x
% 		
% 		%Transportation of velocity vector from x_old to x
% 		Ac = transp(x_old, x);
% 		U(model.nbVarPos+1:end) = Ac * U(model.nbVarPos+1:end); %Transport of velocity
% 		
% 		%Transportation of uCov from xTar to x
% 		Ac = transp(xTar, x);
% 		uCovTmp = Ac * uCov * Ac';
% 		Q = blkdiag(inv(uCovTmp), zeros(model.nbVarPos*(model.nbDeriv-1))); %Precision matrix %+ eye(model.nbVar) * 1E-8
% 		P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), (Q+Q')/2);
% 		L = (B'*P*B + R) \ B'*P*A; %Feedback gain (discrete version)		
% 		
% 		ddu = L * [logmap(xTar,x); dxTar-U(model.nbVarPos+1:end)]; %Compute acceleration 
% 		U = A*U + B*ddu; %Update U
% 		x_old = x;
% 		x = expmap(U(1:model.nbVarPos), x);
% 	end
% end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Timeline plot
figure('PaperPosition',[0 0 6 8],'position',[10,10,650,650],'name','timeline data'); 
for k=1:4
	subplot(2,2,k); hold on; 
	for n=1:nbRepros
		plot(1:nbData, r(n).x(k,:), '-','linewidth',1,'color',[0 0 0]);
	end
	plot([1 nbData], [xTar(k) xTar(k)], '-','linewidth',1,'color',[.8 0 0]);
	plot([1 nbData], [-xTar(k) -xTar(k)], '--','linewidth',1,'color',[.8 0 0]);
	xlabel('t'); ylabel(['q_' num2str(k)]);
	axis([1 nbData -1 1]);
end

%3D plot
figure('PaperPosition',[0 0 6 8],'position',[670,10,650,650],'name','timeline data'); hold on; axis off; rotate3d on;
colormap([.9 .9 .9]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
Rq = quat2rotm(xTar');

plot3Dframe(Rq, zeros(3,1), eye(3)*.3);
view(3); axis equal; axis tight; axis vis3d;  
h=[];
for n=1:min(nbRepros,1)
	for t=1:nbData
		delete(h);
		Rq = quat2rotm(r(n).x(:,t)');
		h = plot3Dframe(Rq, zeros(3,1));
		drawnow;
		if t==1
			%pause;
		end
	end
end
%print('-dpng','graphs/demo_Riemannian_S3_infHorLQR01.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, mu)
	x = QuatMatrix(mu) * expfct(u);
end

function u = logmap(x, mu)
	if norm(mu-[1;0;0;0])<1e-6
		Q = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
	else
		Q = QuatMatrix(mu);
	end
	u = logfct(Q'*x);
end

function Exp = expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2+u(3,:).^2);
	Exp = real([cos(normv) ; u(1,:).*sin(normv)./normv ; u(2,:).*sin(normv)./normv ; u(3,:).*sin(normv)./normv]);
	Exp(:,normv < 1e-16) = repmat([1;0;0;0],1,sum(normv < 1e-16));
end

function Log = logfct(x)
% 	scale = acos(x(3,:)) ./ sqrt(1-x(3,:).^2);
	scale = acoslog(x(1,:)) ./ sqrt(1-x(1,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(2,:).*scale; x(3,:).*scale; x(4,:).*scale];
end

function Q = QuatMatrix(q)
	Q = [q(1) -q(2) -q(3) -q(4);
			 q(2)  q(1) -q(4)  q(3);
			 q(3)  q(4)  q(1) -q(2);
			 q(4) -q(3)  q(2)  q(1)];
end				 

% Arcosine re-defitinion to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
function acosx = acoslog(x)
	for n=1:size(x,2)
		% sometimes abs(x) is not exactly 1.0
		if(x(n)>=1.0)
			x(n) = 1.0;
		end
		if(x(n)<=-1.0)
			x(n) = -1.0;
		end
		if(x(n)>=-1.0 && x(n)<0)
			acosx(n) = acos(x(n))-pi;
		else
			acosx(n) = acos(x(n));
		end
	end
end

function Ac = transp(g,h)
	E = [zeros(1,3); eye(3)];
	vm = QuatMatrix(g) * [0; logmap(h,g)];
	mn = norm(vm);
    if mn < 1e-10
        disp('Angle of rotation too small (<1e-10)');
        Ac = eye(3);
        return;
    end
	uv = vm / mn;
	Rpar = eye(4) - sin(mn)*(g*uv') - (1-cos(mn))*(uv*uv');	
	Ac = E' * QuatMatrix(h)' * Rpar * QuatMatrix(g) * E; %Transportation operator from g to h 
end