function demo_Riemannian_Sd_MPC_infHor01
% Linear quadratic regulation on S^d by relying on Riemannian manifold and infinite-horizon LQR.
% (formulation with tangent space of the same dimension as the dimension of the manifold)
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
nbData = 200; %Number of datapoints
nbRepros = 1; %Number of reproductions

model.nbVarPos = 4; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector in the tangent space
model.params_diagRegFact = 1E-4; %Regularization of covariance
model.dt = 5E-3; %Time step duration
model.rfactor = 1E-2;	%Control cost in LQR 

%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;

%Target and desired covariance
xTar = (rand(model.nbVarPos,1)-0.5) * 2;
xTar = xTar / norm(xTar);
%xTar = [1; 0; 0; 0];

%[Ar,~] = qr(randn(model.nbVarPos));
% uCov = Ar * diag([1E0,1E0,1E6]) * Ar' * 1E-3;
% S0 = diag([1E0,1E0,1E6]) * 1E-3;
% uCov = quat2rotm(xTar') * S0 * quat2rotm(xTar')'
uCov = eye(model.nbVarPos) * 1E-2;


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
	x = [-1; -1; 1; 0] + randn(model.nbVarPos,1)*9E-1;
	x = x / norm(x);
	u = [-logmap(x,xTar); zeros(model.nbVarPos*(model.nbDeriv-1),1)];
	for t=1:nbData
		r(n).x(:,t) = x; %Log data
		%U = [-logmap(x,xTar); U(model.nbVarPos+1:end)];
		ddu = L * [logmap(x,xTar); dxTar - u(model.nbVarPos+1:end)]; %Compute acceleration (with only feedback terms)
		u = A * u +  B * ddu; %Update U
		x = expmap(-u(1:model.nbVarPos), xTar);
	end
end


% %% Discrete LQR with infinite horizon (computation centered on x)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dxTar = zeros(model.nbVarPos*(model.nbDeriv-1),1);
% for n=1:nbRepros
% 	x = [-1; -1; 1; 0] + randn(model.nbVarPos,1)*9E-1;
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
% 		x
% 		Ac = transp(xTar, x)
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
figure('PaperPosition',[0 0 6 8],'position',[10,10,650,650]); 
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

% %3D plot
% figure('PaperPosition',[0 0 6 8],'position',[670,10,650,650]); hold on; axis off; rotate3d on;
% colormap([.9 .9 .9]);
% [X,Y,Z] = sphere(20);
% mesh(X,Y,Z,'facealpha',.3,'edgealpha',.3);
% plot3Dframe(quat2rotm(xTar'), zeros(3,1), eye(3)*.3);
% h=[];
% for n=1:min(nbRepros,1)
% 	for t=floor(linspace(1,nbData,10))
% % 		delete(h);
% 		h = plot3Dframe(quat2rotm(r(n).x(:,t)'), zeros(3,1));
% % 		drawnow;
% 		if t==1
% 			%pause;
% 		end
% 	end
% end
% view(3); axis equal; axis tight; axis vis3d;  

%print('-dpng','graphs/demo_Riemannian_Sd_MPC_infHor01.png');
pause;
close all;
end

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u,x0)
	theta = sqrt(sum(u.^2,1)); %norm(u,'fro')
	x = real(repmat(x0,[1,size(u,2)]) .* repmat(cos(theta),[size(u,1),1]) + u .* repmat(sin(theta)./theta,[size(u,1),1]));
	x(:,theta<1e-16) = repmat(x0,[1,sum(theta<1e-16)]);	
end

function u = logmap(x,x0)
	theta = acos(x0'*x); %acos(trace(x0'*x))
	u = (x - repmat(x0,[1,size(x,2)]) .* repmat(cos(theta),[size(x,1),1])) .* repmat(theta./sin(theta),[size(x,1),1]);
	u(:,theta<1e-16) = 0;
end

% function Ac = transp(x1,x2,t)
% 	if nargin==2
% 		t=1;
% 	end
% 	xdir = logmap(x2,x1);
% 	nrxdir = xdir./norm(xdir,'fro');
% 	Ac = -x1 * sin(norm(xdir,'fro')*t)*nrxdir' + nrxdir*cos(norm(xdir,'fro')*t)*nrxdir' + eye(size(xdir,1))-nrxdir*nrxdir';
% end