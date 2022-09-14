function demo_OC_LQT_constrained01
% Constrained batch LQT by using quadratic programming solver, with an encoding of position and velocity data.
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
% }
% 
% Copyright (c) 2016 Idiap Research Institute, http://idiap.ch/
% Written by Martijn Zeestraten and Sylvain Calinon
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbSamples = 5; %Number of demonstrations
nbRepros = 5; %Number of reproductions in new situations
nbData = 100; %Number of datapoints

model.nbStates = 5; %Number of Gaussians in the GMM
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.rfactor = 1E-6;	%Control cost in LQR
model.dt = 0.01; %Time step duration

%Dynamical System settings (discrete version), see Eq. (33)
A = kron([1, model.dt; 0, 1], eye(model.nbVarPos));
B = kron([0; model.dt], eye(model.nbVarPos));
%C = kron([1, 0], eye(model.nbVarPos));
%Control cost matrix
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData-1),R);

%Build Sx and Su matrices for batch LQR, see Eq. (35)
Su = zeros(model.nbVar*nbData, model.nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(model.nbVar)); 
M = B;
for n=2:nbData
	id1 = (n-1)*model.nbVar+1:nbData*model.nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*model.nbVar+1:n*model.nbVar; 
	id2 = 1:(n-1)*model.nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:model.nbVarPos), M];
end

%% Define Convex State Constraints Ax*x < bx
Ax = -[1,1,0,0];
bx = 8;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data=[];
	for m=1:model.nbDeriv
		if m==1
			dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
		else
			dTmp = gradient(dTmp) / model.dt; %Compute derivatives
		end
		s(n).Data = [s(n).Data; dTmp];
	end
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation...');
%model = init_GMM_kmeans(Data,model);
model = init_GMM_kbins(Data, model, nbSamples);

% %Initialization based on position data
% model0 = init_GMM_kmeans(Data(1:model.nbVarPos,:), model);
% [~, GAMMA2] = EM_GMM(Data(1:model.nbVarPos,:), model0);
% model.Priors = model0.Priors;
% for i=1:model.nbStates
% 	model.Mu(:,i) = Data * GAMMA2(i,:)';
% 	DataTmp = Data - repmat(model.Mu(:,i),1,nbData*nbSamples);
% 	model.Sigma(:,:,i) = DataTmp * diag(GAMMA2(i,:)) * DataTmp';
% end

%Refinement of parameters
[model, H] = EM_GMM(Data, model);

%Precomputation of inverse and eigencomponents (optional)
for i=1:model.nbStates
	[model.V(:,:,i), model.D(:,:,i)] = eigs(model.Sigma(:,:,i));
	model.invSigma(:,:,i) = inv(model.Sigma(:,:,i));
end

%Set list of states according to first demonstration (alternatively, an HSMM can be used)
[~,qList] = max(H(:,1:nbData),[],1); %works also for nbStates=1

%Create single Gaussian N(MuQ,SigmaQ) based on optimal state sequence q, see Eq. (27)
MuQ = reshape(model.Mu(:,qList), model.nbVar*nbData, 1); 

L = zeros(model.nbVar*nbData);
for t=1:nbData
	id = (t-1)*model.nbVar+1:t*model.nbVar;
	Ltmp = chol(inv(model.Sigma(:,:,qList(t))));	
	L(id,id)       = Ltmp;
end	
%SigmaQ can alternatively be computed with: 
%SigmaQ = (kron(ones(nbData,1), eye(model.nbVar)) * reshape(model.Sigma(:,:,qList), model.nbVar, model.nbVar*nbData)) .* kron(eye(nbData), ones(model.nbVar));


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbRepros
	X = Data(:,1) + [randn(model.nbVarPos,1)*2E0; zeros(model.nbVarPos,1)];
 		
	% Create constraint for each control action u_t:
	% All u_t need to make sure that the state constraint Ax*x < bx is met.
	% So we first set-up state constraints for all nbData states:
	Axall = kron(eye(nbData),Ax);
	bxall = repmat(bx,nbData,1);
	% Then we transform them to u_t space:
	Au = Axall*Su;
	bu = bxall - Axall*Sx*X;
	
	% Minimization function:
	% min (||L*Su*U - L*(Muq-Sx*x)|| + rFactor*||U||)
	W = L*Su;
	v = L*(MuQ-Sx*X);	
	H =	2*(W'*W+R);
	f = -2*v'*W;	

	% Set optimizer options:
	options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');
	% Solve problem:
	uc = quadprog(H,f,Au,bu,[],[],[],[],[],options);
	u = quadprog(H,f,[],[],[],[],[],[],[],options);
 	
	% Recompute trajectory:
	r(n).Data = reshape(Sx*X+Su*u, model.nbVar, nbData);
	% Recompute trajectory:
	rnew(n).Data = reshape(Sx*X+Su*uc, model.nbVar, nbData);
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1]); hold on; axis off;

% Plot GMM and Data;
plotGMM(model.Mu(1:2,:), model.Sigma(1:2,1:2,:), [0.5 0.5 0.5],.3);
for n=1:nbSamples
	dem = plot(s(n).Data(1,:), s(n).Data(2,:), '-','color',[.7 .7 .7]);
end

cmap = colormap('lines');
for n=1:nbRepros
	lunc = plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',2,'color',[1 .7 .7]);
end
for n=1:nbRepros
	lc   = plot(rnew(n).Data(1,:), rnew(n).Data(2,:), '-','linewidth',3,'color',[.8 0 0]);
end

% Plot Constraint:
x1 = -10:.1:10;
x2 = Ax(:,2)\(bx-Ax(:,1)*x1);
cline = plot(x1,x2,'-','color',[0,0,0],'linewidth',3);

leg = legend([dem,cline,lunc,lc],'Demonstrations','Convex Constraint $A_xx<b_x$','Unconstrained reproduction','Constrained repro');
set(leg,'Interpreter','Latex','Location','Best');

axis equal; 

set(gca,'xlim',[min(Data(1,:))*1.2,max(Data(1,:))*1.2],'ylim',[min(Data(2,:))*1.2,max(Data(2,:))*1.2]);

% print('-dpng','graphs/demo_batchLQR01_constrained.png');
pause;
close all;