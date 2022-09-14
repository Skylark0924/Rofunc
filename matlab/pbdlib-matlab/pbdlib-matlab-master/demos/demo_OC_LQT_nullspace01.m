function demo_OC_LQT_nullspace01
% Batch LQT with nullspace formulation.
% (see also demo_MPC_legibility01.m)
%
% If this code is useful for your research, please cite the related publication:
% @article{Girgin19,
% 	author="Girgin, H. and Calinon, S.",
% 	title="Nullspace Structure in Model Predictive Control",
% 	journal="arXiv:1905.09679",
% 	year="2019",
% 	pages="1--16"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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
nbData = 50; %Number of datapoints
nbRepros = 60; %Number of stochastic reproductions
nbPoints = 1; %Number of keypoints
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 1; %Number of static and dynamic features (nbDeriv=2 for [x,dx] and u=ddx)
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1E-1; %Time step duration
rfactor = 1E-4; %dt^nbDeriv;	%Control cost in LQR
R = speye((nbData-1)*nbVarPos) * rfactor;
Rx = speye(nbData*nbVar) * rfactor;
x0 = zeros(nbVar,1); %Initial point


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A1d = zeros(nbDeriv);
%for i=0:nbDeriv-1
%	A1d = A1d + diag(ones(nbDeriv-i,1),i) * dt^i * 1/factorial(i); %Discrete 1D
%end
%B1d = zeros(nbDeriv,1); 
%for i=1:nbDeriv
%	B1d(nbDeriv-i+1) = dt^i * 1/factorial(i); %Discrete 1D
%end
%A = kron(A1d, speye(nbVarPos)); %Discrete nD
%B = kron(B1d, speye(nbVarPos)); %Discrete nD

%%Build Sx and Su transfer matrices
%Su = sparse(nbVar*nbData, nbVarPos*(nbData-1));
%Sx = kron(ones(nbData,1), speye(nbVar));
%M = B;
%for n=2:nbData
%	id1 = (n-1)*nbVar+1:nbData*nbVar;
%	Sx(id1,:) = Sx(id1,:) * A;
%	id1 = (n-1)*nbVar+1:n*nbVar; 
%	id2 = 1:(n-1)*nbVarPos;
%	Su(id1,id2) = M;
%	M = [A*M(:,1:nbVarPos), M]; 
%end

%For nbDeriv = 1
Su = [zeros(nbVarPos, nbVarPos*(nbData-1)); kron(tril(ones(nbData-1)), eye(nbVarPos)*dt)];
Sx = kron(ones(nbData,1), eye(nbVarPos));


%% Task setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tl = linspace(1,nbData,nbPoints+1);
tl = round(tl(2:end)); %[nbData/2, nbData];
%Mu = rand(nbVarPos,nbPoints) - 0.5; 
Mu = [20; 10];

Sigma = repmat(eye(nbVarPos).*1E-3, [1,1,nbPoints]);
% Sigma(:,:,2) = Sigma(:,:,2) .* 1E2;

MuQ = zeros(nbVar*nbData,1); 
Q = zeros(nbVar*nbData);
for t=1:length(tl)
	id(:,t) = [1:nbVarPos] + (tl(t)-1) * nbVar;
	Q(id(:,t), id(:,t)) = inv(Sigma(:,:,t));
	MuQ(id(:,t)) = Mu(:,t);
end


%% Batch LQR reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Precomputation of basis functions to generate structured stochastic u through Bezier curves
nbRBF = 10;
H = zeros(nbRBF,nbData-1);
tl = linspace(0,1,nbData-1);
nbDeg = nbRBF - 1;
for i=0:nbDeg
	H(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-tl).^(nbDeg-i) .* tl.^i; %Bernstein basis functions
end

%Reproduction with nullspace planning
[V,D] = eig(Q);
U = V * D.^.5;
J = U' * Su; %Jacobian

% %Right pseudoinverse solution
% pinvJ = J' / (J * J' + Rx); %Right pseudoinverse
% N = eye((nbData-1)*nbVarPos) - pinvJ * J; %Nullspace projection matrix
% u1 = pinvJ * U' * (MuQ - Sx * x0); %Principal task (least squares solution)

%Left pseudoinverse solution
% pinvJ = pinv(J);
pinvJ = (J' * J + R) \ J'; %Left pseudoinverse
N = speye((nbData-1)*nbVarPos) - pinvJ * J; %Nullspace projection matrix
u1 = pinvJ * U' * (MuQ - Sx * x0); %Principal task (least squares solution)
x = reshape(Sx*x0+Su*u1, nbVar, nbData); %Reshape data for plotting

%General solutions
%w = randn(nbVarPos,nbRBF) * 1E1;
for n=1:nbRepros
	w = randn(nbVarPos,nbRBF) * 1E1; %Random weights
	%w(:,5) = w(:,5) + [1; -.1]; 
	
	u2 = w * H; %Reconstruction of control signals by a weighted superposition of basis functions
	u = u1 + N * u2(:);
	r(n).x = reshape(Sx*x0+Su*u, nbVar, nbData); %Reshape data for plotting
end


%% Plot 2D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 800 800],'color',[1 1 1],'name','x1-x2 plot'); hold on; axis off;
%plotGMM(Mu, Sigma, [.8 0 0], .2);
%plot(Mu(1,:), Mu(2,:), '.','markersize',30,'color',[.8 0 0]);
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',1,'color',.9-[.7 .7 .7].*rand(1));
end
plot(x(1,:), x(2,:), '-','linewidth',2,'color',[.8 0 0]);
plot(x(1,1), x(2,1), 'o','linewidth',2,'markersize',8,'color',[.8 0 0]);
plot(MuQ(id(1,:)), MuQ(id(2,:)), '.','markersize',30,'color',[.8 0 0]);
axis equal; 
%print('-dpng','graphs/LQT_nullspace02.png');

% %Plot nullspace matrix
% figure('position',[10 1050 300 300]); hold on; axis off;
% colormap(flipud(gray));
% imagesc(abs(N)); 
% axis tight; axis square; axis ij;

pause;
close all;
