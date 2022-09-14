function demo_OC_LQT_viapoints03
% Equivalence between cubic Bezier curve and batch LQR with double integrator (formulation with position and velocity).
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
% @inproceedings{Berio17GI,
%   author="Berio, D. and Calinon, S. and Fol Leymarie, F.",
%   title="Generating Calligraphic Trajectories with Model Predictive Control",
%   booktitle="Proc. 43rd Conf. on Graphics Interface",
%   year="2017",
%   month="May",
%   address="Edmonton, AL, Canada",
%   pages="132--139",
%   doi="10.20380/GI2017.17"
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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
nbData = 100; %Number of datapoints
nbRepros = 1; %Number of reproductions

nbStates = 2; %Number of Gaussians in the GMM
nbVarPos = 2; %Dimension of position data (here: x1,x2)
nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
nbVar = nbVarPos * nbDeriv; %Dimension of state vector
dt = 1/nbData; %Time step duration
rfactor = 1E-10;	%Control cost in LQR

%Control cost matrix
R = eye(nbVarPos) * rfactor;
R = kron(eye(nbData-1),R);

%Setting cubic Bezier curve parameters
P = rand(nbVarPos, nbStates*2);
Mu = [P(:,[1,end]); zeros(nbVarPos, nbStates)];


%% Dynamical System settings (discrete version)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Integration with higher order Taylor series expansion
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

%Build Sx and Su matrices for batch LQR, see Eq. (35)
Su = zeros(nbVar*nbData, nbVarPos*(nbData-1));
Sx = kron(ones(nbData,1),eye(nbVar));
M = B;
for n=2:nbData
	id1 = (n-1)*nbVar+1:nbData*nbVar;
	Sx(id1,:) = Sx(id1,:) * A;
	id1 = (n-1)*nbVar+1:n*nbVar; 
	id2 = 1:(n-1)*nbVarPos;
	Su(id1,id2) = M;
	M = [A*M(:,1:nbVarPos), M]; %Also M = [A^(n-1)*B, M] or M = [Sx(id1,:)*B, M]
end


%% Cubic Bezier curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Cubic Bezier curve plot from Bernstein polynomials
%See e.g. http://blogs.mathworks.com/graphics/2014/10/13/bezier-curves/ or 
%http://www.ams.org/samplings/feature-column/fcarc-bezier#2
t = linspace(0,1,nbData);
x = P * [(1-t).^3; 3.*(1-t).^2.*t; 3.*(1-t).*t.^2; t.^3];
% x = P(:,1) * (1-t).^3 + P(:,2) * 3.*(1-t).^2.*t + P(:,3) * 3.*(1-t).*t.^2 + P(:,4) * t.^3;
% x = kron((1-t).^3, P(:,1)) + kron(3*(1-t).^2.*t, P(:,2)) + kron(3*(1-t).*t.^2, P(:,3)) + kron(t.^3, P(:,4));
%dx = kron(3*(1-t).^2, P(:,2)-P(:,1)) + kron(6*(1-t).*t, P(:,3)-P(:,2)) + kron(3*t.^2, P(:,4)-P(:,3));
% Mu(3:4,1) = dx(1:2,1);
% Mu(3:4,2) = dx(1:2,end);
Mu(3:4,2) = (P(1:2,4) - P(1:2,3)) * 3;
Mu(3:4,1) = (P(1:2,2) - P(1:2,1)) * 3;


%% MPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
invSigma = repmat(eye(nbVar)*1E0, [1,1,nbStates]);
% invSigma = repmat(blkdiag(eye(nbVarPos),ones(nbVarPos)*1E-18), [1,1,nbStates]);
% xt = rand(2,1);
% xt = xt /norm(xt);
% invSigma = repmat(blkdiag(xt*xt'+eye(nbVarPos)*1E-4, eye(nbVarPos)*1E-18), [1,1,nbStates]);

%Create single Gaussian N(MuQ,Q^-1) 
% qList = round(linspace(1,nbStates,nbData));
% MuQ = reshape(Mu(:,qList), nbVar*nbData, 1); %Only the first and last values will be used
MuQ = [Mu(:,1); zeros(nbVar*(nbData-2),1); Mu(:,2)]; %Only the first and last values will be used

%Set cost for two viapoints at the beginning and at the end
Q = blkdiag(invSigma(:,:,1), zeros(nbVar*(nbData-2)), invSigma(:,:,2));

%Batch LQR reproduction
X = Mu(:,1); 
%X = [Mu(1:2,1); zeros(2,1)]; 
for n=1:nbRepros	
	SuInvSigmaQ = Su' * Q;
	Rq = SuInvSigmaQ * Su + R;
	rq = SuInvSigmaQ * (MuQ-Sx*X);
	u = Rq \ rq; 
	r(n).x = reshape(Sx*X+Su*u, nbVar, nbData);
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1200 1200],'color',[1 1 1]); hold on; axis off;
plotGMM(Mu(1:2,:), invSigma(1:2,1:2,:)*1E-4, [0 0 0], .3);
plotGMM(P(1:2,2:3), invSigma(3:4,3:4,:)*1E-4, [0 0 0], .3);
%plot(P(1,:), P(2,:), '.','color',[.7 .7 .7]);
plot(P(1,1:2), P(2,1:2), '-','color',[.7 .7 .7]);
plot(P(1,3:4), P(2,3:4), '-','color',[.7 .7 .7]);
plot(x(1,:),x(2,:),'-','color',[0 .7 0]);
h(1) = plot(x(1,:),x(2,:),'.','markersize',6,'color',[0 .6 0]);
for n=1:nbRepros
	plot(r(n).x(1,:), r(n).x(2,:), '-','linewidth',1,'color',[.8 0 0]);
	h(2) = plot(r(n).x(1,:), r(n).x(2,:), '.','markersize',6,'color',[.8 0 0]);
end
legend(h,{'Standard Bezier curve computation','Computation through MPC'});
axis equal; 

%print('-dpng','graphs/demo_batchLQR_viapoints03.png');
pause;
close all;