function demo_Riemannian_SE2_GMM01
% GMM on SE(2) manifold 
% (Implementation of exp and log maps based on "Lie Groups for 2D and 3D Transformations" by Ethan Eade)
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
model.nbVarPos = 2; %Number of variables [x1,x2]
model.nbVar = model.nbVarPos+1; %Number of variables [w,x1,x2]
model.nbStates = 3; %Number of states in the GMM
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 30; %Number of iteration for the EM algorithm
nbData = 30; %Number of datapoints
nbSamples = 1;


%% Generate random homogeneous matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:model.nbStates
	uMu(1,i) = rand(1) * pi;
	uMu(2:3,i) = rand(model.nbVarPos, 1) * .2;
end
idList = repmat(kron(1:model.nbStates,ones(1,ceil(nbData/model.nbStates))),1,nbSamples);
x = zeros(model.nbVar,model.nbVar,nbData*nbSamples);
u = zeros(model.nbVar,nbData*nbSamples);
for t=1:nbData*nbSamples
	w = uMu(1,idList(t)) + rand(1) * pi * 1E-1;
	v = uMu(2:3,idList(t)) + rand(model.nbVarPos, 1) * 1E-1;
	R = [cos(w) -sin(w); sin(w) cos(w)];
	x(:,:,t) = [R, v; 0 0 1];
end
u = logmap(x,eye(model.nbVar));


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Random initialization
% model.Priors = ones(1,model.nbStates) / model.nbStates;
% model.MuMan = expmap(randn(model.nbVar,model.nbStates), randn(model.nbVarMan,1)); %Center on the manifold
% model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
% model.Sigma = repmat(eye(model.nbVar)*5E-1,[1,1,model.nbStates]); %Covariance in the tangent plane at point MuMan

%Initialization based on kbins
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu, eye(model.nbVar)); %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,nbData*nbSamples);
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x, model.MuMan(:,:,i)), model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	W = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = logmap(x, model.MuMan(:,:,i));
			model.MuMan(:,:,i) = expmap(uTmp(:,:,i)*W(i,:)', model.MuMan(:,:,i));
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(W(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1650,1250]); hold on; axis off;
plot2Dframe(x(1:2,1:2,:)*5E-2, x(1:2,end,:), min(eye(3)+.6,1));
axis equal; 
% print('-dpng','graphs/demo_Riemannian_SE2_GMM01a.png');

plot2Dframe(model.MuMan(1:2,1:2,:)*5E-2, model.MuMan(1:2,end,:), eye(3)*.8, 6);
% print('-dpng','graphs/demo_Riemannian_SE2_GMM01b.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%v%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = expmap(u,S)
	for n=1:size(u,2)
		w = u(1,n);
		v = u(2:3,n);

		%Rotation part
		R = [cos(w) -sin(w); sin(w) cos(w)];

		%Translation part
		V = [sin(w), -(1-cos(w)); 1-cos(w), sin(w)] .* (1/w);
		t = V * v;

		X(:,:,n) = S * [R, t; 0 0 1];
	end
end

function u = logmap(X,S)
	for n=1:size(X,3)
		invS = [S(1:2,1:2)', -S(1:2,1:2)' * S(1:2,end); S(end,:)];
		H = invS * X(:,:,n);

		%Rotation part
		%Htmp = -logm(H(1:2,1:2)); w = Htmp(1,2);  %implementation more efficient?
		w = atan2(H(2,1), H(1,1));

		%Translation part
		a = sin(w)/w;
		b = (1-cos(w))/w;
		invV = [a, b; -b, a] .* 1./(a^2+b^2);
		v = invV * H(1:2,end);

		u(:,n) = [w; v];
	end
end