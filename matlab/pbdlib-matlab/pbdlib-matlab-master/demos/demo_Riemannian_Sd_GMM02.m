function demo_Riemannian_Sd_GMM02
% GMM for data on a d-sphere (here, with a ball) by relying on Riemannian manifold
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
model.nbVar = 3; %Dimension for the embedding of the sphere
model.nbStates = 4; %Number of states
model.params_diagRegFact = 1E-8; %Regularization of covariance

nbData = 50; %Number of datapoints
nbSamples = 5; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 30; %Number of iteration for the EM algorithm


%% Generate data on the sphere 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x = [randn(model.nbVar,nbData/2).*5E-1-1, randn(model.nbVar,nbData/2).*1E0+1];
% for t=1:nbData
% 	x(:,t) = x(:,t) ./ norm(x(:,t));
% end
% u = logmap(x, [0;0;1]);

demos=[];
load('data/2Dletters/S.mat');
x=[]; 
for n=1:nbSamples
	xTmp = [spline(1:size(demos{n}.pos,2), demos{n}.pos.*1.3E-1, linspace(1,size(demos{n}.pos,2),nbData)); ones(1,nbData)];
	for t=1:nbData
		xTmp(:,t) = xTmp(:,t) ./ norm(xTmp(:,t));
	end
	x = [x, xTmp];
end
u = logmap(x, [0;0;1]);


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Random initialization
% model.Priors = ones(1,model.nbStates) / model.nbStates;
% model.MuMan = expmap(randn(model.nbVar,model.nbStates), randn(model.nbVarMan,1)); %Center on the manifold
% model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
% model.Sigma = repmat(eye(model.nbVar)*5E-1,[1,1,model.nbStates]); %Covariance in the tangent plane at point MuMan

%Initialization based on kbins
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu, [0;0;1]); %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x, model.MuMan(:,i)), model.Mu(:,i), model.Sigma(:,:,i)+eye(model.nbVar)*1E-4);
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			u(:,:,i) = logmap(x, model.MuMan(:,i));
			model.MuMan(:,i) = expmap(u(:,:,i)*H(i,:)', model.MuMan(:,i));
		end
		%Update Sigma
		model.Sigma(:,:,i) = u(:,:,i) * diag(H(i,:)) * u(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids
clrmap = lines(model.nbStates);
[X,Y,Z] = sphere(20);

%Computation of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVar, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	[d,id] = sort(diag(D),'descend');
	V = V(:,id);
	D = diag(d);
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl); zeros(1,nbDrawingSeg)], model.MuMan(:,i));
% 	Gdisp2(:,:,i) = real(V*D.^.5) * [cos(tl); sin(tl); zeros(1,nbDrawingSeg)] + repmat(model.MuMan(:,i),1,nbDrawingSeg); 
end

figure('position',[10,10,800,800]); hold on; axis off; rotate3d on; 
colormap([.9 .9 .9]);
mesh(X,Y,Z);
plot3(x(1,:), x(2,:), x(3,:), '.','markersize',10,'color',[0 0 0]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',20,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',1,'color',clrmap(i,:));
% 	%Plot tangent space
% 	msh = repmat(model.MuMan(:,i),1,5) + rotM(model.MuMan(:,i))' * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
% 	patch(msh(1,:),msh(2,:),msh(3,:), [.8 .8 .8],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
% 	%Plot covariance in tangent space
% 	%plotGMM3D(model.MuMan, model.Sigma, [.7 .7 .7],.2);
% 	plot3(Gdisp2(1,:,i), Gdisp2(2,:,i), Gdisp2(3,:,i), '-','linewidth',1,'color',clrmap(i,:));
end
view(0,70); axis equal; axis tight; axis vis3d;  
%print('-dpng','graphs/demo_Riemannian_Sd_GMM02.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, x0)
	theta = sqrt(sum(u.^2,1)); 
	x = real(repmat(x0,[1,size(u,2)]) .* repmat(cos(theta),[size(u,1),1]) + u .* repmat(sin(theta)./theta,[size(u,1),1]));
	x(:,theta<1e-16) = repmat(x0,[1,sum(theta<1e-16)]);	
end

function u = logmap(x, x0)
	theta = acos(x0'*x);	
	u = (x - repmat(x0,[1,size(x,2)]) .* repmat(cos(theta),[size(x,1),1])) .* repmat(theta./sin(theta),[size(x,1),1]);
	u(:,theta<1e-16) = 0;
end

function Ac = transp(x1,x2,t)
	if nargin==2
		t=1;
	end
	u = logmap(x2,x1);
	e = norm(u,'fro');
	u = u ./ (e+realmin);
	Ac = -x1 * sin(e*t) * u' + u * cos(e*t) * u' + eye(size(u,1)) - u * u';
end