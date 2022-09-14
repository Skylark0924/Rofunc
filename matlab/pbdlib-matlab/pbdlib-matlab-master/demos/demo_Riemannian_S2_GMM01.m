function demo_Riemannian_S2_GMM01
% GMM for data on a sphere by relying on Riemannian manifold. 
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
nbData = 50; %Number of datapoints
nbSamples = 8; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 30; %Number of iteration for the EM algorithm
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids

model.nbStates = 1; %Number of states in the GMM
model.nbVar = 2; %Dimension of the tangent space
model.nbVarMan = 3; %Dimension of the manifold
model.params_diagRegFact = 1E-4; %Regularization of covariance
x0 = [0; -1; 0]; %Point on the sphere to project handwriting data


%% Generate data on a sphere from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/N.mat');
u=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	u = [u [s(n).Data*1.1E-1]]; 
end
x = expmap(u, x0);


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %Random initialization
% model.Priors = ones(1,model.nbStates) / model.nbStates;
% model.MuMan = expmap(randn(model.nbVar,model.nbStates), randn(model.nbVarMan,1)); %Center on the manifold
% model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
% model.Sigma = repmat(eye(model.nbVar)*5E-1,[1,1,model.nbStates]); %Covariance in the tangent plane at point MuMan

%Initialization based on k-bins/k-means
model = init_GMM_kbins(u, model, nbSamples);
% model = init_GMM_kmeans(u, model);
model.MuMan = expmap(model.Mu, x0); %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x, model.MuMan(:,i)), model.Mu(:,i), model.Sigma(:,:,i));
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
clrmap = lines(model.nbStates);

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(:,i));
end

%Display of covariance contours on the tangent space
Gdisp2 = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	S = blkdiag(real(V*D.^.5), 1);
	Gdisp2(:,:,i) = rotM(model.MuMan(:,i))' * S * [cos(tl); sin(tl); zeros(1,nbDrawingSeg)] + repmat(model.MuMan(:,i),1,nbDrawingSeg);
end

%Manifold plot
figure('position',[10,10,650,650]); hold on; axis off; grid off; rotate3d on; 
colormap([.8 .8 .8]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);
%plot3(x(1,:), x(2,:), x(3,:), '--','markersize',12,'color',[0 0 0]);
for t=1:nbData*nbSamples
	plot3(x(1,t), x(2,t), x(3,t), '.','markersize',12,'color',GAMMA(:,t)'*clrmap);
end
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',20,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
% 	%Draw tangent plane
% 	msh = repmat(model.MuMan(:,i),1,5) + rotM(model.MuMan(:,i))' * [1 1 -1 -1 1; 1 -1 -1 1 1; 0 0 0 0 0] * 1E0;
% 	patch(msh(1,:),msh(2,:),msh(3,:), [.8 .8 .8],'edgecolor',[.6 .6 .6],'facealpha',.3,'edgealpha',.3);
% 	plot3(Gdisp2(1,:,i), Gdisp2(2,:,i), Gdisp2(3,:,i), '-','linewidth',1,'color',clrmap(i,:));
end
view(-20,15); axis equal; axis tight; axis vis3d;  
% print('-dpng','graphs/Riemannian_sphere_GMM01a.png');


%% Tangent space plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[670,10,1950,650]); 
for i=1:model.nbStates
	clrmap2 = ones(model.nbStates) .* .6;
	clrmap2(i,:) = clrmap(i,:);
	%subplot(ceil(model.nbStates/2),2,i); hold on; axis off; %title(['k=' num2str(i)]);
	subplot(1,model.nbStates,i); hold on; axis off; %title(['k=' num2str(i)]);
	plot(0,0,'+','markersize',40,'linewidth',2,'color',[.7 .7 .7]);
	%plot(u(1,:,i), u(2,:,i), '.','markersize',6,'color',[0 0 0]);
	for t=1:nbData*nbSamples
		plot(u(1,t,i), u(2,t,i), '.','markersize',12,'color',GAMMA(:,t)'*clrmap2); %w(1)*[1 0 0] + w(2)*[0 1 0]
	end
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i)*3, clrmap(i,:), .3);
	axis equal; axis tight;
	
% 	%Plot contours of Gaussian j in tangent plane of Gaussian i (version 1)
% 	for j=1:model.nbStates
% 		if j~=i
% 			udisp = logmap(Gdisp(:,:,j), model.MuMan(:,i));
% 			patch(udisp(1,:), udisp(2,:), clrmap(j,:),'lineWidth',1,'EdgeColor',clrmap(j,:)*0.5,'facealpha',.1,'edgealpha',.1);	
% 		end
% 	end

% 	%Plot contours of Gaussian j in tangent plane of Gaussian i (version 2)
% 	for j=1:model.nbStates
% 		if j~=i
% 			%Computing the eigenvectors of Gaussian j
% 			[V,D] = eig(model.Sigma(:,:,j));
% 			U0 = V * D.^.5;
% 			xU0 = expmap(U0, model.MuMan(:,j));
% 			%Transportation of Mu
% 			uMuTmp = logmap(model.MuMan(:,j), model.MuMan(:,i));
% 			%Transportation of eigenvectors and reconstruction of the corresponding Sigma
% 			U = logmap(xU0, model.MuMan(:,i));
% 			Uc = U - repmat(uMuTmp,1,model.nbVar);
% 			SigmaTmp = Uc * Uc';
% 			plotGMM(uMuTmp, SigmaTmp*3, clrmap(j,:), .1);
% 		end
% 	end
		
end

% subplot(2,2,4); hold on; axis off;
% u0 = logmap(x, [0; 0; 1]);
% plot(u0(1,:), u0(2,:), '.','markersize',6,'color',[.5 .5 .5]);

% print('-dpng','graphs/Riemannian_sphere_GMM01b.png');

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