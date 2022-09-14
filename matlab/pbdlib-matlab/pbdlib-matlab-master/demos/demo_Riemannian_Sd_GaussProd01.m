function demo_Riemannian_Sd_GaussProd01
% Product of Gaussians on a d-sphere by relying on Riemannian manifold. 
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
nbSamples = 5; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids

model.nbStates = 2; %Number of Gaussians
model.nbVar = 3; %Dimension of the tangent space
model.params_diagRegFact = 1E-3; %Regularization of covariance


%% Setting Gaussian parameters explicitly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model.MuMan = randn(model.nbVarMan,model.nbStates);
model.MuMan(:,1) = [.1; -.3; .1];
model.MuMan(:,2) = [-.4; -.4; .7];
for i=1:model.nbStates
	model.MuMan(:,i) = model.MuMan(:,i) / norm(model.MuMan(:,i));
end
model.Mu = zeros(model.nbVar,model.nbStates);
model.Sigma(:,:,1) = rotM(model.MuMan(:,1))' * diag([.01,.2,0]) * rotM(model.MuMan(:,1));
model.Sigma(:,:,2) = rotM(model.MuMan(:,2))' * diag([.4,.02,0]) * rotM(model.MuMan(:,2));


%% Sampling from GMM 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nb = round(nbData*nbSamples/model.nbStates);
x = [];
for i=1:model.nbStates
	if i==model.nbStates
		nb = nbData*nbSamples - (model.nbStates-1)*nb;
	end
	[V,D] = eig(model.Sigma(:,:,i));
	utmp = V*D.^.5 * randn(model.nbVar,nb);
	x = [x, expmap(utmp, model.MuMan(:,i))];
end
%Compute points on tangent spaces
for i=1:model.nbStates
	u(:,:,i) = logmap(x, model.MuMan(:,i));
end


%% Product of Gaussians (version transporting the covariances)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MuMan = model.MuMan(:,1);
%MuMan = [0; -1; 0]; 
%MuMan = [0; 0; 1];
MuMan = [-1; -1; 0];
MuMan = MuMan / norm(MuMan);

%Compute MuMan incrementally
MuTmp = zeros(model.nbVar,model.nbStates);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
for n=1:nbIter
	Mu = zeros(model.nbVar,1);
	SigmaSum = zeros(model.nbVar);
	for i=1:model.nbStates
		%Transportation of covariance from model.MuMan(:,i) to MuMan 
		Ac = transp(model.MuMan(:,i), MuMan);
		SigmaTmp(:,:,i) = Ac * model.Sigma(:,:,i) * Ac';
		%Tracking component for Gaussian i
		SigmaSum = SigmaSum + inv(SigmaTmp(:,:,i));
		MuTmp(:,i) = logmap(model.MuMan(:,i), MuMan);
		Mu = Mu + SigmaTmp(:,:,i) \ MuTmp(:,i);
	end
	Sigma = inv(SigmaSum);
	%Gradient computation
	Mu = Sigma * Mu;	
% 	%Keep an history for plotting
% 	hist(n).MuMan = MuMan;
% 	hist(n).Sigma = Sigma;
	%Update MuMan
	MuMan = expmap(Mu, MuMan);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVar, nbDrawingSeg, model.nbStates);

%Display of covariance contours on the sphere
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	[d,id] = sort(diag(D),'descend');
	V = V(:,id);
	D = diag(d);
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl); zeros(1,nbDrawingSeg)], model.MuMan(:,i));
end
% for n=1:nbIter
% 	[V,D] = eig(hist(n).Sigma);
% 	hist(n).Gdisp2 = expmap(V*D.^.5*[cos(tl); sin(tl)], hist(n).MuMan);
% end
[V,D] = eig(Sigma);
[d,id] = sort(diag(D),'descend');
V = V(:,id);
D = diag(d);
Gdisp2 = expmap(V*D.^.5*[cos(tl); sin(tl); zeros(1,nbDrawingSeg)], MuMan);

%Manifold plot
figure('PaperPosition',[0 0 12 12],'color',[1 1 1],'position',[10,10,800,800]); hold on; axis off; grid off; rotate3d on; 
colormap(repmat(linspace(.95,.1,64),3,1)');
colormap([.8 .8 .8]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);

% plot3(x(1,:), x(2,:), x(3,:), '.','markersize',12,'color',[0 0 0]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',12,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
end
view(-20,15); axis equal; axis vis3d;
% print('-dpng','graphs/demo_Riemannian_Sd_GaussProd01.png');

plot3(MuMan(1), MuMan(2), MuMan(3), '.','markersize',12,'color',[0 0 0]);
plot3(Gdisp2(1,:), Gdisp2(2,:), Gdisp2(3,:), '-','linewidth',2,'color',[0 0 0]);	
% print('-dpng','graphs/demo_Riemannian_Sd_GaussProd02.png');

% %Plot history
% for n=1:nbIter
% 	coltmp = [.3 1 .3] * (nbIter-n)/nbIter;
% 	plot3(hist(n).MuMan(1), hist(n).MuMan(2), hist(n).MuMan(3), '.','markersize',12,'color',coltmp);
% 	plot3(hist(n).Gdisp2(1,:), hist(n).Gdisp2(2,:), hist(n).Gdisp2(3,:), '-','linewidth',1,'color',coltmp);
% end

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