function demo_Riemannian_S2_GaussProd01
% Product of Gaussians on a sphere by relying on Riemannian manifold. 
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
model.nbVar = 2; %Dimension of the tangent space
model.nbVarMan = 3; %Dimension of the manifold
model.params_diagRegFact = 1E-3; %Regularization of covariance


%% Setting GMM parameters explicitly
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model.MuMan = randn(model.nbVarMan,model.nbStates);
model.MuMan(:,1) = [-.6; -.6; -.5];
model.MuMan(:,2) = [-2; -.4; -2];
for i=1:model.nbStates
	model.MuMan(:,i) = model.MuMan(:,i) / norm(model.MuMan(:,i));
end
model.Mu = zeros(model.nbVar,model.nbStates);
model.Sigma(:,:,1) = diag([.1,.9]) * 1E-1;
model.Sigma(:,:,2) = diag([4,.2]) * 1E-1;


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
%Compute likelihoods
L = zeros(model.nbStates,size(x,2));
for i=1:model.nbStates
	L(i,:) = gaussPDF(logmap(x, model.MuMan(:,i)), model.Mu(:,i), model.Sigma(:,:,i));
end
GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);


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
% 	%Keep log data for plotting
% 	hist(n).MuMan = MuMan;
% 	hist(n).Sigma = Sigma;
	%Update MuMan
	MuMan = expmap(Mu, MuMan);
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);

%Display of covariance contours on the sphere
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(:,i));
end
% for n=1:nbIter
% 	[V,D] = eig(hist(n).Sigma);
% 	hist(n).Gdisp2 = expmap(V*D.^.5*[cos(tl); sin(tl)], hist(n).MuMan);
% end
[V,D] = eig(Sigma);
R = real(V*D.^.5);
Gdisp2 = expmap(R*[cos(tl); sin(tl)], MuMan);

%Manifold plot
figure('PaperPosition',[0 0 12 12],'color',[1 1 1],'position',[10,10,1300,1300]); hold on; axis off; grid off; rotate3d on; 
colormap(repmat(linspace(.95,.1,64),3,1)');

%colored sphere
nbp = 40;
[X,Y,Z] = sphere(nbp-1);
p = [reshape(X,1,nbp^2); reshape(Y,1,nbp^2); reshape(Z,1,nbp^2)];
c = zeros(nbp^2,1);
for i=1:model.nbStates
	dtmp = logmap(p,model.MuMan(:,i))';
	c = c + sum((dtmp/model.Sigma(:,:,i)).*dtmp, 2);
end
surf(X,Y,Z,reshape(c,nbp,nbp),'linestyle','none');

% plot3(x(1,:), x(2,:), x(3,:), '.','markersize',12,'color',[0 0 0]);
for n=1:nbData*nbSamples
	plot3(x(1,n), x(2,n), x(3,n), '.','markersize',12,'color',H(:,n)'*clrmap);
end
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',12,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',2,'color',clrmap(i,:));
end
view(-70,-30); axis equal; axis vis3d;
% print('-dpng','graphs/demo_Riemannian_S2_GaussProd01a.png');

plot3(MuMan(1), MuMan(2), MuMan(3), '.','markersize',12,'color',[0 0 0]);
plot3(Gdisp2(1,:), Gdisp2(2,:), Gdisp2(3,:), '-','linewidth',2,'color',[0 0 0]);	
% print('-dpng','graphs/demo_Riemannian_S2_GaussProd01b.png');

% %Plot log history
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