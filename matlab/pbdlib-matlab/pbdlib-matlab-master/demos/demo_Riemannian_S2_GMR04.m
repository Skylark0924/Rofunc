function demo_Riemannian_S2_GMR04
% GMR with input data on a sphere and output data in Euclidean space by relying on Riemannian manifold. 
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
nbData = 100; %Number of datapoints
nbSamples = 5; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 20; %Number of iteration for the EM algorithm
nbDrawingSeg = 30; %Number of segments used to draw ellipsoids

model.nbStates = 3; %Number of states in the GMM
model.nbVar = 5; %Dimension of the tangent space (2D input + 3D output)
model.nbVarMan = 6; %Dimension of the manifold (3D input + 3D output)
model.params_diagRegFact = 1E-2; %Regularization of covariance


%% Generate input data on a sphere and Euclidean output data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate input data
demos=[];
load('data/2Dletters/U.mat');
uIn=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uIn = [uIn [s(n).Data*1.3E-1]]; 
end
xIn = expmap(uIn, [0; -1; 0]);
%Generate output data
demos=[];
load('data/2Dletters/C.mat');
uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut [s(n).Data([1:end,end],:)*1.3E-1]]; 
end
xOut = uOut;
x = [xIn; xOut];
u = [uIn; uOut];


%% GMM parameters estimation (joint distribution with sphere as input, Euclidean space as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [expmap(model.Mu(1:2,:), [0; -1; 0]); model.Mu(3:5,:)]; %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		uTmp(:,:,i) = [logmap(xIn, model.MuMan(1:3,i)); xOut-repmat(model.MuMan(4:6,i),1,nbData*nbSamples)];
		L(i,:) = model.Priors(i) * gaussPDF(uTmp(:,:,i), model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [logmap(xIn, model.MuMan(1:3,i)); xOut-repmat(model.MuMan(4:6,i),1,nbData*nbSamples)];
			model.MuMan(:,i) = [expmap(uTmp(1:2,:,i)*H(i,:)', model.MuMan(1:3,i)); (uTmp(3:5,:,i)+repmat(model.MuMan(4:6,i),1,nbData*nbSamples))*H(i,:)']; 
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(H(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1:2; out=3:5; 
inMan=1:3; outMan=4:6;
nbVarOut = length(out);
nbVarOutMan = length(outMan);

% %Adding artificial noise on the inputs
% xIn(:,1:nbData) = xIn(:,1:nbData) + randn(3,nbData)*2E-1; 
% for t=1:nbData
% 	xIn(:,t) = xIn(:,t) / norm(xIn(:,t));
% end

xhat = zeros(nbVarOutMan,nbData);
uOut = zeros(nbVarOut,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbData);

%VERSION with single optimization loop
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(logmap(xIn(:,t), model.MuMan(inMan,i)), model.Mu(in,i), model.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin); 

	%Compute conditional mean (with covariance transportation)
	for i=1:model.nbStates
		%Transportation of covariance (with both input and output components)
		AcIn = transp(model.MuMan(inMan,i), xIn(:,t));
		SigmaTmp(:,:,i) = blkdiag(AcIn,eye(3)) * model.Sigma(:,:,i) * blkdiag(AcIn,eye(3))';
		%Gaussian conditioning on the tangent space
		uOut(:,i,t) = model.MuMan(outMan,i) - SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * logmap(model.MuMan(inMan,i), xIn(:,t)); 
	end
	xhat(:,t) = uOut(:,:,t) * H(:,t);
	
% 	%Compute conditional covariances (by ignoring influence of centers uOut(:,i,t))
% 	for i=1:model.nbStates
% 		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaTmp(out,out,i) - SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * SigmaTmp(in,out,i));
% 	end
	
	%Compute conditional covariances (note that since uhat=0, the final part in the GMR computation is dropped)
	for i=1:model.nbStates
		SigmaOutTmp = SigmaTmp(out,out,i) - SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * SigmaTmp(in,out,i);
		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaOutTmp + uOut(:,i,t) * uOut(:,i,t)');
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);
[X,Y,Z] = sphere(20);

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
GdispIn = zeros(nbVarOutMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	%in
	[V,D] = eig(model.Sigma(1:2,1:2,i));
	GdispIn(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(1:3,i));
end
	
%Manifold plot
figure('PaperPosition',[0 0 12 6],'position',[10,10,1350,650],'name','manifold data'); rotate3d on; 
colormap([.9 .9 .9]);
%in
subplot(1,2,1); hold on; axis off; grid off;
mesh(X,Y,Z);
plot3(x(1,:), x(2,:), x(3,:), '.','markersize',6,'color',[.4 .4 .4]);
plot3(xIn(1,1:nbData), xIn(2,1:nbData), xIn(3,1:nbData), '-','linewidth',3,'color',[0 0 0]);
for i=1:model.nbStates
	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',20,'color',clrmap(i,:));
	plot3(GdispIn(1,:,i), GdispIn(2,:,i), GdispIn(3,:,i), '-','linewidth',1,'color',clrmap(i,:));
end
view(-20,2); axis equal; axis vis3d; 
%out
subplot(1,2,2); hold on; axis off; grid off;
plot3(x(4,:), x(5,:), x(6,:), '.','markersize',6,'color',[.4 .4 .4]);
for i=1:model.nbStates
	plotGMM3D(model.MuMan(4:6,i), model.Sigma(3:5,3:5,i), clrmap(i,:), .1);
end
plot3(xhat(1,:), xhat(2,:), xhat(3,:), '-','linewidth',3,'color',[.8 0 0]);
view(-20,2); axis equal; axis vis3d;  
%print('-dpng','graphs/demo_Riemannian_sphere_GMR04.png');

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