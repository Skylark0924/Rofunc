function demo_Riemannian_Sd_GMR01
% Retrieval of periodic motion with GMR, by considering input data on a 1-sphere and output data in Euclidean space
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
nbData = 100; %Number of datapoints
nbSamples = 5; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 20; %Number of iteration for the EM algorithm

model.nbStates = 6; %Number of states in the GMM
model.nbVar = 5; %Dimension of the tangent space (2D circle input + 3D output)
model.params_diagRegFact = 1E-6; %Regularization of covariance


%% Generate input data on a 0-sphere and Euclidean output data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
th = repmat(linspace(-pi,pi,nbData),1,nbSamples);
xIn = [cos(th); sin(th)];
uIn = logmap(xIn, [0;1]);

demos=[];
load('data/2Dletters/B.mat');
uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut, s(n).Data([1:end,end],:)*1.3E-1]; 
end
xOut = uOut;

x = [xIn; xOut];
u = [uIn; uOut];


%% GMM parameters estimation (joint distribution with sphere as input, Euclidean space as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [expmap(model.Mu(1:2,:), [0;1]); model.Mu(3:5,:)]; %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		uTmp(:,:,i) = [logmap(xIn, model.MuMan(1:2,i)); xOut-repmat(model.MuMan(3:5,i),1,nbData*nbSamples)];
		L(i,:) = model.Priors(i) * gaussPDF(uTmp(:,:,i), model.Mu(:,i), model.Sigma(:,:,i)+eye(model.nbVar)*1E-2);
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [logmap(xIn, model.MuMan(1:2,i)); xOut-repmat(model.MuMan(3:5,i),1,nbData*nbSamples)];
			model.MuMan(:,i) = [expmap(uTmp(1:2,:,i)*H(i,:)', model.MuMan(1:2,i)); (uTmp(3:5,:,i)+repmat(model.MuMan(3:5,i),1,nbData*nbSamples))*H(i,:)']; 
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(H(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1:2; out=3:5; 
inMan=1:2; outMan=3:5;
nbVarOut = length(out);
nbVarOutMan = length(outMan);

xhat = zeros(nbVarOutMan,nbData);
uOut = zeros(nbVarOut,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbData);

%VERSION with single optimization loop
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(logmap(xIn(:,t), model.MuMan(inMan,i)), model.Mu(in,i), model.Sigma(in,in,i)+eye(length(in))*1E-2);
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
nbDrawingSeg = 50;
tl = linspace(-pi, pi, nbDrawingSeg);
%Computation of covariance contours on the circle
Gdisp = zeros(2, nbDrawingSeg, model.nbStates);
Gdisp2 = zeros(2, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(1:2,1:2,i));
	Gdisp(:,:,i) = expmap(real(V*D.^.5)*[cos(tl); sin(tl)], model.MuMan(1:2,i));
	Gdisp2(:,:,i) = real(V*D.^.5) * [cos(tl); sin(tl)] + repmat(model.MuMan(1:2,i),1,nbDrawingSeg);
end

figure('PaperPosition',[0 0 12 6],'position',[10,10,1350,650],'name','manifold data'); rotate3d on; 
colormap([.9 .9 .9]);

%in
subplot(1,2,1); hold on; axis off; grid off;
plot(x(1,:), x(2,:), '.','markersize',6,'color',[.4 .4 .4]);
for i=1:model.nbStates
	plot(model.MuMan(1,i), model.MuMan(2,i), '.','markersize',20,'color',clrmap(i,:));
	%Plot covariance in tangent space adn on manifold
	plot(Gdisp(1,:,i), Gdisp(2,:,i), '-','linewidth',1,'color',clrmap(i,:));
	plot(Gdisp2(1,:,i), Gdisp2(2,:,i), '-','linewidth',1,'color',clrmap(i,:));
end
axis equal; 

%out
subplot(1,2,2); hold on; axis off; grid off;
plot3(x(3,:), x(4,:), x(5,:), '.','markersize',6,'color',[.4 .4 .4]);
for i=1:model.nbStates
	plotGMM3D(model.MuMan(3:5,i), model.Sigma(3:5,3:5,i), clrmap(i,:), .1);
end
plot3(xhat(1,:), xhat(2,:), xhat(3,:), '-','linewidth',3,'color',[.8 0 0]);
view(-20,2); axis equal; axis vis3d;  
% print('-dpng','graphs/demo_Riemannian_Sd_GMR01.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u,x0)
	theta = sqrt(sum(u.^2,1)); 
	x = real(repmat(x0,[1,size(u,2)]) .* repmat(cos(theta),[size(u,1),1]) + u .* repmat(sin(theta)./theta,[size(u,1),1]));
	x(:,theta<1e-16) = repmat(x0,[1,sum(theta<1e-16)]);	
end

function u = logmap(x,x0)
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