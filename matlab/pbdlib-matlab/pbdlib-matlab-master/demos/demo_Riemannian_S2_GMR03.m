function demo_Riemannian_S2_GMR03
% GMR with 3D Euclidean data as input and spherical data as output by relying on Riemannian manifold. 
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
nbSamples = 4; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 10; %Number of iteration for the EM algorithm
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 5; %Number of states in the GMM
model.nbVar = 5; %Dimension of the tangent space (3D input + 2D output)
model.nbVarMan = 6; %Dimension of the manifold (3D input + 3D output)
model.dt = 0.01; %Time step duration
model.params_diagRegFact = 1E-3; %Regularization of covariance
%e0 = [0; 0; 1]; %Origin on the manifold


%% Generate input and output data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate 3D Euclidean input data 
demos=[];
load('data/2Dletters/U.mat');
uIn=[]; 
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uIn = [uIn, s(n).Data([1:end,end],:)*1.3E-1];
end
xIn = uIn;
%Generate spherical output data 
demos=[];
load('data/2Dletters/C.mat');
uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut [s(n).Data*1.3E-1]]; 
end
xOut = expmap(uOut, [0; -1; 0]);
%xOut = expmap(uOut, e0);
u = [uIn; uOut];
x = [xIn; xOut];


%% GMM parameters estimation (joint distribution with time as input, sphere as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [model.Mu(1:3,:); expmap(model.Mu(4:5,:), [0; -1; 0])]; %Center on the manifold 
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF([xIn-repmat(model.MuMan(1:3,i),1,nbData*nbSamples); logmap(xOut, model.MuMan(4:end,i))], model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [xIn-repmat(model.MuMan(1:3,i),1,nbData*nbSamples); logmap(xOut, model.MuMan(4:end,i))];
			model.MuMan(:,i) = [(uTmp(1:3,:,i)+repmat(model.MuMan(1:3,i),1,nbData*nbSamples))*H(i,:)'; ...
				expmap(uTmp(4:end,:,i)*H(i,:)', model.MuMan(4:end,i))];
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(H(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1:3; out=4:5; outMan=4:6;
nbVarOut = length(out);
nbVarOutMan = length(outMan);

% %Adding artificial distortion and noise on the inputs
% xIn(:,1:nbData) = xIn(:,1:nbData) * 1.2 + randn(1,nbData)*1E-4; 

uhat = zeros(nbVarOut,nbData);
xhat = zeros(nbVarOutMan,nbData);
uOut = zeros(nbVarOut,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbData);

%% Version 1 (with single optimization loop)
for t=1:nbData
	%Compute activation weight
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(xIn(:,t)-model.MuMan(in,i), model.Mu(in,i), model.Sigma(in,in,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	%Compute conditional mean (with covariance transportation)
	if t==1
		[~,id] = max(H(:,t));
		xhat(:,t) = model.MuMan(outMan,id); %Initial point
	else
		xhat(:,t) = xhat(:,t-1);
	end
	for n=1:nbIter
		for i=1:model.nbStates
			%Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t) 
			Ac = transp(model.MuMan(outMan,i), xhat(:,t));
			SigmaTmp(:,:,i) = blkdiag(eye(3),Ac) * model.Sigma(:,:,i) * blkdiag(eye(3),Ac)';
			%Gaussian conditioning on the tangent space
			uOut(:,i,t) = logmap(model.MuMan(outMan,i), xhat(:,t)) + SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * (xIn(:,t)-model.MuMan(in,i)); 
		end
		uhat(:,t) = uOut(:,:,t) * H(:,t);
		xhat(:,t) = expmap(uhat(:,t), xhat(:,t));
	end
	
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
Gdisp = zeros(nbVarOutMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(4:5,4:5,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(4:6,i));
end
Gregr = zeros(nbVarOutMan, nbDrawingSeg, nbData);
for t=1:nbData
	[V,D] = eig(expSigma(:,:,t));
	Gregr(:,:,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], xhat(:,t));
end

%Manifold plot
figure('PaperPosition',[0 0 8 8],'position',[10,10,1350,650],'name','manifold data'); rotate3d on; 
colormap([.9 .9 .9]);
%in
subplot(1,2,1); hold on; axis off; grid off;
plot3(x(1,:), x(2,:), x(3,:), '.','markersize',6,'color',[.4 .4 .4]);
plot3(xIn(1,1:nbData), xIn(2,1:nbData), xIn(3,1:nbData), '-','linewidth',3,'color',[0 0 0]);
for i=1:model.nbStates
	%plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',20,'color',clrmap(i,:));
	plotGMM3D(model.MuMan(1:3,i), model.Sigma(1:3,1:3,i), clrmap(i,:), .1);
end
view(-20,2); axis equal; axis vis3d; 
%out
subplot(1,2,2); hold on; axis off; grid off;
mesh(X,Y,Z);
plot3(x(4,:), x(5,:), x(6,:), '.','markersize',6,'color',[.4 .4 .4]);
for t=1:nbData
	plot3(Gregr(1,:,t), Gregr(2,:,t), Gregr(3,:,t), '-','linewidth',.5,'color',[1 .6 .6]);
end
plot3(xhat(1,:), xhat(2,:), xhat(3,:), '-','linewidth',2,'color',[.8 0 0]);
%plot3(xhat(1,:), xhat(2,:), xhat(3,:), '.','markersize',12,'color',[.8 0 0]);
for i=1:model.nbStates
	plot3(model.MuMan(4,i), model.MuMan(5,i), model.MuMan(6,i), '.','markersize',20,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',3,'color',clrmap(i,:));
end
view(-20,2); axis equal; axis vis3d;  
%print('-dpng','graphs/demo_Riemannian_S2_GMR03a.png');

% %Tangent plane plot
% figure('PaperPosition',[0 0 6 8],'position',[670,10,650,650],'name','tangent space data'); 
% for i=1:model.nbStates
% 	subplot(ceil(model.nbStates/2),2,i); hold on; axis off; title(['k=' num2str(i) ', output space']);
% 	plot(0,0,'+','markersize',40,'linewidth',1,'color',[.7 .7 .7]);
% 	plot(uTmp(4,:,i), uTmp(5,:,i), '.','markersize',4,'color',[0 0 0]);
% 	plotGMM(model.Mu(4:5,i), model.Sigma(4:5,4:5,i)*3, clrmap(i,:), .3);
% 	axis equal;
% end
% %print('-dpng','graphs/demo_Riemannian_S2_GMR03b.png');

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