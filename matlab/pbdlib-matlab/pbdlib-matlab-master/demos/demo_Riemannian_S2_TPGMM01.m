function demo_Riemannian_S2_TPGMM01
% TP-GMM for data on a sphere by relying on Riemannian manifold (with single coordinate system).
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
nbData = 20; %Number of datapoints
nbSamples = 1; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 10; %Number of iteration for the EM algorithm
nbDrawingSeg = 25; %Number of segments used to draw ellipsoids

model.nbStates = 5; %Number of states in the GMM
model.nbFrames = 1; %Number of candidate frames of reference
model.nbVar = 3; %Dimension of the tangent space (incl. time)
model.nbVarMan = 4; %Dimension of the manifold (incl. time)
model.dt = 1E-3; %Time step duration
model.params_diagRegFact = 1E-5; %Regularization of covariance
e0 = [0; 0; 1]; %Origin on the manifold


%% Generate dataset on a sphere from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
uIn=[]; uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut, s(n).Data*1.4E-1];
	uIn = [uIn, [1:nbData]*model.dt];
	for m=1:model.nbFrames
		s(n).p(m).b = (rand(model.nbVarMan-1,1)-0.5)*2;
		s(n).p(m).b = s(n).p(m).b / norm(s(n).p(m).b);
		%s(n).p(m).A = eye(model.nbVar-1);
		[s(n).p(m).A,~] = qr(randn(model.nbVar-1));
	end
end
xOut = expmap(uOut, e0);
xIn = uIn;
u = [uIn; uOut];
x = [xIn; xOut];

% %Generate data from TP-GMM example
% load('data/Data01.mat');
% nbD = 200;
% uIn=[]; uOut=[];
% for n=1:nbSamples
% 	Dtmp = squeeze(Data(:,2,(n-1)*nbD+1:n*nbD));
% 	s(n).Data =  spline(1:nbD, Dtmp, linspace(1,nbD,nbData));
% 	uOut = [uOut, s(n).Data*1.5E-1];
% 	uIn = [uIn, [1:nbData]*model.dt];
% 	for m=1:model.nbFrames
% 		s(n).p(m).b = (rand(model.nbVarMan-1,1)-0.5)*2;
% 		s(n).p(m).b = s(n).p(m).b / norm(s(n).p(m).b);
% 		%s(n).p(m).A = eye(model.nbVar-1);
% 		[s(n).p(m).A,~] = qr(randn(model.nbVar-1));
% 	end
% end
% xOut = expmap(uOut, e0);
% xIn = uIn;
% u = [uIn; uOut];
% x = [xIn; xOut];


%% GMM parameters estimation (joint distribution with time as input, sphere as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:3,:), e0)]; %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF([xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:4,i))], model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:4,i))];
			model.MuMan(:,i) = [(model.MuMan(1,i)+uTmp(1,:,i))*GAMMA2(i,:)'; expmap(uTmp(2:3,:,i)*GAMMA2(i,:)', model.MuMan(2:4,i))];
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(GAMMA2(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR in each frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1; out=2:3; outMan=2:4;
nbVarOut = length(out);
nbVarOutMan = length(outMan);

uhat = zeros(nbVarOut,nbData);
xhat = zeros(nbVarOutMan,nbData);
uOutTmp = zeros(nbVarOut,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbData);

%VERSION 1 (with single optimization loop)
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
			SigmaTmp(:,:,i) = blkdiag(1, Ac) * model.Sigma(:,:,i) * blkdiag(1, Ac)'; %First variable in Euclidean space
			%Gaussian conditioning on the tangent space
			uOutTmp(:,i,t) = logmap(model.MuMan(outMan,i), xhat(:,t)) + SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * (xIn(:,t)-model.MuMan(in,i)); 
		end
		uhat(:,t) = uOutTmp(:,:,t) * H(:,t);
		xhat(:,t) = expmap(uhat(:,t), xhat(:,t));
	end
	
	%Compute conditional covariances
	for i=1:model.nbStates
		SigmaOutTmp = SigmaTmp(out,out,i) - SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * SigmaTmp(in,out,i);
		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaOutTmp + uOutTmp(:,i,t)*uOutTmp(:,i,t)');
	end
	expSigma(:,:,t) = expSigma(:,:,t) - uhat(:,t)*uhat(:,t)' + eye(nbVarOut) * model.params_diagRegFact; 
end


% %VERSION 2 (with K+1 optimization loop)
% for t=1:nbData
% 	%Compute activation weight
% 	for i=1:model.nbStates
% 		H(i,t) = model.Priors(i) * gaussPDF(xIn(:,t)-model.MuMan(in,i), model.Mu(in,i), model.Sigma(in,in,i));
% 	end
% 	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
% 	
% 	%Initialization 
% 	if t==1
% 		[~,id] = max(H(:,t));
% 		xhat(:,t) = model.MuMan(outMan,id); %Initial point
% 	else
% 		xhat(:,t) = xhat(:,t-1);
% 	end
% 	
% 	%Run K optimizations to find conditional probability estimate for each Gaussian
% 	for i=1:model.nbStates
% 		outTmp(:,i,t) = xhat(:,t); %Initial point
% 		for n=1:nbIter
% 			%Transportation of covariance from g to h 
% 			g = model.MuMan(outMan,i);
% 			h = outTmp(:,i,t);
% 			Rg = rotM(g);
% 			Rh = rotM(h);
% 			vm = Rg' * [logmap(h,g); 0];
% 			m = norm(vm);
% 			u = vm/m;
% 			Rpar = eye(nbVarOutMan) - sin(m)*g*u' - (1-cos(m))*u*u';	
% 			U = blkdiag(1, E' * Rh * Rpar * Rg' * E) * U0(:,:,i); %First variable in Euclidean space
% 			SigmaTmp(:,:,i) = U * U';
% 			%Gaussian conditioning
% 			uOutTmp(:,i,t) = logmap(model.MuMan(outMan,i), outTmp(:,i,t)) + SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * (xIn(:,t)-model.MuMan(in,i)); 
% 			outTmp(:,i,t) = expmap(uOutTmp(:,i,t), outTmp(:,i,t));
% 		end
% 	end
% 	
% 	%Run weighted average optimization 
% 	for n=1:nbIter
% 		uc = logmap(outTmp(:,:,t), xhat(:,t));
% 		uhat(:,t) = uc * H(:,t);
% 		xhat(:,t) = expmap(uhat(:,t), xhat(:,t));
% 	end
% end


%% Transportation of GMR results from e0 to s(n).p(1).b 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbSamples
	%uTmp = s(n).A * logmap(model.MuMan, e0);
	%s(n).MuMan = expmap(uTmp, s(n).b);
	
	s(n).x = expmap(s(n).p(1).A * uOut, s(n).p(1).b);	
	
	uTmp = s(n).p(1).A * logmap(xhat, e0);
	s(n).xhat = expmap(uTmp, s(n).p(1).b);
	for t=1:nbData
		%Approximated version with expSigma defined in the tangent space of s(n).p(m).b
		s(n).expSigma0(:,:,t) = s(n).p(1).A * expSigma(:,:,t) * s(n).p(1).A';
		%Correct version with transportation
		Ac = transp(s(n).p(1).b, s(n).xhat(:,t));
		s(n).expSigma(:,:,t) = Ac * s(n).p(1).A * expSigma(:,:,t) * s(n).p(1).A' * Ac';
	end
end

% E = [eye(2); zeros(1,2)];
% for n=1:nbSamples
% 	%Transportation of vectors from g to h
% 	g = e0;
% 	h = s(n).p(1).b;
% 	vm = rotM(g)' * [logmap(h,g); 0];
% 	m = norm(vm);
% 	u = vm/m;
% 	Rpar = eye(model.nbVarMan) - sin(m)*g*u' - (1-cos(m))*u*u';
% 	Ac = s(n).p(1).A * E' * rotM(h) * Rpar * rotM(g)' * E;
% 	%Transportation of data
% 	Utmp = Ac * Data(:,(n-1)*nbData+1:n*nbData);
% 	s(n).x = expmap(Utmp, s(n).b);
% 	%Transportation of Mu
% 	Utmp = Ac * logmap(model.MuMan, e0);
% 	s(n).MuMan = expmap(Utmp, s(n).p(1).b);
% 	%Transportation of Sigma
% 	for i=1:model.nbStates
% 		Utmp = Ac * U0(:,:,i);
% 		s(n).Sigma(:,:,i) = Utmp * Utmp';
% 	end
% end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(nbVarOutMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(2:3,2:3,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(2:4,i));
end
Gregr = zeros(nbVarOutMan, nbDrawingSeg, nbData);
for t=1:nbData
	[V,D] = eig(expSigma(:,:,t));
	Gregr(:,:,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], xhat(:,t));
end
for n=1:nbSamples
	for t=1:nbData
		%Approximated version with expSigma defined in the tangent space of s(n).p(m).b
		[V,D] = eig(s(n).expSigma0(:,:,t));
		utmp = logmap(s(n).xhat(:,t), s(n).p(1).b);
		s(n).Gdisp0(:,:,t) = expmap(V*D.^.5*[cos(tl); sin(tl)] + repmat(utmp,1,nbDrawingSeg), s(n).p(1).b);
		%Correct version with transportation
		[V,D] = eig(s(n).expSigma(:,:,t));
		s(n).Gdisp(:,:,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], s(n).xhat(:,t));
	end
end
	
%Manifold plot
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650]); hold on; axis off; grid off; rotate3d on; 
colormap([.9 .9 .9]);
[X,Y,Z] = sphere(20);
mesh(X,Y,Z);
%plot3(xOut(1,:), xOut(2,:), xOut(3,:), '-','linewidth',1,'color',[1 .6 .6]);
plot3(xhat(1,:), xhat(2,:), xhat(3,:), '.','markersize',8,'color',[.8 0 0]);
for t=1:nbData
	plot3(Gregr(1,:,t), Gregr(2,:,t), Gregr(3,:,t), '-','linewidth',1,'color',[.8 0 0]);
end

for n=1:nbSamples
	plot3(s(n).p(1).b(1), s(n).p(1).b(2), s(n).p(1).b(3), '+','markersize',12,'color',[0 0 0]);
% 	for nb=1:nbSamples
% 		plot3(s(n).x(1,(nb-1)*nbData+1:nb*nbData), s(n).x(2,(nb-1)*nbData+1:nb*nbData), s(n).x(3,(nb-1)*nbData+1:nb*nbData), '-','linewidth',.5,'color',[.6 .6 .6]);
% 	end
	plot3(s(n).xhat(1,:), s(n).xhat(2,:), s(n).xhat(3,:), '.','markersize',8,'color',[0 0 0]);
	for t=1:nbData
		h(1) = plot3(s(n).Gdisp0(1,:,t), s(n).Gdisp0(2,:,t), s(n).Gdisp0(3,:,t), '-','linewidth',1,'color',[.5 .5 .5]);
		h(2) = plot3(s(n).Gdisp(1,:,t), s(n).Gdisp(2,:,t), s(n).Gdisp(3,:,t), '-','linewidth',1,'color',[.2 .2 .2]); %Transportation?
	end
end
legend(h,'Without transportation','With transportation'); %Transportation?


% for i=1:model.nbStates
% 	plot3(model.MuMan(1,i), model.MuMan(2,i), model.MuMan(3,i), '.','markersize',8,'color',clrmap(i,:));
% 	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',1,'color',clrmap(i,:));
% end
view(-20,70); axis equal; axis tight; axis vis3d;  
%print('-dpng','graphs/demo_Riemannian_S2_TPGMM01.png');

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