function demo_Riemannian_S2_TPGMM02
% TP-GMM for data on a sphere by relying on Riemannian manifold (with two coordinate systems).
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
nbSamples = 4; %Number of demonstrations
nbRepros = 3; %Number of reproductions
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 10; %Number of iteration for the EM algorithm
nbDrawingSeg = 50; %Number of segments used to draw ellipsoids

model.nbStates = 5; %Number of states in the GMM
model.nbFrames = 2; %Number of candidate frames of reference
model.nbVar = 3; %Dimension of the tangent space (incl. time)
model.nbVarMan = 4; %Dimension of the manifold (incl. time)
model.dt = 0.01; %Time step duration
model.params_diagRegFact = 1E-4; %Regularization of covariance
e0 = [0; 0; 1]; %Origin on the manifold


%% Generate data from TP-GMM example
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The MAT file contains a structure 's' with the multiple demonstrations. 's(n).Data' is a matrix data for
% sample n (with 's(n).nbData' datapoints). 's(n).p(m).b' and 's(n).p(m).A' contain the position and
% orientation of the m-th candidate coordinate system for this demonstration. 'Data' contains the observations
% in the different frames. It is a 3rd order tensor of dimension D x P x N, with D=2 the dimension of a
% datapoint, P=2 the number of candidate frames, and N=TM the number of datapoints in a trajectory (T=200)
% multiplied by the number of demonstrations (M=5).
load('data/Data01.mat');
sc = 1.5E-1;
nbD = 200;
uIn=[]; uOut=[];
for n=1:nbSamples
	Dtmp = [];
	for m=1:model.nbFrames
		Dtmp = [Dtmp; squeeze(Data(:,m,(n-1)*nbD+1:n*nbD))*2]; 
	end
	s(n).Data =  spline(1:nbD, Dtmp, linspace(1,nbD,nbData));
	uOut = [uOut, s(n).Data*sc];
	uIn = [uIn, [1:nbData]*model.dt];
	for m=1:model.nbFrames
		s(n).p(m).b = expmap(s(n).p(m).b, e0); 
		for v=1:2
			s(n).p(m).A(:,v) = s(n).p(m).A(:,v) / norm(s(n).p(m).A(:,v));
		end
% 		s(n).p(m).b = (rand(model.nbVarMan-1,1)-0.5)*0.3;
% 		s(n).p(m).b = s(n).p(m).b / norm(s(n).p(m).b);
% 		s(n).p(m).A = eye(model.nbVar-1);
% 		[s(n).p(m).A,~] = qr(randn(model.nbVar-1));
	end
end
xOut = [expmap(uOut(1:2,:), e0); expmap(uOut(3:4,:), e0)];
xIn = uIn;
u = [uIn; uOut];
x = [xIn; xOut];


%% GMM parameters estimation (joint distribution with time as input, sphere as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning...');
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:3,:), e0); expmap(model.Mu(4:5,:), e0)]; %Center on the manifold
model.Mu = zeros(1+(model.nbVar-1)*2,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

%Learning of task-parameterized model as a meta Gaussian mixture model (see also demo_TPmetaGMM02.m)
uTmp = zeros(1+(model.nbVar-1)*2,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		xcTmp = [xIn-model.MuMan(1,i); logmap(xOut(1:3,:), model.MuMan(2:4,i)); logmap(xOut(4:6,:), model.MuMan(5:7,i))];
		L(i,:) = model.Priors(i) * gaussPDF(xcTmp, model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [xIn-model.MuMan(1,i); logmap(xOut(1:3,:), model.MuMan(2:4,i)); logmap(xOut(4:6,:), model.MuMan(5:7,i))];
			model.MuMan(:,i) = [(model.MuMan(1,i)+uTmp(1,:,i))*GAMMA2(i,:)'; expmap(uTmp(2:3,:,i)*GAMMA2(i,:)', model.MuMan(2:4,i)); ...
				 expmap(uTmp(4:5,:,i)*GAMMA2(i,:)', model.MuMan(5:7,i))];
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(GAMMA2(i,:)) * uTmp(:,:,i)' + eye(size(uTmp,1)) * model.params_diagRegFact;
	end
end

%Reformatting as a tensor GMM
MuManOld = model.MuMan;
SigmaOld = model.Sigma;
model.Mu = zeros(model.nbVar,model.nbFrames,model.nbStates); %Center in the tangent plane at point MuMan of the manifold
model.MuMan = [];
model.Sigma = [];
for i=1:model.nbStates
	for m=1:model.nbFrames
		id = [1,[1:2]+(m-1)*2+1];
		idMan = [1,[1:3]+(m-1)*3+1];
		model.MuMan(:,m,i) = MuManOld(idMan,i);
		model.Sigma(:,:,m,i) = SigmaOld(id,id,i);
	end
end


%% GMR in each frame (expressed in e0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('GMR in each frame...');
in=1; out=2:3; outMan=2:4;
nbVarOut = length(out);
nbVarOutMan = length(outMan);

uhat = zeros(nbVarOut,model.nbFrames,nbData);
xhat = zeros(nbVarOutMan,model.nbFrames,nbData);
umOut = zeros(nbVarOut,model.nbFrames,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbFrames,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,model.nbFrames,nbData);

%VERSION 1 (with single optimization loop)
for t=1:nbData
	%Compute activation weight (common input shared by all the frames)
	for i=1:model.nbStates
		H(i,t) = model.Priors(i) * gaussPDF(xIn(:,t)-model.MuMan(in,1,i), model.Mu(in,1,i), model.Sigma(in,in,1,i));
	end
	H(:,t) = H(:,t) / sum(H(:,t)+realmin);
	
	%Compute conditional mean (with covariance transportation)
	for m=1:model.nbFrames
		if t==1
			[~,id] = max(H(:,t));
			xhat(:,m,t) = model.MuMan(outMan,m,id); %Initial point
		else
			xhat(:,m,t) = xhat(:,m,t-1);
		end
		for k=1:nbIter
			for i=1:model.nbStates
				%Transportation of covariance from model.MuMan(outMan,m,i) to xhat(:,m,t) 
				Ac = transp(model.MuMan(outMan,m,i), xhat(:,m,t));
				SigmaTmp(:,:,m,i) = blkdiag(1,Ac) * model.Sigma(:,:,m,i) * blkdiag(1,Ac)'; %First variable in Euclidean space
				%Gaussian conditioning on the tangent space
				umOut(:,m,i,t) = logmap(model.MuMan(outMan,m,i), xhat(:,m,t)) + SigmaTmp(out,in,m,i)/SigmaTmp(in,in,m,i) * (xIn(:,t)-model.MuMan(in,m,i));
			end %i
			uhat(:,m,t) = squeeze(umOut(:,m,:,t)) * H(:,t);
			xhat(:,m,t) = expmap(uhat(:,m,t), xhat(:,m,t));
		end %k
		
		%Compute conditional covariances (by ignoring influence of centers uOut(:,i,t))
		for i=1:model.nbStates
			expSigma(:,:,m,t) = expSigma(:,:,m,t) + H(i,t) * (SigmaTmp(out,out,m,i) - SigmaTmp(out,in,m,i)/SigmaTmp(in,in,m,i) * SigmaTmp(in,out,m,i));
			%expSigma(:,:,m,t) = expSigma(:,:,m,t) + eye(nbVarOut) * model.params_diagRegFact;
		end %i
		
		%Compute conditional covariances (note that since uhat=0, the final part in the GMR computation is dropped)
		for i=1:model.nbStates
			SigmaOutTmp = SigmaTmp(out,out,m,i) - SigmaTmp(out,in,m,i)/SigmaTmp(in,in,m,i) * SigmaTmp(in,out,m,i);
			expSigma(:,:,m,t) = expSigma(:,:,m,t) + H(i,t) * (SigmaOutTmp + umOut(:,m,i,t) * umOut(:,m,i,t)');
		end %i
	end %m
end %t


%% Transportation of GMR results from e0 to s(n).p(m).b 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Transportation...');
for n=1:nbRepros
	s(n).x = zeros(nbVarOutMan,model.nbFrames,nbData*nbSamples);
	s(n).xhat = zeros(nbVarOutMan,model.nbFrames,nbData);
	s(n).expSigma0 = zeros(nbVarOut,nbVarOut,model.nbFrames,nbData);
	s(n).expSigma = zeros(nbVarOut,nbVarOut,model.nbFrames,nbData); %Transportation?
	for m=1:model.nbFrames
		s(n).x(:,m,:) = expmap(s(n).p(m).A * uOut([1:2]+(m-1)*2,:), s(n).p(m).b);
		uTmp = s(n).p(m).A * logmap(squeeze(xhat(:,m,:)), e0);
		s(n).xhat(:,m,:) = expmap(uTmp, s(n).p(m).b);
		for t=1:nbData
			%Approximated version with expSigma defined in the tangent space of s(n).p(m).b
			s(n).expSigma0(:,:,m,t) = s(n).p(m).A * expSigma(:,:,m,t) * s(n).p(m).A';
			%Correct version with transportation
			Ac = transp(s(n).p(m).b, s(n).xhat(:,m,t));
			s(n).expSigma(:,:,m,t) = Ac * s(n).p(m).A * expSigma(:,:,m,t) * s(n).p(m).A' * Ac';
		end
	end
end


%% Products of linearly transformed Gaussians
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Products of linearly transformed Gaussians...');
for n=1:nbRepros
	for t=1:nbData
		MuTmp = zeros(nbVarOut,model.nbStates);
		SigmaTmp = zeros(nbVarOut,model.nbVar-1,model.nbStates);
		s(n).xh(:,t) = s(n).xhat(:,1,t);
		for k=1:nbIter
			Mu = zeros(nbVarOut,1);
			SigmaSum = zeros(model.nbVar-1);
			for m=1:model.nbFrames
				%Transportation of covariance from s(n).xhat(:,m,t) to s(n).xh(:,t) 
				Ac = transp(s(n).xhat(:,m,t), s(n).xh(:,t));
				SigmaTmp(:,:,i) = Ac * s(n).expSigma(:,:,m,t) * Ac';
				%Tracking component for Gaussian i
				SigmaSum = SigmaSum + inv(SigmaTmp(:,:,i));
				MuTmp(:,i) = logmap(s(n).xhat(:,m,t), s(n).xh(:,t));
				Mu = Mu + SigmaTmp(:,:,i) \ MuTmp(:,i);
			end
			Sigma = inv(SigmaSum); % + eye(nbVarOut)* model.params_diagRegFact;
			Mu = Sigma * Mu; %Gradient computation
			s(n).xh(:,t) = expmap(Mu, s(n).xh(:,t)); %Update MuMan
			s(n).xhSigma(:,:,t) = Sigma;
		end
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Plotting...');
%clrmap = lines(model.nbFrames);
clrmap = [0.2863 0.0392 0.2392; 0.9137 0.4980 0.0078; 0.7412 0.0824 0.3137];

%Display of covariance contours on the sphere
tl = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVarMan-1, nbDrawingSeg, model.nbFrames, model.nbStates);
for i=1:model.nbStates
	for m=1:model.nbFrames
		[V,D] = eig(model.Sigma(2:3,2:3,m,i));
		Gdisp(:,:,m,i) = expmap(V*D.^.5*[cos(tl); sin(tl)], model.MuMan(2:4,m,i));
	end
end
Gregr = zeros(nbVarOutMan, nbDrawingSeg, model.nbFrames, nbData);
for m=1:model.nbFrames
	for t=1:nbData
		[V,D] = eig(expSigma(:,:,m,t));
		Gregr(:,:,m,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], xhat(:,m,t));
	end
end
for n=1:nbRepros
	for m=1:model.nbFrames
		for t=1:nbData
			%Approximated version with expSigma defined in the tangent space of s(n).p(m).b
			[V,D] = eig(s(n).expSigma0(:,:,m,t));
			utmp = logmap(s(n).xhat(:,m,t), s(n).p(m).b);
			s(n).Gdisp0(:,:,m,t) = expmap(V*D.^.5*[cos(tl); sin(tl)] + repmat(utmp,1,nbDrawingSeg), s(n).p(m).b);
			%Correct version with transportation
			[V,D] = eig(s(n).expSigma(:,:,m,t));
			s(n).Gdisp(:,:,m,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], s(n).xhat(:,m,t));
		end
	end
end
for n=1:nbRepros
	for t=1:nbData
		[V,D] = eig(s(n).xhSigma(:,:,t));
		s(n).Gh(:,:,t) = expmap(V*D.^.5*[cos(tl); sin(tl)], s(n).xh(:,t));
	end
end

%Manifold plot
figure('position',[10,10,2300,900]); %'PaperPosition',[0 0 18 6],
for n=1:nbRepros
	subplot(1,3,n); hold on; axis off; rotate3d on; 
	colormap([.7 .7 .7]);
	[X,Y,Z] = sphere(20);
	mesh(X,Y,Z);
end

% plot3(xOut(1,:), xOut(2,:), xOut(3,:), '.','markersize',6,'color',[.6 .6 .6]);
% for m=1:model.nbFrames
% 	for i=1:model.nbStates
% 		plot3(model.MuMan(1,m,i), model.MuMan(2,m,i), model.MuMan(3,m,i), '.','markersize',12,'color',clrmap(m,:));
% 		plot3(Gdisp(1,:,m,i), Gdisp(2,:,m,i), Gdisp(3,:,m,i), '-','linewidth',1,'color',clrmap(m,:));
% 	end
% end
% for m=1:model.nbFrames
% 	plot3(squeeze(xhat(1,m,:)), squeeze(xhat(2,m,:)), squeeze(xhat(3,m,:)), '.','markersize',12,'color',clrmap(m,:));
% 	for t=1:nbData
% 		plot3(Gregr(1,:,m,t), Gregr(2,:,m,t), Gregr(3,:,m,t), '-','linewidth',1,'color',clrmap(m,:));
% 	end
% end

% %Plot to test the effect of transportation
% plot3(s(n).p(1).b(1), s(n).p(1).b(2), s(n).p(1).b(3), '+','markersize',12,'color',[0 0 0]);
% for m=1:model.nbFrames
% 	for nb=1:nbSamples
% 		plot3(squeeze(s(n).x(1,m,(nb-1)*nbData+1:nb*nbData)), squeeze(s(n).x(2,m,(nb-1)*nbData+1:nb*nbData)), squeeze(s(n).x(3,m,(nb-1)*nbData+1:nb*nbData)), '-','linewidth',.5,'color',min(clrmap(m,:)*1.5,1));
% 	end
% 	plot3(squeeze(s(n).xhat(1,m,:)), squeeze(s(n).xhat(2,m,:)), squeeze(s(n).xhat(3,m,:)), '.','markersize',12,'color',clrmap(m,:));
% 	for t=1:nbData
% 		h(1) = plot3(s(n).Gdisp0(1,:,m,t), s(n).Gdisp0(2,:,m,t), s(n).Gdisp0(3,:,m,t), '-','linewidth',1,'color',clrmap(m,:));
% 		h(2) = plot3(s(n).Gdisp(1,:,m,t), s(n).Gdisp(2,:,m,t), s(n).Gdisp(3,:,m,t), '-','linewidth',1,'color',clrmap(m,:)*0.4); %Transportation?
% 	end
% end
% legend(h,'Without transportation','With transportation'); %Transportation?
% plot3(s(n).xh(1,:), s(n).xh(2,:), s(n).xh(3,:), '.','markersize',8,'color',[0 .7 0]);
% for t=1:nbData
% 	plot3(s(n).Gh(1,:,t), s(n).Gh(2,:,t), s(n).Gh(3,:,t), '-','linewidth',1,'color',[0 .7 0]);
% end
% view(-20,70); axis equal; axis tight; axis vis3d;  


for m=1:model.nbFrames
	for n=1:nbRepros
		subplot(1,3,n); hold on;
		s(n).p(m).b = s(n).p(m).b .* 1.05;
		plot3(s(n).p(m).b(1), s(n).p(m).b(2), s(n).p(m).b(3), '.','markersize',40,'color',[.8 0 0]);
	% 	for nb=1:nbSamples
	% 		plot3(squeeze(s(n).x(1,m,(nb-1)*nbData+1:nb*nbData)), squeeze(s(n).x(2,m,(nb-1)*nbData+1:nb*nbData)), squeeze(s(n).x(3,m,(nb-1)*nbData+1:nb*nbData)), '-','linewidth',.5,'color',min(clrmap(m,:)*1.5,1));
	% 	end
	% 	plot3(squeeze(s(n).xhat(1,m,:)), squeeze(s(n).xhat(2,m,:)), squeeze(s(n).xhat(3,m,:)), '.','markersize',12,'color',clrmap(m,:));
		for t=2:nbData
			plot3(s(n).Gdisp(1,:,m,t), s(n).Gdisp(2,:,m,t), s(n).Gdisp(3,:,m,t), '-','linewidth',1,'color',clrmap(m,:)); %Transportation?
		end
		view(-20,70);  axis vis3d; axis tight;
	end
% 		print('-dpng','-r300',['graphs/demo_Riemannian_sphere_TPGMM' num2str(m) '.png']);
end

for n=1:nbRepros
	subplot(1,3,n); hold on;
% 	plot3(s(n).xh(1,:), s(n).xh(2,:), s(n).xh(3,:), '.','markersize',8,'color',[0 0 0]);
	plot3(s(n).xh(1,:), s(n).xh(2,:), s(n).xh(3,:), '-','linewidth',3,'color',[0 0 0]);
	for t=2:nbData
		plot3(s(n).Gh(1,:,t), s(n).Gh(2,:,t), s(n).Gh(3,:,t), '-','linewidth',2,'color',[0 0 0]);
	end
end	%n

% print('-dpng','-r300','graphs/demo_Riemannian_S2_TPGMM02.png');
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