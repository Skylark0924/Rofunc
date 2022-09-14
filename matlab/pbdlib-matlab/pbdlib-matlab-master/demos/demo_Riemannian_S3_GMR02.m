function demo_Riemannian_S3_GMR02
% GMR with time as input and unit quaternion (orientation) as output by relying on Riemannian manifold. 
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
%nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 5; %Number of states in the GMM
model.nbVar = 4; %Dimension of the tangent space (incl. time)
model.nbVarMan = 5; %Dimension of the manifold (incl. time)
model.dt = 0.01; %Time step duration
model.params_diagRegFact = 1E-4; %Regularization of covariance
%e0 = [1; 0; 0; 0]; %Origin on the manifold


%% Generate artificial unit quaternion as output data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
uIn=[]; uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut, s(n).Data([1:end,end],:)*2E-2];
	uIn = [uIn, [1:nbData]*model.dt];
end
xOut = expmap(uOut, [0; 1; 0; 0]);
%xOut = expmap(uOut, e0);
xIn = uIn;
u = [uIn; uOut];
x = [xIn; xOut];


%% GMM parameters estimation (joint distribution with time as input, unit quaternion as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:end,:), [0; 1; 0; 0])]; %Center on the manifold %Data(1,nbData/2)
%model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:end,:), e0)]; %Center on the manifold %Data(1,nbData/2)
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF([xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:end,i))], model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:end,i))];
			model.MuMan(:,i) = [(model.MuMan(1,i)+uTmp(1,:,i))*GAMMA2(i,:)'; expmap(uTmp(2:end,:,i)*GAMMA2(i,:)', model.MuMan(2:end,i))];
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(GAMMA2(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1; out=2:4; outMan=2:5;
nbVarOut = length(out);
nbVarOutMan = length(outMan);
 
% %Adding artificial distortion and noise on the inputs
% xIn(:,1:nbData) = xIn(:,1:nbData) * 1.2 + randn(1,nbData)*1E-4; 

uhat = zeros(nbVarOut,nbData);
xhat = zeros(nbVarOutMan,nbData);
uOut = zeros(nbVarOut,model.nbStates,nbData);
SigmaTmp = zeros(model.nbVar,model.nbVar,model.nbStates);
expSigma = zeros(nbVarOut,nbVarOut,nbData);

%Version with single optimization loop
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
			SigmaTmp(:,:,i) = blkdiag(1,Ac) * model.Sigma(:,:,i) * blkdiag(1,Ac)';
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

% %Display of covariance contours on the tangent space
% tl = linspace(-pi, pi, nbDrawingSeg);
% Gdisp = zeros(nbVarOutMan, nbDrawingSeg, model.nbStates);
% for i=1:model.nbStates
% 	[V,D] = eig(model.Sigma(2:end,2:end,i));
% 	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(tl); sin(tl); zeros(1,nbDrawingSeg)], model.MuMan(2:end,i));
% end
% Gregr = zeros(nbVarOutMan, nbDrawingSeg, nbData);
% for t=1:nbData
% 	[V,D] = eig(expSigma(:,:,t));
% 	R = real(V*D.^.5);
% 	Gregr(:,:,t) = expmap(R*[cos(tl); sin(tl); zeros(1,nbDrawingSeg)], xhat(:,t));
% end

%Timeline plot
figure('PaperPosition',[0 0 6 8],'position',[10,10,650,650],'name','timeline data'); 
for k=1:4
	subplot(2,2,k); hold on; 
	for n=1:nbSamples
		plot(x(1,(n-1)*nbData+1:n*nbData), x(1+k,(n-1)*nbData+1:n*nbData), '-','color',[.6 .6 .6]);
	end
	plot(x(1,1:nbData), xhat(k,:), '-','linewidth',2,'color',[.8 0 0]);
	xlabel('t'); ylabel(['q_' num2str(k)]);
end

%Tangent space plot
figure('PaperPosition',[0 0 6 8],'position',[670,10,650,650],'name','tangent space data'); 
for i=1:model.nbStates
	subplot(ceil(model.nbStates/2),2,i); hold on; axis off; title(['k=' num2str(i) ', output space']);
	plot(0,0,'+','markersize',40,'linewidth',1,'color',[.7 .7 .7]);
	plot(uTmp(2,:,i), uTmp(3,:,i), '.','markersize',4,'color',[0 0 0]);
	plotGMM(model.Mu(2:3,i), model.Sigma(2:3,2:3,i)*3, clrmap(i,:), .3);
	axis equal;
end
%print('-dpng','graphs/demo_Riemannian_S3_GMR02.png');

pause;
close all;
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = expmap(u, mu)
	x = QuatMatrix(mu) * expfct(u);
end

function u = logmap(x, mu)
	if norm(mu-[1;0;0;0])<1e-6
		Q = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
	else
		Q = QuatMatrix(mu);
	end
	u = logfct(Q'*x);
end

function Exp = expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2+u(3,:).^2);
	Exp = real([cos(normv) ; u(1,:).*sin(normv)./normv ; u(2,:).*sin(normv)./normv ; u(3,:).*sin(normv)./normv]);
	Exp(:,normv < 1e-16) = repmat([1;0;0;0],1,sum(normv < 1e-16));
end

function Log = logfct(x)
% 	scale = acos(x(3,:)) ./ sqrt(1-x(3,:).^2);
	scale = acoslog(x(1,:)) ./ sqrt(1-x(1,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(2,:).*scale; x(3,:).*scale; x(4,:).*scale];
end

function Q = QuatMatrix(q)
	Q = [q(1) -q(2) -q(3) -q(4);
			 q(2)  q(1) -q(4)  q(3);
			 q(3)  q(4)  q(1) -q(2);
			 q(4) -q(3)  q(2)  q(1)];
end				 

% Arcosine re-defitinion to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
function acosx = acoslog(x)
	for n=1:size(x,2)
		% sometimes abs(x) is not exactly 1.0
		if(x(n)>=1.0)
			x(n) = 1.0;
		end
		if(x(n)<=-1.0)
			x(n) = -1.0;
		end
		if(x(n)>=-1.0 && x(n)<0)
			acosx(n) = acos(x(n))-pi;
		else
			acosx(n) = acos(x(n));
		end
	end
end

function Ac = transp(g,h)
	E = [zeros(1,3); eye(3)];
	vm = QuatMatrix(g) * [0; logmap(h,g)];
	mn = norm(vm);
	if mn < 1e-10
		disp('Angle of rotation too small (<1e-10)');
		Ac = eye(3);
		return;
	end
	uv = vm / mn;
	Rpar = eye(4) - sin(mn)*(g*uv') - (1-cos(mn))*(uv*uv');	
	Ac = E' * QuatMatrix(h)' * Rpar * QuatMatrix(g) * E; %Transportation operator from g to h 
end