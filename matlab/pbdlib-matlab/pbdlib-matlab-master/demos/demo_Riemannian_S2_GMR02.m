function demo_Riemannian_S2_GMR02
% GMR with time as input and spherical data as output by relying on Riemannian manifold. 
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
nbSamples = 4; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 10; %Number of iteration for the EM algorithm
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 4; %Number of states in the GMM
model.nbVar = 3; %Dimension of the tangent space (incl. time)
model.nbVarMan = 4; %Dimension of the manifold (incl. time)
model.dt = 0.01; %Time step duration
model.params_diagRegFact = 1E-3; %Regularization of covariance
%e0 = [0; 0; 1]; %Origin on the manifold


%% Generate output data on a sphere from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
uIn=[]; uOut=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	uOut = [uOut, s(n).Data*1.3E-1];
	uIn = [uIn, [1:nbData]*model.dt];
end
xOut = expmap(uOut, [0; -1; 0]);
%xOut = expmap(uOut, e0);
xIn = uIn;
u = [uIn; uOut];
x = [xIn; xOut];


%% GMM parameters estimation (joint distribution with time as input, sphere as output)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:3,:), [0; -1; 0])]; %Center on the manifold 
%model.MuMan = [model.Mu(1,:); expmap(model.Mu(2:3,:), e0)]; %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

uTmp = zeros(model.nbVar,nbData*nbSamples,model.nbStates);
for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF([xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:4,i))], model.Mu(:,i), model.Sigma(:,:,i));
	end
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	H = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			uTmp(:,:,i) = [xIn-model.MuMan(1,i); logmap(xOut, model.MuMan(2:4,i))];
			model.MuMan(:,i) = [(model.MuMan(1,i)+uTmp(1,:,i))*H(i,:)'; expmap(uTmp(2:3,:,i)*H(i,:)', model.MuMan(2:4,i))];
		end
		%Update Sigma
		model.Sigma(:,:,i) = uTmp(:,:,i) * diag(H(i,:)) * uTmp(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in=1; out=2:3; outMan=2:4;
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
	
% 	%Compute conditional mean (approximative version without covariance transportation)
% 	for i=1:model.nbStates
% 		uOutTmp(:,i,t) = model.Sigma(out,in,i)/model.Sigma(in,in,i) * (xIn(:,t)-model.MuMan(in,i));
% 		xOutTmp(:,i,t) = expmap(uOutTmp(:,i,t), model.MuMan(outMan,i));
% 	end
% 	[~,id] = max(H(:,t));
% 	xhat(:,t) = expmap(uOutTmp(:,id,t), model.MuMan(outMan,id)); %Initial point
% 	%xhat(:,t) = model.MuMan(outMan,id); %Initial point
% 	%xhat(:,t) = randn(nbVarOutMan,1); %Initial point
% 	%xhat(:,t) = xhat(:,t) / norm(xhat(:,t));
% 	%Compute xhat iteratively
% 	for n=1:nbIter
% 		uhat(:,t) = logmap(xOutTmp(:,:,t)*H(:,t), xhat(:,t));
% 		xhat(:,t) = expmap(uhat(:,t), xhat(:,t));
% 	end
	
	%Compute conditional mean (with covariance transportation)
	if t==1
		[~,id] = max(H(:,t));
		xhat(:,t) = model.MuMan(outMan,id); %Initial point
	else
		xhat(:,t) = xhat(:,t-1);
	end
	%xhat(:,t) = randn(nbVarOutMan,1); %Initial point
	%xhat(:,t) = xhat(:,t) / norm(xhat(:,t));
	for n=1:nbIter
		for i=1:model.nbStates
			%Transportation of covariance from model.MuMan(outMan,i) to xhat(:,t) 
			Ac = transp(model.MuMan(outMan,i), xhat(:,t));
			SigmaTmp(:,:,i) = blkdiag(1,Ac) * model.Sigma(:,:,i) *  blkdiag(1,Ac)';
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

% 	%Compute conditional covariances
% 	for i=1:model.nbStates
% 		SigmaOutTmp = SigmaTmp(out,out,i) - SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * SigmaTmp(in,out,i);
% 		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaOutTmp + uOutTmp(:,i,t)*uOutTmp(:,i,t)');
% 	end
% 	expSigma(:,:,t) = expSigma(:,:,t) - uhat(:,t)*uhat(:,t)' + eye(nbVarOut) * model.params_diagRegFact; 

% 	%Compute conditional covariances
% 	for i=1:model.nbStates
% 		SigmaTmp = model.Sigma(out,out,i) - model.Sigma(out,in,i)/model.Sigma(in,in,i) * model.Sigma(in,out,i);
% 		expSigma(:,:,t) = expSigma(:,:,t) + H(i,t) * (SigmaTmp + MuTmp(:,i)*MuTmp(:,i)');
% 	end
% 	expSigma(:,:,t) = expSigma(:,:,t) - expData(:,t)*expData(:,t)' + eye(nbVarOut) * diagRegularizationFactor; 
end


% %% Version 2 (with K+1 optimization loop)
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
% 		xhat2(:,t) = model.MuMan(outMan,id); %Initial point
% 	else
% 		xhat2(:,t) = xhat2(:,t-1);
% 	end
% 	%xhat(:,t) = randn(nbVarOutMan,1); %Initial point
% 	%xhat(:,t) = xhat(:,t) / norm(xhat(:,t));
% 	
% 	%Run K optimizations to find conditional probability estimate for each Gaussian
% 	for i=1:model.nbStates
% 		outTmp(:,i,t) = xhat2(:,t); %Initial point
% 		for n=1:nbIter
% 			%Transportation of covariance from model.MuMan(outMan,i) to outTmp(:,i,t)
% 			Ac = transp(model.MuMan(outMan,i), outTmp(:,i,t));
% 			SigmaTmp(:,:,i) = blkdiag(1,Ac) * model.Sigma(:,:,i) * blkdiag(1,Ac)';
% 			%Gaussian conditioning
% 			uOut(:,i,t) = logmap(model.MuMan(outMan,i), outTmp(:,i,t)) + SigmaTmp(out,in,i)/SigmaTmp(in,in,i) * (xIn(:,t)-model.MuMan(in,i)); 
% 			outTmp(:,i,t) = expmap(uOut(:,i,t), outTmp(:,i,t));
% 		end
% 	end
% 	
% 	%Run weighted average optimization 
% 	for n=1:nbIter
% 		uc = logmap(outTmp(:,:,t), xhat2(:,t));
% 		uhat(:,t) = uc * H(:,t);
% 		xhat2(:,t) = expmap(uhat(:,t), xhat2(:,t));
% 	end
% end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clrmap = lines(model.nbStates);
[X,Y,Z] = sphere(20);

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

%Manifold plot
figure('PaperPosition',[0 0 8 8],'position',[10,10,650,650],'name','manifold data'); hold on; axis off; grid off; rotate3d on; 
colormap([.8 .8 .8]);
mesh(X,Y,Z);
plot3(x(2,:), x(3,:), x(4,:), '.','markersize',6,'color',[.4 .4 .4]);
plot3(xhat(1,:), xhat(2,:), xhat(3,:), '-','linewidth',3,'color',[.8 0 0]);
%plot3(xhat2(1,:), xhat2(2,:), xhat2(3,:), '-','linewidth',2,'color',[0 0 .8]);
for t=1:nbData
	plot3(Gregr(1,:,t), Gregr(2,:,t), Gregr(3,:,t), '-','linewidth',.5,'color',[1 .6 .6]);
end
for i=1:model.nbStates
	plot3(model.MuMan(2,i), model.MuMan(3,i), model.MuMan(4,i), '.','markersize',20,'color',clrmap(i,:));
	plot3(Gdisp(1,:,i), Gdisp(2,:,i), Gdisp(3,:,i), '-','linewidth',3,'color',clrmap(i,:));
end
view(-20,2); axis equal; axis vis3d;  
%print('-dpng','-r300','graphs/demo_Riemannian_S2_GMR_time_01.png');

% %Tangent plane plot
% figure('PaperPosition',[0 0 6 8],'position',[670,10,650,650],'name','tangent space data'); 
% for i=1:model.nbStates
% 	subplot(ceil(model.nbStates/2),2,i); hold on; axis off; title(['k=' num2str(i) ', output space']);
% 	plot(0,0,'+','markersize',40,'linewidth',1,'color',[.7 .7 .7]);
% 	plot(uTmp(2,:,i), uTmp(3,:,i), '.','markersize',4,'color',[0 0 0]);
% 	plotGMM(model.Mu(2:3,i), model.Sigma(2:3,2:3,i)*3, clrmap(i,:), .3);
% 	axis equal;
% end
% %print('-dpng','graphs/demo_Riemannian_S2_GMR02b.png');


% %Additional plots
% figure('position',[670,10,650,650]); 
% subplot(3,1,1); hold on;
% for i=1:model.nbStates
% 	plot(xIn(1:nbData), H(i,1:nbData),'linewidth',3,'color',clrmap(i,:));
% end
% ylabel('h');
% subplot(3,1,2); hold on;
% % for i=1:model.nbStates
% % 	msh = [xIn(1:nbData); squeeze(xOutTmp(1,i,:))'];
% % 	msh = [msh, fliplr(msh)];
% % 	patch(msh(1,:), msh(2,:), clrmap(i,:),'facecolor','none','linewidth',3,'edgecolor',clrmap(i,:),'edgealpha',.3);
% % end
% for i=1:model.nbStates
% 	plotGMM(model.MuMan([1,2],i), model.Sigma([1,2],[1,2],i), clrmap(i,:),.3);
% end
% plot(xIn(1:nbData), xhat(1,:), '-','linewidth',2,'color',[0 0 0]);
% ylabel('x_1'); 
% subplot(3,1,3); hold on;
% % for i=1:model.nbStates
% % 	msh = [xIn(1:nbData); squeeze(xOutTmp(2,i,:))'];
% % 	msh = [msh, fliplr(msh)];
% % 	patch(msh(1,:), msh(2,:), clrmap(i,:),'facecolor','none','linewidth',3,'edgecolor',clrmap(i,:),'edgealpha',.3);
% % end
% for i=1:model.nbStates
% 	plotGMM(model.MuMan([1,3],i), model.Sigma([1,3],[1,3],i), clrmap(i,:),.3);
% end
% plot(xIn(1:nbData), xhat(2,:), '-','linewidth',2,'color',[0 0 0]);
% ylabel('x_2'); xlabel('t');

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