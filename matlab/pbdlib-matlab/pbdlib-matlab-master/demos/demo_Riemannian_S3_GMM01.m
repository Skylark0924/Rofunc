function demo_Riemannian_S3_GMM01
% GMM for orientation data as unit quaternions by relying on Riemannian manifold. 
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
nbIterEM = 10; %Number of iteration for the EM algorithm
nbDrawingSeg = 20; %Number of segments used to draw ellipsoids

model.nbStates = 4; %Number of states in the GMM
model.nbVar = 3; %Dimension of the tangent space
model.nbVarMan = 4; %Dimension of the manifold
model.params_diagRegFact = 1E-4; %Regularization of covariance


%% Generate artificial unit quaternion data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
u=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	u = [u, s(n).Data([1:end,end],:)*2E-2]; 
end
x = expmap(u, [0; 1; 0; 0]);


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu, [0; 1; 0; 0]); %Center on the manifold
model.Mu = zeros(model.nbVar,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

for nb=1:nbIterEM
	%E-step
	L = zeros(model.nbStates,size(x,2));
	for i=1:model.nbStates
		L(i,:) = model.Priors(i) * gaussPDF(logmap(x, model.MuMan(:,i)), model.Mu(:,i), model.Sigma(:,:,i));
    end
   
	GAMMA = L ./ repmat(sum(L,1)+realmin, model.nbStates, 1);
	GAMMA2 = GAMMA ./ repmat(sum(GAMMA,2),1,nbData*nbSamples);
	%M-step
	for i=1:model.nbStates
		%Update Priors
		model.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
		%Update MuMan
		for n=1:nbIter
			u(:,:,i) = logmap(x, model.MuMan(:,i));
			model.MuMan(:,i) = expmap(u(:,:,i)*GAMMA2(i,:)', model.MuMan(:,i));
		end
		%Update Sigma
		model.Sigma(:,:,i) = u(:,:,i) * diag(GAMMA2(i,:)) * u(:,:,i)' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Display of covariance contours on the sphere
t = linspace(-pi, pi, nbDrawingSeg);
Gdisp = zeros(model.nbVarMan, nbDrawingSeg, model.nbStates);
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	Gdisp(:,:,i) = expmap(V*D.^.5*[cos(t); sin(t); zeros(1,nbDrawingSeg)], model.MuMan(:,i));
end

clrmap = lines(model.nbStates);
%Tangent space plot
figure('PaperPosition',[0 0 6 8],'position',[10,10,1300,650]); 
for i=1:model.nbStates
	subplot(ceil(model.nbStates/2),2,i); hold on; axis off; title(['k=' num2str(i)]);
	plot(0,0,'+','markersize',40,'linewidth',1,'color',[.7 .7 .7]);
	%plot(u(1,:,i), u(2,:,i), '.','markersize',6,'color',[0 0 0]);
	for t=1:nbData*nbSamples
		plot(u(1,t,i), u(2,t,i), '.','markersize',6,'color',GAMMA(:,t)'*clrmap); %w(1)*[1 0 0] + w(2)*[0 1 0]
	end
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i)*3, clrmap(i,:), .3);
	axis equal; axis tight;
	
	%Plot contours of Gaussian j in tangent plane of Gaussian i (version 1)
	for j=1:model.nbStates
		if j~=i
			udisp = logmap(Gdisp(:,:,j), model.MuMan(:,i));
			patch(udisp(1,:), udisp(2,:), clrmap(j,:),'lineWidth',1,'EdgeColor',clrmap(j,:)*0.5,'facealpha',.1,'edgealpha',.1);	
		end
	end
		
end
%print('-dpng','graphs/demo_Riemannian_S3_GMM01.png');

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
	u = logfct(Q' * x);
end

function Exp = expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2+u(3,:).^2);
	Exp = real([cos(normv) ; u(1,:) .* sin(normv) ./ normv ; u(2,:) .* sin(normv) ./ normv ; u(3,:) .* sin(normv) ./ normv]);
	Exp(:,normv < 1e-16) = repmat([1;0;0;0], 1, sum(normv < 1e-16));
end

function Log = logfct(x)
% 	scale = acos(x(3,:)) ./ sqrt(1-x(3,:).^2);
	scale = acoslog(x(1,:)) ./ sqrt(1-x(1,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(2,:) * scale; x(3,:) * scale; x(4,:) * scale];
end

function Q = QuatMatrix(q)
	Q = [q(1) -q(2) -q(3) -q(4);
	     q(2)  q(1) -q(4)  q(3);
	     q(3)  q(4)  q(1) -q(2);
	     q(4) -q(3)  q(2)  q(1)];
end				 

%Arcosine redefinition to make sure the distance between antipodal quaternions is zero (2.50 from Dubbelman's Thesis)
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
