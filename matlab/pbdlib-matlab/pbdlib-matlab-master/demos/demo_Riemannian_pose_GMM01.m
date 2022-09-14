function demo_Riemannian_pose_GMM01
% GMM to encode 3D position and orientation as unit quaternion by relying on Riemannian manifold.
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
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Andras Kupcsik and Sylvain Calinon
%
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
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
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 50; %Number of datapoints
nbSamples = 5; %Number of demonstrations
nbIter = 10; %Number of iteration for the Gauss Newton algorithm
nbIterEM = 10; %Number of iteration for the EM algorithm

model.nbStates = 5; %Number of states in the GMM
model.nbVar = 6; %Dimension of the tangent space
model.nbVarMan = 7; %Dimension of the manifold
model.params_diagRegFact = 1E-4; %Regularization of covariance


%% Generate artificial unit quaternion data from handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/S.mat');
u=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	u = [u, s(n).Data([1:end,end],:)*1E-1];
end
u(3, :) = sin(u(3, :)) + randn(1, size(u, 2))*.1;

u = [u; [u(1:2, :); cos(u( 2, :))+ randn(1, size(u, 2))*.1]]; % S shape position and orientation
x = expmap(u, [0; 0; 0; 1; 0; 0; 0]);


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kbins(u, model, nbSamples);
model.MuMan = expmap(model.Mu, [0; 0; 0; 1; 0; 0; 0]); %Center on the manifold
model.Mu(4:6, :) = zeros(3,model.nbStates); %Center in the tangent plane at point MuMan of the manifold

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
		model.Mu(1:3, i) = model.MuMan(1:3, i);
		%Update Sigma
		model.Sigma(:,:,i) = (u(:,:,i) - model.Mu(:, i)) * diag(GAMMA2(i,:)) * (u(:,:,i) - model.Mu(:, i))' + eye(size(u,1)) * model.params_diagRegFact;
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure,
plot3(x(1, :), x(2, :), x(3, :), '.k')
grid on, hold on,
for i=1:model.nbStates
	R = QuatToRotMat(model.MuMan(4:7, i));
	plot3Dframe(R/2, model.MuMan(1:3, i), eye(3), 3);
end

for i = 1:size(x, 2)
	R =  QuatToRotMat(x(4:7, i));
	plot3Dframe(R/10,x(1:3, i), eye(3), 1);
end

plotGMM3D(model.MuMan(1:3, :), model.Sigma(1:3, 1:3, :), [.3;.3;.3]', .3); hold on

xlabel('x'), ylabel('y'), zlabel('z')
axis equal

%print('-dpng','graphs/demo_Riemannian_quat_GMM01.png');
pause;
close all;
end

%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = QuatToRotMat(q)
	w = q(1);
	x = q(2);
	y = q(3);
	z = q(4);
	R = [ 1 - 2 * y * y - 2 * z * z , 2 * x * y - 2 * z * w , 2 * x * z + 2 * y * w ;
		2 * x * y + 2 * z * w , 1 - 2 * x * x - 2 * z * z,  2 * y * z - 2 * x * w ;
		2 * x * z - 2 * y * w , 2 * y * z + 2 * x * w , 1 - 2 * x * x - 2 * y * y ];
end


function x = expmap(u, mu) % tangent (6) to manifold (7)
	x = [	u(1:3, :); Q_expmap(u(4:end, :), mu(4:end)) ];
end

function u = logmap(x, mu) % manifold (7) to tangent (6)
	u = [	x(1:3, :); Q_logmap(x(4:end, :), mu(4:end)) ];
end

function x = Q_expmap(u, mu)
	x = QuatMatrix(mu) * Q_expfct(u);
end

function u = Q_logmap(x, mu)
	if norm(mu-[1;0;0;0])<1e-6
		Q = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
	else
		Q = QuatMatrix(mu);
	end
	u = Q_logfct(Q'*x);
end

function Exp = Q_expfct(u)
	normv = sqrt(u(1,:).^2+u(2,:).^2+u(3,:).^2);
	Exp = real([cos(normv) ; u(1,:).*sin(normv)./normv ; u(2,:).*sin(normv)./normv ; u(3,:).*sin(normv)./normv]);
	Exp(:,normv < 1e-16) = repmat([1;0;0;0],1,sum(normv < 1e-16));
end

function Log = Q_logfct(x)
	scale = acoslog(x(1,:)) ./ sqrt(1-x(1,:).^2);
	scale(isnan(scale)) = 1;
	Log = [x(2,:).*scale; x(3,:).*scale; x(4,:).*scale];
	%Log = x(2:4,:) .* scale;
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
	%Ac = zeros(6,6);
	%Ac(1:3,1:3) = diag(h(1:3));
	Ac = eye(6,6);
	Ac(4:end,4:end) = Q_transp(g(4:end),h(4:end));
end

function Ac = Q_transp(g,h)
	E = [zeros(1,3); eye(3)];
	vm = QuatMatrix(g)' * [Q_logmap(h,g); 0];
	mn = norm(vm);
	uv = vm / (mn+realmin);
	Rpar = eye(4) - sin(mn)*(g*uv') - (1-cos(mn))*(uv*uv');
	Ac = E' * QuatMatrix(h) * Rpar * QuatMatrix(g)' * E; %Transportation operator from g to h
end