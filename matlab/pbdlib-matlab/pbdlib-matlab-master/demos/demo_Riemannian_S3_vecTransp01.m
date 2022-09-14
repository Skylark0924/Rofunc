function demo_Riemannian_S3_vecTransp01
% Parallel transport for unit quaternions (orientation). 
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
model.nbStates = 2; %Number of states in the GMM
model.nbVar = 3; %Dimension of the tangent space
model.nbVarMan = 4; %Dimension of the manifold


%% Setting GMM parameters manually
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Priors = ones(1,model.nbStates) / model.nbStates;

%model.MuMan = randn(model.nbVarMan,model.nbStates);
model.MuMan(:,1) = [1; -.5; 0; .1];
model.MuMan(:,2) = [0; -1; .5; .2];
for i=1:model.nbStates
	model.MuMan(:,i) = model.MuMan(:,i) / norm(model.MuMan(:,i));
end

model.Mu = zeros(model.nbVar,model.nbStates);

model.Sigma(:,:,1) = diag([2,4,3]) * 5E-2;
model.Sigma(:,:,2) = diag([2,4,3]) * 5E-2;

%Eigendecomposition of Sigma
for i=1:model.nbStates
	[V,D] = eig(model.Sigma(:,:,i));
	U0(:,:,i) = V * D.^.5;
end


%% Transportation of covariance 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
g = model.MuMan(:,1);
h = model.MuMan(:,2);
tl = linspace(0,1,20);
for n=1:20
	t = tl(n);
	
	hist(n).MuMan = expmap(logmap(h,g)*t, g);
	
	Ac = transp(g, hist(n).MuMan);
	hist(n).U = Ac * U0(:,:,1);
	hist(n).Sigma = hist(n).U * hist(n).U';
    
	% Direction of the geodesic
	hist(n).dirG = logmap(h, hist(n).MuMan);
	if norm(hist(n).dirG) > 1E-5
		% Normalise the direction
		hist(n).dirG = hist(n).dirG ./ norm(hist(n).dirG);
		% Compute the inner product with the first eigenvector
		inprod(n) = hist(n).dirG' * hist(n).U(:,1);
	end
end

% %Check that the two vectors below are the same
% p1 = -logmap(g,h);
% p2 = E' * rotM(h) * Rpar * rotM(g)' * E * logmap(h,g);
% norm(p1-p2)
% 
% %Check that the transported eigenvectors remain orthogonal
% hist(end).U(:,1)' * hist(end).U(:,2)

% Check that the inner product between the direction of the geodesic and
% parallel transported vectors (e.g. eigenvectors) is conserved.
inprod

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
% u = 2 * acos(q.s) * q.v/norm(q.v);


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