function demo_Riemannian_Gdp_vecTransp01
% Parallel transport on the Grassmann manifold G(3,2)
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
% Written by Noémie Jaquier and Sylvain Calinon
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

addpath('./m_fcts');


%% Generate data on Grassmann
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = 3;
p = 2;

[X1, ~] = qr(randn(n, p), 0);
[X2, ~] = qr(randn(n, p), 0);
[X3, ~] = qr(randn(n, p), 0);
X4 = [X1(:,1) X1(:,1)+X1(:,2)];


%% Representation of Grassmann manifold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% One point of the manifold is the subspace spanned by the column of X such
% that X'X = I, X in R^(3x2). Thus, one point on the manifold is not
% uniquely represented by the matrix X, but can be represented by the
% equivalent class [X].
% In this case, X1 and X4 are not identical, but span the same subspace, so
% that they represent the same point on the manifold.
c1 = [10,-10,-10,10];
c2 = [10,10,-10,-10];

spanX1 = X1(:,1)*c1 + X1(:,2)*c2;
spanX2 = X2(:,1)*c1 + X2(:,2)*c2;
spanX3 = X3(:,1)*c1 + X3(:,2)*c2;
spanX4 = X4(:,1)*c1 + X4(:,2)*c2;

figure('PaperPosition',[0 0 8 8],'position',[10,10,850,850],'Name','Representation of Grassmann manifold'); hold on; axis off; rotate3d on;
patch(spanX1(1,:),spanX1(2,:),spanX1(3,:),[1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3);
patch(spanX2(1,:),spanX2(2,:),spanX2(3,:),[0 1 0],'edgecolor',[0 1 0],'facealpha',.3,'edgealpha',.3);
patch(spanX3(1,:),spanX3(2,:),spanX3(3,:),[0 0 1],'edgecolor',[0 0 1],'facealpha',.3,'edgealpha',.3);
% Check that span(X1) = span(X4)
patch(spanX4(1,:),spanX4(2,:),spanX4(3,:),[1 0 1],'edgecolor',[1 0 1],'facealpha',.3,'edgealpha',.3);
axis equal; axis vis3d; view(30,12);
legend('span(X_1)','span(X_2)','span(X_3)','span(X_4)');


%% Tangent space
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The tangent space of X is composed of all matrices Delta such that
% X'Delta = 0. It corresponds to directions free of rotations mixing the
% basis given by the column of X.
x12 = logmap(X2,X1);
x13 = logmap(X3,X1);
X2e = expmap(x12,X1);
X3e = expmap(x13,X1);

spanx12 = x12(:,1)*c1 + x12(:,2)*c2;
spanX2e = X2e(:,1)*c1 + X2e(:,2)*c2;
spanx13 = x13(:,1)*c1 + x13(:,2)*c2;
spanX3e = X3e(:,1)*c1 + X3e(:,2)*c2;

figure('PaperPosition',[0 0 8 8],'position',[10,10,850,850],'Name','Tangent space and exponential map'); hold on; axis off; rotate3d on;
patch(spanX1(1,:),spanX1(2,:),spanX1(3,:),[1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3);
patch(spanx12(1,:),spanx12(2,:),spanx12(3,:),[0.8 0.8 0],'edgecolor',[0.8 0.8 0],'facealpha',1,'edgealpha',1);
patch(spanX2(1,:),spanX2(2,:),spanX2(3,:),[0 1 0],'edgecolor',[0 1 0],'facealpha',.3,'edgealpha',.3);
patch(spanX2e(1,:),spanX2e(2,:),spanX2e(3,:),[0 0.8 0.2],'edgecolor',[0 0.8 0.2],'facealpha',.3,'edgealpha',.3);
patch(spanx13(1,:),spanx13(2,:),spanx13(3,:),[0.8 0 0.8],'edgecolor',[0.8 0 0.8],'facealpha',1,'edgealpha',1);
patch(spanX3(1,:),spanX3(2,:),spanX3(3,:),[0 0 1],'edgecolor',[0 0 1],'facealpha',.3,'edgealpha',.3);
patch(spanX3e(1,:),spanX3e(2,:),spanX3e(3,:),[0.2 0 0.8],'edgecolor',[0.2 0 0.8],'facealpha',.3,'edgealpha',.3);
axis equal; axis vis3d; view(30,12);
legend('span(X_1)','span(x_{12})','span(X_2)','span(expmap_{X1}(x_{12}))','span(x_{13})','span(X_3)','span(expmap_{X1}(x_{13}))');


%% Follow the geodesic from S1 to S2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t = 0:0.05:1;
g12 = geodesic(x12,X1,t);
g13 = geodesic(x13,X1,t);

figure('PaperPosition',[0 0 8 8],'position',[10,10,850,850],'Name','Following the geodesic'); hold on; axis off; rotate3d on; 
% patch(spanX1(1,:),spanX1(2,:),spanX1(3,:),[1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3);
for i=1:length(t)
	span = g12(:,1,i)*c1 + g12(:,2,i)*c2;
	patch(span(1,:),span(2,:),span(3,:),[1*(1-t(i)) 1*t(i) 0],'edgecolor',[1*(1-t(i)) 1*t(i) 0],'facealpha',.3,'edgealpha',.3);
end
axis equal; axis vis3d; view(30,12);


%% Parallel transport of x12 from X1 to X3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p13 = transp(X1,X3)*x12;
spanp13 = p13(:,1)*c1 + p13(:,2)*c2;

figure('PaperPosition',[0 0 8 8],'position',[10,10,850,850],'Name','Following the geodesic'); hold on; axis off; rotate3d on; 
patch(spanX1(1,:),spanX1(2,:),spanX1(3,:),[1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3);
patch(spanX3(1,:),spanX3(2,:),spanX3(3,:),[0 0 1],'edgecolor',[0 0 1],'facealpha',.3,'edgealpha',.3);
patch(spanx12(1,:),spanx12(2,:),spanx12(3,:),[0.8 0 0.4],'edgecolor',[0.8 0 0.4],'facealpha',1,'edgealpha',1);
patch(spanp13(1,:),spanp13(2,:),spanp13(3,:),[0 0.8 0.4],'edgecolor',[0 0.8 0.4],'facealpha',1,'edgealpha',1);
axis equal; axis vis3d; view(30,12);
legend('span(X_1)','span(X_3)','span(x_{12})','span(transp(x_{12}))');


%% Parallel transport along the geodesic
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 8],'position',[10,10,850,850],'Name','Following the geodesic'); hold on; axis off; rotate3d on;
patch(spanX1(1,:),spanX1(2,:),spanX1(3,:),[1 0 0],'edgecolor',[1 0 0],'facealpha',.3,'edgealpha',.3);
patch(spanX3(1,:),spanX3(2,:),spanX3(3,:),[0 0 1],'edgecolor',[0 0 1],'facealpha',.3,'edgealpha',.3);
axis equal; axis vis3d; view(30,12); 

for i=1:length(t)
	p13 = transp(X1,X3,t(i))*x12;
	spanp13 = p13(:,1)*c1 + p13(:,2)*c2;
	patch(spanp13(1,:),spanp13(2,:),spanp13(3,:),[0 0.8 0.8],'edgecolor',[0.8*(1-t(i)) 0.8*t(i) 0.8],'facealpha',1,'edgealpha',1);
	% Normalised direction of the geodesic
    dir = logmap(X3,g13(:,:,i));
	innormdir = trace(dir'*dir);
%     innormdir = (trace(g13(:,:,i)^-.5*dir* g13(:,:,i)^-1 *dir*g13(:,:,i)^-.5));
    if innormdir > 1E-5
        dirnorm = dir./sqrt(innormdir);
        % Inner product between the transported vector and the direction of
        % the geodesic
        inprod(i) = trace((g13(:,:,i)'*g13(:,:,i))^-1*dirnorm' * p13);
    end
end


%% Verification of parallel transport operation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The parallel transport from one point to the same point is the identity
% map => x12 and px12 are equal.
x12
px12 = transp(X1,X1)*x12

% If p, r and q are three points on a geodesic, parallel transport from p 
% to q is the same than parallel transport from p to r followed by parallel
% transport from r to q.
p13 = transp(X1,X3)*x12
Xm = geodesic(x13,X1,rand());
p1m3 = transp(Xm,X3)*(transp(X1,Xm)*x12)

% Inner product conserved (metric parallel tranport (Levi-Civita
% connection)
inprod
end


%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Y = expmap(H,X)
	[U,S,V] = svd(H,'econ');
	cosS = diag(cos(diag(S)));
	sinS = diag(sin(diag(S)));
	
	Y = [X*V U]*[cosS; sinS]*V';
end

function H = logmap(Y,X)
	[U,S,V] = svd((eye(size(X,1))-X*X')*Y/(X'*Y),'econ');
	H = U*diag(atan(diag(S)))*V';
end

function pt = transp(X1,X2,t)
% Parallel transported vector v: pt * v
	if nargin == 2
		t = 1;
	end
	H = logmap(X2,X1);
	[U,S,V] = svd(H,'econ');
	cosS = diag(cos(diag(S)*t));
	sinS = diag(sin(diag(S)*t));
	
	pt = [X1*V, U] * [-sinS; cosS] * U' + (eye(size(X1,1))-U*U');
end

function g = geodesic(H,X,t)
	for i=1:length(t)
		[U,S,V] = svd(H,'econ');
		cosS = diag(cos(diag(S)*t(i)));
		sinS = diag(sin(diag(S)*t(i)));
		g(:,:,i) = [X*V, U] * [cosS; sinS] * V';
	end
end