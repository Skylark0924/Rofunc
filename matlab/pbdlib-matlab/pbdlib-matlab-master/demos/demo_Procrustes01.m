function demo_Procrustes01
% SVD solution of orthogonal Procrustes problem.
%
% If this code is useful for your research, please cite the related publication:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
% }
% 
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Number of variables [x1,x2]
nbData = 100; %Length of each trajectory


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[]; Data=[];
load('data/2Dletters/G.mat');
x = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),nbData)); %Resampling
% xa = [x; zeros(1,nbData)];

theta = 1.23;
R = [cos(theta) -sin(theta); sin(theta) cos(theta)]
v = [15.43; 8.5]
% H = [R, v; 0 0 1];

y = R * x + repmat(v, 1, nbData) + randn(nbVar,nbData).*1E-1;
% ya = H * xa + [randn(nbVar,nbData).*0E-5; zeros(1,nbData)];


%% Procrustes (see Eq. (7.6) in Godall'91, or https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xc = x - repmat(mean(x,2),1,nbData);
yc = y - repmat(mean(y,2),1,nbData);
[U,S,V] = svd(xc * yc');
R_est = V * U'
% R_est = (yc * xc' * xc * yc')^-.5 * yc * xc'
v_est = mean(y - R_est * x, 2)

y_est = R_est * x + repmat(v_est, 1, nbData); %Reconstruction

% [U,S,V] = svd(xa*ya');
% H_est = V * U'

%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,700,500]); hold on; axis off;
plot(x(1,:),x(2,:),'.','markersize',15,'color',[0 0 0]);
h(1) = plot(y(1,:),y(2,:),'.','markersize',15,'color',[.8 0 0]);
h(2) = plot(y_est(1,:),y_est(2,:),'.','markersize',15,'color',[.5 .5 .5]);
legend(h,{'Original','Reconstruction'});
axis equal; 

%print('-dpng','graphs/demo_Procrustes01.png');
pause;
close all;