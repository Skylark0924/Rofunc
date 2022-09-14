function demo_Bezier02
% Bezier curves fitting
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
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
nbVar = 2; %Dimension of datapoint
nbDeg = 12; %Degree of the Bezier curve
nbData = 100; %Number of datapoints in a trajectory

t = linspace(0,1,nbData);
phi = zeros(nbData,nbDeg);
for i=0:nbDeg
	phi(:,i+1) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-t).^(nbDeg-i) .* t.^i; %Bernstein basis functions
end


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/A.mat');
x = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),nbData)); %Resampling


%% Control points extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = pinv(phi) * x'; %Control points
x2 = phi * w; %Reconstruction

%Alternative computation through vectorization (useful for extending the approach to proMP representations including trajectory covariance)
Psi = kron(phi, eye(nbVar));
w2 = pinv(Psi) * x(:); %Control points
x2Vec = Psi * w2; %Reconstruction


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,800,1200]); 
subplot(2,1,1); hold on; axis off;
% plot(w(:,1), w(:,2), '.','markersize',30,'color',[.6 .6 .6]);
plot(x(1,:), x(2,:), '-','linewidth',6,'color',[.7 .7 .7]);
plot(x2(:,1), x2(:,2), 'k-','linewidth',2);
plot(x2Vec(1:2:end), x2Vec(2:2:end), 'r--','linewidth',2);
% plot(w(1,:), w(2,:), 'r.','markersize',20);
subplot(2,1,2); hold on;
for i=0:nbDeg
	plot(t, phi(:,i+1), '-','linewidth',3);
end
xlabel('t','fontsize',18); ylabel('b_i','fontsize',18);
set(gca,'xtick',[0,1],'ytick',[0,1],'fontsize',16);

% %Plot covariance
% figure('position',[1640 10 800 800]); hold on; 
% colormap(flipud(gray));
% pcolor(abs(Psi*Psi')); 
% set(gca,'xtick',[1,nbData*nbVar],'ytick',[1,nbData*nbVar]);
% axis square; axis([1 nbData*nbVar 1 nbData*nbVar]); shading flat;

pause;
close all;