function demo_Bezier_illustr01
% Fitting Bezier curves of different degrees
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
% nbVar = 2; %Dimension of datapoint
nbData = 100; %Number of datapoints in a trajectory
t = linspace(0,1,nbData);


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/N.mat');
x = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),nbData)); %Resampling

figure('position',[10,10,1300,800]); 


%% Control points extraction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lst = [1,2,3,7];
for nb=1:length(lst) 
nbDeg = lst(nb); %Degree of the Bezier curve
	
B = zeros(nbDeg,nbData);
for i=0:nbDeg
	B(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-t).^(nbDeg-i) .* t.^i; %Bernstein basis functions
end

P = x / B; %Control points
x2 = P * B; %Reconstruction


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subplot(2,length(lst),nb); hold on; axis off;
plot(x(1,:), x(2,:), '-','linewidth',2,'color',[.7 .7 .7]);
plot(x2(1,:), x2(2,:), '-','linewidth',2,'color',[0 0 0]);
% plot(P(1,:), P(2,:), 'r.','markersize',20);
axis([min(x(1,:))-3, max(x(1,:))+3, min(x(2,:))-3, max(x(2,:))+3]);
subplot(2,length(lst),length(lst)+nb); hold on;
for i=0:nbDeg
	plot(t,B(i+1,:),'linewidth',2);
end
set(gca,'xtick',[],'ytick',[]);
xlabel('t','fontsize',20); ylabel('b_i','fontsize',20);

end %nb

% print('-dpng','graphs/demo_Bezier_illustr01.png');
pause;
close all;