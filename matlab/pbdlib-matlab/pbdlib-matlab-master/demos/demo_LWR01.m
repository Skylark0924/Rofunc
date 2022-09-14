function demo_LWR01
% Locally weighted regression (LWR) with radial basis functions and local polynomial fitting
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
% 	author="Calinon, S.",
% 	title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
% 	booktitle="Mixture Models and Applications",
% 	publisher="Springer",
% 	editor="Bouguila, N. and Fan, W.", 
% 	year="2019",
% 	pages="39--57",
% 	doi="10.1007/978-3-030-23876-6_3"
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

% addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbStates = 8; %Number of radial basis functions 
nbData = 200; %Length of a trajectory
polDeg = 3; %Degree of polynomial 


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),nbData)); %Resampling
t = linspace(0,1,nbData); %Time range


%% LWR with radial basis functions and local polynomail fitting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MuRBF = linspace(t(1), t(end), nbStates); %Set centroids equally spread in time
SigmaRBF = 5E-3; %Set constant shared bandwidth
phi = zeros(nbStates, nbData);
for i=1:nbStates
	tc = t - repmat(MuRBF(:,i),1,nbData); %Centered data
	phi(i,:) = exp(-0.5 .* sum( (SigmaRBF .\ tc) .* tc, 1)); %Eq.(2)
end
phi = phi ./ repmat(sum(phi,1),nbStates,1); %Rescaling (Eq.(3), optional)

%Locally weighted regression 
X = zeros(nbData,polDeg+1);
for d=0:polDeg 
	X(:,d+1) = t'.^d; %Input
end
Y = Data'; %Output
A = zeros(polDeg+1, size(Y,2), nbStates);
for i=1:nbStates
	W = diag(phi(i,:)); %Eq.(4)
	A(:,:,i) = X' * W * X \ X' * W * Y; %Weighted least squares estimate (Eq.(1))
end

%Reconstruction of signal
Yr = zeros(size(Y));
for i=1:nbStates
	W = diag(phi(i,:)); %Eq.(4)
	Yr = Yr + W * X * A(:,:,i); %Eq.(5)
end
r(1).Data = Yr';


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 16 4],'position',[10,10,2300,600],'color',[1 1 1]); 
clrmap = lines(nbStates);

%Spatial plot
subplot(2,4,[1,5]); hold on; axis off;
plot(Data(1,:), Data(2,:), '-','linewidth',3,'color',[.7 .7 .7]);
plot(r(1).Data(1,:), r(1).Data(2,:), '-','linewidth',3,'color',[0 0 0]);
axis equal; axis tight;

%Timeline plot 
subplot(2,4,2:4); hold on; 
[~,id] = max(phi,[],1);
for i=1:nbStates
	plot(t(id==i), X(id==i,:) * A(:,1,i), '-','linewidth',8,'color',clrmap(i,:)); %Local polynomials
end
plot(t, Data(1,:), '-','linewidth',3,'color',[.7 .7 .7]);
plot(t, r(1).Data(1,:), '-','linewidth',3,'color',[0 0 0]);
axis([t(1), t(end), min([Data(1,:) r(1).Data(1,:)])-1, max([Data(1,:) r(1).Data(1,:)])+1]);
xlabel('t','fontsize',16); ylabel('x_1','fontsize',16);

%RBFs
subplot(2,4,6:8); hold on; 
for i=1:nbStates
	patch([t(1), t, t(end)], [0, phi(i,:), 0], clrmap(i,:), 'EdgeColor', clrmap(i,:), 'linewidth',2,'facealpha', .3, 'edgealpha', .3);
end
axis([t(1), t(end), 0, max(phi(:))]);
xlabel('t','fontsize',16); ylabel('\phi_k','fontsize',16);

%print('-dpng','graphs/demo_LWR01.png');
pause;
close all;