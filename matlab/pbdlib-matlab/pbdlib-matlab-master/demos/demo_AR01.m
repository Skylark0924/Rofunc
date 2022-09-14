function demo_AR01
% Multivariate autoregressive (AR) model parameters estimation with least-squares
%
% Sylvain Calinon, 2019
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


addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Dimension of datapoint
nbData = 200; %Number of datapoints in a trajectory
nbHist = 2; %Length of time window
nbRepros = 10; %Number of reproduction attempts
nbData2 = nbData * 2;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/E.mat');
for n=1:nbRepros
	r(n).x = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	r(n).x = [r(n).x(2,:); -r(n).x(1,:)];
end


% %% Data generation
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for n=1:nbRepros
% 	r(n).A = [eye(nbVar), zeros(nbVar, nbVar*(nbHist-1))] + [randn(nbVar).*1E-5, randn(nbVar, nbVar*(nbHist-1)).*1E-3]; %Set random AR model parameters
% 	x = kron(ones(1,nbHist), rand(nbVar,1)-.5); %Matrix containing history for the last nbHist iterations
% 	for t=1:nbData
% 		r(n).x(:,t) = x(:,1) + randn(nbVar,1).*1E-3; %Add noise and log data
% 		x = [r(n).A * x(:), x(:,1:end-1)]; %Update data history matrix
% 	end
% end


% %% Learning (linear regression)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for n=1:nbRepros
% 	Y = r(n).x(:,2:end);
% 	X = [];
% 	for t=1:nbData-1
% % 		xtmp = [r(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(r(n).x(:,1),1,nbHist-t)]; %Without offset
% 		xtmp = [r(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(r(n).x(:,1),1,nbHist-t), ones(nbVar,1)]; %With offset
% 		X = [X, xtmp(:)];
% 	end
% 	r(n).A_est = Y / X; %Least-squares estimate of AR parameters
% 	%Simulations
% 	x = kron(ones(1,nbHist), r(n).x(:,1)); %Matrix containing history for the last nbHist iterations
% 	for t=1:nbData2
% 		r(n).x_est(:,t) = x(:,1); %Log data
% % 		x = [r(n).A_est * x(:), x(:,1:end-1)]; %Update data history matrix (without offset)
% 		x = [r(n).A_est * [x(:); ones(nbVar,1)], x(:,1:end-1)]; %Update data history matrix (with offset)
% 	end
% end


%% Learning (linear regression)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = []; Y = [];
for n=1:nbRepros
	Y = [Y, r(n).x(:,2:end)];
	for t=1:nbData-1
% 		xtmp = [r(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(r(n).x(:,1),1,nbHist-t)]; %Without offset
		xtmp = [r(n).x(:,t:-1:max(t-nbHist+1,1)), repmat(r(n).x(:,1),1,nbHist-t), ones(nbVar,1)]; %With offset
		X = [X, xtmp(:)];
	end
end
A_est = Y / X; %Least-squares estimate of AR parameters
%Simulations
for n=1:nbRepros
	x = kron(ones(1,nbHist), r(n).x(:,1)); %Matrix containing history for the last nbHist iterations
	for t=1:nbData2
		r(n).x_est(:,t) = x(:,1); %Log data
% 		x = [A_est * x(:), x(:,1:end-1)]; %Update data history matrix (without offset)
		x = [A_est * [x(:); ones(nbVar,1)], x(:,1:end-1)]; %Update data history matrix (with offset)
	end
end



%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1600,1200]); hold on; axis off;
for n=1:nbRepros
	h(1) = plot(r(n).x(1,:), r(n).x(2,:), 'k-');
	h(2) = plot(r(n).x_est(1,:), r(n).x_est(2,:), 'r-');
	plot(r(n).x(1,1), r(n).x(2,1), 'k.');
end
plot(0, 0, 'b+');
legend(h,{'Observed data','Regenerated data'});

% print('-dpng','graphs/demo_AR01.png'); 
pause;
close all;