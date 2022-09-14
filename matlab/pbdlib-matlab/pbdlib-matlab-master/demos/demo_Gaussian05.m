function demo_Gaussian05
% Stochastic sampling with multivariate Gaussian distribution
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

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 4; %Dimension of datapoint
nbData = 5000; %Number of datapoints


%% Generate random normally distributed data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = zeros(nbVar,1); %rand(nbVar,1);
x = rand(nbVar,4);
xn = randn(nbVar,nbData);
Sigma = cov(x');

[V,D] = eig(Sigma);
U = V * D.^.5;
x1 = U * xn + repmat(Mu,1,nbData); 
Sigma1 = cov(x1');

% %This decomposition is wrong (used here only for illustrative purpose) 
% U2 = sqrtm(Sigma)
% U2 = V * D.^.5 * V'
% x2 = U2 * xn + repmat(Mu,1,nbData); 
% Sigma2 = cov(x2');

% %This decomposition is wrong (used here only for illustrative purpose) 
% U3 = chol(Sigma);
% x3 = U3 * randn(nbVar,nbData) + repmat(Mu,1,nbData); 
% Sigma3 = cov(x3');


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; axis off; hold on;
% plotGMM(Mu(1:2), Sigma2(1:2,1:2), [0 .8 0], .5);
% plotGMM(Mu(1:2), Sigma3(1:2,1:2), [0 0 .8], .5);
plot(x1(1,:), x1(2,:), '.','markersize',1,'color',[.8 0 0]);
% plot(x2(1,:), x2(2,:), '.','color',[0 .8 0]);
plotGMM(Mu(1:2), Sigma(1:2,1:2), [.5 .5 .5], .5);
plotGMM(Mu(1:2), Sigma1(1:2,1:2), [.8 0 0], .5);
axis equal;

%print('-dpng','graphs/demo_Gaussian05.png');
pause;
close all;