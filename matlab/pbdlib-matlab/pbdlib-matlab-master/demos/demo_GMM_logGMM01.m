function demo_GMM_logGMM01
% Multivariate lognormal mixture model parameters estimation with EM algorithm.
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
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
model.nbStates = 2; %Number of states in the mixture
model.nbVar = 2; %Number of variables [x1,x2]
nbData = 500; %Number of datapoints


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model.Mu(:,1) = [1; 0.5];
% model.Mu(:,2) = [0.5; 1];
% model.Sigma(:,:,1) = diag([0.8, 0.1]);
% model.Sigma(:,:,2) = diag([0.1, 0.8]);
model.Mu = rand(model.nbVar, model.nbStates);
for i=1:model.nbStates
	V = rand(model.nbVar);
	model.Sigma(:,:,i) = V*V';
end
model.Priors = ones(1,model.nbStates) / model.nbStates;

for t=1:nbData
	%Select state randomly
	[~,i] = min(abs(rand-cumsum(model.Priors)));
	%Generate datapoint from i-th lognormal distribution
	[V,D] = eig(model.Sigma(:,:,i));
	Data(:,t) = exp(V*D^.5 * randn(model.nbVar,1) + model.Mu(:,i));
end


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_kmeans(Data, model);
model = EM_logGMM(Data, model);

nbX = 100;
[X0,Y0] = meshgrid(linspace(min(Data(1,:)),max(Data(1,:)),nbX), linspace(min(Data(2,:)),max(Data(2,:)),nbX));
X = reshape(X0,1,nbX^2);
Y = reshape(Y0,1,nbX^2);
DataIn = [X;Y];

%Probability density function of lognormal distributions
for i=1:model.nbStates
	h(i,:) = model.Priors(i) * logGaussPDF(DataIn, model.Mu(:,i), model.Sigma(:,:,i));
end
hmix = sum(h,1);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 24 4],'position',[10,10,1600,900]); hold on; hold on; axis off;
pcolor(X0,Y0,reshape(hmix,nbX,nbX)); shading flat;
plot(Data(1,:), Data(2,:), '.','markersize',6,'color',[0 0 0]);
axis([min(Data(1,:)), max(Data(1,:)), min(Data(2,:)), max(Data(2,:))]); 

%print('-dpng','graphs/demo_logGMM01.png');
pause;
close all;