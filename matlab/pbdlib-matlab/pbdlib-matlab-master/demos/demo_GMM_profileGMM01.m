function demo_GMM_profileGMM01
% Univariate velocity profile fitting with a Gaussian mixture model (GMM) and a weighted EM algorithm.
% The approach shares links with radial basis function networks (RBFN), but adapts the Gaussian basis functions with EM 
% based on the distribution profiles instead of considering a simple clustering of the input space.
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
model.nbStates = 40; %Number of states in the GMM
model.nbVar = 1; %Number of variables [x1]
nbData = 1000; %Length of each trajectory


%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/velprofile01.mat');
Data = spline(1:size(Data,2),Data,linspace(1,size(Data,2),nbData)); %Resample data
tlist = Data(1,:) + 1E-8; %The first value should be non-zero
w = Data(2,:);
Data = tlist;

%Rescale data and make it positive to represent a probability density function
w = w - min(w);
w = w / max(w);


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(Data, model);
model = EM_weighted_univariateGMM(Data, w, model);

%Recompute Priors by regression as in RBFN (The resulting priors will represent weights and do not sum to one)
for i=1:model.nbStates
	Phi(:,i) = gaussPDF(Data, model.Mu(:,i), model.Sigma(:,:,i));
end
model.Priors = (Phi' * Phi) \ Phi' * w';

%Probability density function of normal distributions
for i=1:model.nbStates
	h(i,:) = model.Priors(i) * gaussPDF(tlist, model.Mu(:,i), model.Sigma(:,:,i));
end
hmix = sum(h,1);

disp(['Error: ' num2str(norm(hmix-w))]);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2300,500]); hold on; box on; 
%[~,hmix] = plotGMM1D(model, [.8 0 0], [tlist(1) 0 tlist(end) 1], .8, nbData);
hf(1) = plot(tlist, w, '-','linewidth',2,'color',[.7 .7 .7]);
for i=1:model.nbStates
	plot(tlist, h(i,:),'-','linewidth',2,'color',[1 .7 .7]);
end
hf(2) = plot(tlist, hmix,'-','linewidth',2,'color',[.8 0 0]);
axis([tlist(1) tlist(end) 0 1.05]); set(gca,'Xtick',[]); set(gca,'Ytick',[]);
xlabel('$t$','fontsize',18,'interpreter','latex'); 
ylabel('$|\dot{x}|$','fontsize',18,'interpreter','latex');
legend(hf, {'Reference','Reconstructed'});
% print('-dpng','graphs/demo_GMM_profileGMM01.png');

pause;
close all;