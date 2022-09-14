function demo_GMM_profileGMM_multivariate01
% Multivariate velocity profile fitting with a Gaussian mixture model (GMM) and a weighted EM algorithm
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
model.nbStates = 8; %Number of states in the GMM
model.nbVar = 2; %Number of variables [x1]
nbData = 200; %Length of each trajectory


%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = repmat(1:nbData,model.nbVar,1);
load('data/lognormal05.mat'); %load 'w'


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(Data, model);
model.Priors = repmat(model.Priors,model.nbVar,1);
model = EM_weighted_multivariateGMM(Data, w, model);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,2000,900]); 
for k=1:model.nbVar
	subplot(model.nbVar,1,k); hold on; box on; 
	mtmp.nbStates = model.nbStates; 
	mtmp.Priors = model.Priors(k,:);
	mtmp.Mu(1,:) = model.Mu(k,:);
	mtmp.Sigma(1,1,:) = model.Sigma(k,k,:);
	hf(1) = plot(1:nbData, w(k,:), '-','linewidth',2,'color',[.7 .7 .7]);
	[~,hmix(k,:)] = plotGMM1D(mtmp, [1 nbData 0 1], [.8 0 0], .2);
	hf(2) = plot(1:nbData, hmix(k,:),'-','linewidth',2,'color',[.8 0 0]);
	axis([1 nbData 0 1.05]); set(gca,'Xtick',[]); set(gca,'Ytick',[]);
	xlabel('$t$','fontsize',18,'interpreter','latex'); 
	ylabel(['$\dot{x}_' num2str(k) '$'],'fontsize',18,'interpreter','latex');
end
legend(hf, {'Reference','Reconstructed'});
%print('-dpng','graphs/demo_GMM_profileGMM_multivariate01.png');

disp(['Error: ' num2str(norm(hmix-w))]);

pause;
close all;