function demo_Gaussian04
% Gaussian estimate of a GMM with the law of total covariance
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


%% GMM parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbVar = 1;
model.nbStates = 3;
% model.Priors = rand(1,model.nbStates);
% model.Priors = model.Priors / sum(model.Priors);
model.Priors = ones(1,model.nbStates) / model.nbStates;
model.Mu = rand(model.nbVar,model.nbStates) * 3;
for i=1:model.nbStates
	U = rand(model.nbVar,model.nbStates) * 5E-1;
	model.Sigma(:,:,i) = U*U';
end

% %1D Gaussians
% model.nbVar = 1;
% model.nbStates = 3;
% model.Mu(1,1:3) = [0, 1.5, 3.5];
% model.Sigma(1,1,1:3) = [2, 1, .5];
% model.Priors = [.1, .6, .3];


%% Gaussian estimate of a GMM (law of total variance)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mu = model.Mu * model.Priors';
Sigma = zeros(model.nbVar);
for i=1:model.nbStates
	Sigma = Sigma + model.Priors(i) * (model.Sigma(:,:,i) + model.Mu(:,i)*model.Mu(:,i)');
end
Sigma = Sigma - Mu*Mu';

% %Equivalent computation for 2 Gaussians
% Sigma = model.Priors(1) * model.Sigma(:,:,1) + model.Priors(2) * model.Sigma(:,:,2) + model.Priors(1) * model.Priors(2) * (model.Mu(:,1)-model.Mu(:,2)) * (model.Mu(:,1)-model.Mu(:,2))'


%% Plot 1D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limAxes = [-3 7 0 1];
figure('PaperPosition',[0 0 6 4],'position',[10,10,1600,800]); hold on; %axis off; 5.5 2.475
set(gca,'linewidth',2);
Pt = zeros(1,400);
for i=1:model.nbStates
	Ptmp = plotGaussian1D(model.Mu(1,i), model.Sigma(1,1,i), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) gaussPDF(0,0,model.Sigma(1,1,i))*model.Priors(i)], [0 0 0], .2, 'h');
	Pt = Pt + Ptmp(2,:); 
end
% print('-dpng','graphs/demo_lawTotalCov1D01.png');
plotDistrib1D(Pt, [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) max(Pt)], [0 0 0], 1, 'h');
% print('-dpng','graphs/demo_lawTotalCov1D02.png');
plotGaussian1D(Mu(1), Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) gaussPDF(0,0,Sigma(1,1))], [0 .5 0], .3, 'h');

xlabel('$x^{\scriptscriptstyle{O}}_1$','fontsize',20,'interpreter','latex');
ylabel('$\mathcal{P}(x^{\scriptscriptstyle{O}}_1)$','fontsize',20,'interpreter','latex');
axis([limAxes(1) limAxes(2) 0 max([Pt, gaussPDF(0,0,Sigma(1,1))])]); %axis tight;
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% print('-dpng','graphs/demo_lawTotalCov1D03.png');


% %% Plot 2D
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% limAxes = [-.6 1.6 -.6 1.6];
% figure('PaperPosition',[0 0 4 4],'position',[10,10,1200,1200]); hold on; %axis off;
% set(gca,'linewidth',2);
% plotGMM(model.Mu, model.Sigma,[0 0 0],.2);
% % print('-dpng','graphs/demo_lawTotalCov2D01.png');
% 
% nbGrid = 200;
% [xx,yy] = meshgrid(linspace(limAxes(1),limAxes(2),nbGrid), linspace(limAxes(3),limAxes(4),nbGrid));
% z = zeros(nbGrid^2,1);
% for i=1:model.nbStates
%   z = z + model.Priors(i) * gaussPDF([xx(:)'; yy(:)'], model.Mu(:,i), model.Sigma(:,:,i));
% end
% zz = reshape(z,nbGrid,nbGrid);
% contour(xx,yy,zz,[.4,.4], 'color',[0 0 0],'linestyle',':','linewidth',2);
% % print('-dpng','graphs/demo_lawTotalCov2D02.png');
% 
% plotGMM(Mu, Sigma,[0 .6 0],.3);
% 
% xlabel('$x^{\scriptscriptstyle{O}}_1$','fontsize',20,'interpreter','latex');
% ylabel('$x^{\scriptscriptstyle{O}}_2$','fontsize',20,'interpreter','latex');
% axis equal; %axis(limAxes); 
% set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% 
% % print('-dpng','graphs/demo_lawTotalCov2D03.png');

pause;
close all;