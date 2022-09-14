function demo_Gaussian_illustr01
% Illustration of Gaussian conditioning with uncertain inputs.
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
model.nbVar = 2; %Number of variables [x1,x2]


%% Load  data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/faithful.mat');
Data = faithful';
Data(1,:) = Data(1,:)*1E1;


%% Gaussian conditioning with uncertain inputs
%% (see for example section "2.3.1 Conditional Gaussian distributions" in Bishop's book, 
%% or the "Conditional distribution" section on the multivariate normal distribution wikipedia page)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Mu = mean(Data,2);
model.Sigma = cov(Data');

in=1; out=2;
DataIn = 50;
SigmaIn = 1E1;

model.Sigma0 = model.Sigma;
model.Sigma(in,in) = model.Sigma(in,in) + SigmaIn;

MuOut = model.Mu(out) + model.Sigma(out,in) / model.Sigma(in,in) * (DataIn - model.Mu(in));
SigmaOut = model.Sigma(out,out) - model.Sigma(out,in) / model.Sigma(in,in) * model.Sigma(in,out);
slope = model.Sigma(out,in) / model.Sigma(in,in);
slope0 = model.Sigma0(out,in) / model.Sigma0(in,in);


% %% Plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mrg = [10 10];
% limAxes = [min(Data(1,:))-mrg(1) max(Data(1,:))+mrg(1) min(Data(2,:))-mrg(2) max(Data(2,:))+mrg(2)];
% 
% figure('PaperPosition',[0 0 12 9],'position',[10,50,1600,1200]); hold on; %axis off;
% 
% plot(model.Mu(in)+[-50,50], model.Mu(out)+slope0*[-50,50], ':','linewidth',1,'color',[1 .5 .5]);
% plot([model.Mu(1) model.Mu(1)], [limAxes(3) model.Mu(2)], ':','linewidth',1,'color',[1 .5 .5]);
% plot([limAxes(1) model.Mu(1)], [model.Mu(2) model.Mu(2)], ':','linewidth',1,'color',[1 .5 .5]);
% %plot(DataIn,MuOut,'.','markersize',12,'color',[0 0 .8]);
% plot(DataIn,limAxes(3),'.','markersize',12,'color',[0 0 .8]);
% plot([DataIn DataIn], [limAxes(3) MuOut], ':','linewidth',1,'color',[.5 .5 1]);
% plot([limAxes(1) DataIn], [MuOut MuOut], ':','linewidth',1,'color',[.5 .5 1]);
% 
% %Plot joint distribution
% plotGMM(model.Mu, model.Sigma0, [.8 0 0], .4);
% %Plot marginal distribution horizontally
% plotGaussian1D(model.Mu(1), model.Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) 10], [.8 0 0], .4, 'h');
% %Plot marginal distribution vertically
% plotGaussian1D(model.Mu(2), model.Sigma(2,2), [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [.8 0 0], .4, 'v');
% %Plot conditional distribution vertically
% plotGaussian1D(MuOut, SigmaOut, [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [0 0 .8], .4, 'v');
% 
% %text(model.Mu(1)+1, model.Mu(2)-1, '$\mathcal{N}(\boldmath{\mu},\boldmath{\Sigma})$',...
% %	'interpreter','latex','fontsize',18,'color',[.6 0 0]);
% text(model.Mu(1)+1, model.Mu(2)-1, '$\mathcal{N}\left(\left[\begin{array}{c}\boldmath{\mu}^{\scriptscriptstyle{\mathcal{I}}} \\ \boldmath{\mu}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\right], \left[\begin{array}{c}\boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{I}}} \; \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{IO}}} \\ \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{OI}}} \; \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\right] \right)$',...
% 	'interpreter','latex','fontsize',18,'color',[.6 0 0]); 
% text(model.Mu(1), limAxes(3)-2, '$\mathcal{N}(\boldmath{\mu}^{\scriptscriptstyle{\mathcal{I}}}, \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{I}}})$',...
% 	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[.6 0 0]);
% text(limAxes(1)-3.5, model.Mu(2), '$\mathcal{N}(\boldmath{\mu}^{\scriptscriptstyle{\mathcal{O}}}, \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{O}}})$',...
% 	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[.6 0 0]);
% 
% text(DataIn, limAxes(3)-2, '$\boldmath{x}^{\scriptscriptstyle{\mathcal{I}}}$',...
% 	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[0 0 .6]);
% text(limAxes(1)-3.5, MuOut, '$\mathcal{N}(\boldmath{\hat{x}}^{\scriptscriptstyle{\mathcal{O}}}, \boldmath{\hat{\Sigma}}^{\scriptscriptstyle{\mathcal{O}}})$',...
% 	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[0 0 .6]);
% 
% axis(limAxes); 
% set(gca,'Xtick',[]); set(gca,'Ytick',[]);
% %axis equal;
% %plot(Data(1,1:200), Data(2,1:200), 'o','markersize',4,'color',[.5 .5 .5]);
% %xlabel('$\boldmath{x}^{\scriptscriptstyle{\mathcal{I}}}$','fontsize',14,'interpreter','latex');
% %ylabel('$\boldmath{x}^{\scriptscriptstyle{\mathcal{O}}}$','fontsize',14,'interpreter','latex');
% 
% % print('-dpng','-r300','graphs/demo_GaussCond01.png');


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrg = [10 10];
limAxes = [min(Data(1,:))-mrg(1) max(Data(1,:))+mrg(1) min(Data(2,:))-mrg(2) max(Data(2,:))+mrg(2)];

figure('PaperPosition',[0 0 9 5],'position',[10,10,1200,800]); hold on; %axis off;

plot(model.Mu(in)+[-50,50], model.Mu(out)+slope0*[-50,50], '-','linewidth',1,'color',[1 .5 .5]);
plot(model.Mu(in)+[-50,50], model.Mu(out)+slope*[-50,50], '-','linewidth',1,'color',[.3 .8 .3]);
plot([model.Mu(1) model.Mu(1)], [limAxes(3) model.Mu(2)], '-','linewidth',1,'color',[1 .5 .5]);
plot([limAxes(1) model.Mu(1)], [model.Mu(2) model.Mu(2)], '-','linewidth',1,'color',[1 .5 .5]);
plot(DataIn,MuOut,'.','markersize',12,'color',[0 0 .8]);
plot(DataIn,limAxes(3),'.','markersize',12,'color',[0 0 .8]);
plot([DataIn DataIn], [limAxes(3) MuOut], '-','linewidth',1,'color',[.5 .5 1]);
plot([limAxes(1) DataIn], [MuOut MuOut], '-','linewidth',1,'color',[.5 .5 1]);

%Plot joint distribution
plotGMM(model.Mu, model.Sigma0, [.8 0 0], .4);
plotGMM(model.Mu, model.Sigma, [0 .6 0], .4);
%Plot marginal distribution horizontally
plotGaussian1D(model.Mu(1), model.Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) 10], [.8 0 0], .4, 'h');
%Plot marginal distribution vertically
plotGaussian1D(model.Mu(2), model.Sigma(2,2), [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [.8 0 0], .4, 'v');
%Plot input distribution horizontally
plotGaussian1D(DataIn, SigmaIn, [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) 10], [0 0 .8], .4, 'h');
%Plot conditional distribution vertically
plotGaussian1D(MuOut, SigmaOut, [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [0 0 .8], .4, 'v');

% text(model.Mu(1)+1, model.Mu(2)-1, '$\mathcal{N}(\boldmath{\mu},\boldmath{\Sigma})$',...
% 	'interpreter','latex','fontsize',18,'color',[.6 0 0]);
% text(model.Mu(1)+2, model.Mu(2)-2, '$\mathcal{N}\left(\!\left[\!\!\begin{array}{c}\boldmath{\mu}^{\scriptscriptstyle{\mathcal{I}}} \\ \boldmath{\mu}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\!\!\right]\!,\!\left[\!\!\begin{array}{cc}\boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{I}}} \!&\! \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{IO}}} \\ \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{OI}}} \!&\! \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\!\!\right] \!\right)$',...
% 	'interpreter','latex','fontsize',18,'color',[.6 0 0]); 
text(model.Mu(1)+2, model.Mu(2)-2, '$\mathcal{N}\left(\!\left[\!\!\begin{array}{c}\boldmath{\mu}^{\scriptscriptstyle{\mathcal{I}}} \\ \boldmath{\mu}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\!\!\right]\!,\!\left[\!\!\begin{array}{cc}\boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{I}}}\!\!+\!\!\boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{S}}} \!&\! \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{IO}}} \\ \!\!\boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{OI}}} \!&\! \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{O}}}\end{array}\!\!\right] \!\right)$',...
	'interpreter','latex','fontsize',18,'color',[0 .4 0]); 
text(model.Mu(1), limAxes(3)-3.5, '$\mathcal{N}(\boldmath{\mu}^{\scriptscriptstyle{\mathcal{I}}}, \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{I}}})$',...
	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[.6 0 0]);
text(limAxes(1)-4.5, model.Mu(2), '$\mathcal{N}(\boldmath{\mu}^{\scriptscriptstyle{\mathcal{O}}}, \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{O}}})$',...
	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[.6 0 0]);

% text(DataIn, limAxes(3)-3.5, '$\boldmath{x}^{\scriptscriptstyle{\mathcal{I}}}$',...
% 	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[0 0 .6]);
text(DataIn, limAxes(3)-3.5, '$\mathcal{N}(\boldmath{x}^{\scriptscriptstyle{\mathcal{I}}}, \boldmath{\Sigma}^{\scriptscriptstyle{\mathcal{S}}})$',...
	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[0 0 .6]);
text(limAxes(1)-4.5, MuOut, '$\mathcal{N}(\boldmath{\hat{x}}^{\scriptscriptstyle{\mathcal{O}}}, \boldmath{\hat{\Sigma}}^{\scriptscriptstyle{\mathcal{O}}})$',...
	'interpreter','latex','HorizontalAlignment','center','fontsize',18,'color',[0 0 .6]);

axis(limAxes); 
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
%axis equal;
%plot(Data(1,1:200), Data(2,1:200), 'o','markersize',4,'color',[.5 .5 .5]);
%xlabel('$\boldmath{x}^{\scriptscriptstyle{\mathcal{I}}}$','fontsize',14,'interpreter','latex');
%ylabel('$\boldmath{x}^{\scriptscriptstyle{\mathcal{O}}}$','fontsize',14,'interpreter','latex');

% print('-dpng','-r300','graphs/demo_GaussCond02.png');

pause;
close all;