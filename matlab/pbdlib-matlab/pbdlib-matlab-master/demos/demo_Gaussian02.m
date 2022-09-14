function demo_Gaussian02
% Conditional probability with a multivariate normal distribution.
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


%% Gaussian conditioning 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.Mu = mean(Data,2);
model.Sigma = cov(Data');

in=1; out=2;
DataIn = 50;

%See for example section "2.3.1 Conditional Gaussian distributions" in Bishop's book, 
%or Conditional distribution section on the multivariate normal distribution wikipedia page
MuOut = model.Mu(out) + model.Sigma(out,in)/model.Sigma(in,in) * (DataIn-model.Mu(in));
SigmaOut = model.Sigma(out,out) - model.Sigma(out,in)/model.Sigma(in,in) * model.Sigma(in,out);
slope = model.Sigma(out,in)/model.Sigma(in,in);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrg = [10 10];
limAxes = [min(Data(1,:))-mrg(1) max(Data(1,:))+mrg(1) min(Data(2,:))-mrg(2) max(Data(2,:))+mrg(2)];

figure('PaperPosition',[0 0 4 3],'position',[10,10,1200,800]); hold on; %axis off;
plot(DataIn+[-50,50], MuOut+slope*[-50,50], ':','linewidth',1,'color',[.7 .3 .3]);
plot([model.Mu(1) model.Mu(1)], [limAxes(3) model.Mu(2)], ':','linewidth',1,'color',[.7 .3 .3]);
plot([limAxes(1) model.Mu(1)], [model.Mu(2) model.Mu(2)], ':','linewidth',1,'color',[.7 .3 .3]);

%Plot joint distribution
plotGMM(model.Mu, model.Sigma, [.8 0 0], .4);
%Plot marginal distribution horizontally
plotGaussian1D(model.Mu(1), model.Sigma(1,1), [limAxes(1) limAxes(3) limAxes(2)-limAxes(1) 10], [.8 0 0], .4, 'h');
%Plot marginal distribution vertically
plotGaussian1D(model.Mu(2), model.Sigma(2,2), [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [.8 0 0], .4, 'v');

%Plot conditional distribution vertically
plotGaussian1D(MuOut, SigmaOut, [limAxes(1) limAxes(3) 10 limAxes(4)-limAxes(3)], [0 0 .8], .4, 'v');

plot(DataIn,MuOut,'.','markersize',12,'color',[.7 .3 .3]);
plot(DataIn,limAxes(3),'.','markersize',26,'color',[0 0 .8]);
plot([DataIn DataIn], [limAxes(3) MuOut], ':','linewidth',1,'color',[.7 .3 .3]);
plot([limAxes(1) DataIn], [MuOut MuOut], ':','linewidth',1,'color',[.7 .3 .3]);

%plot(Data(1,1:200), Data(2,1:200), 'o','markersize',4,'color',[.5 .5 .5]);
axis(limAxes);
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
%axis equal; 
xlabel('$x_1$','fontsize',16,'interpreter','latex');
ylabel('$x_2$','fontsize',16,'interpreter','latex');

%print('-dpng','-r600','graphs/demo_Gaussian02.png');
pause;
close all;