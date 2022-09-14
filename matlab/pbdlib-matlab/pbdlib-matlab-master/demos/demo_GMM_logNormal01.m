function demo_GMM_logNormal01
% Conditional probability with multivariate log-normal distribution
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
nbData = 1000; %Length of each trajectory
model0.nbVar = 2;
model0.nbStates = 1;
model0.Priors = 1;
model = model0;

	
%% Multivariate normal distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model0.Mu = zeros(model0.nbVar,1);
%model0.Mu = rand(model0.nbVar,1)*10;

V0 = rand(model0.nbVar,1);
model0.Sigma = V0*V0' + diag(rand(model0.nbVar,1))*1E-1;
[V,D] = eigs(model0.Sigma);
Data0 = V*D^.5 * randn(model0.nbVar,nbData) + repmat(model0.Mu,1,nbData);


%% Multivariate log-normal distribution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = exp(Data0);
% for i=1:model.nbVar
% 	model.Mu(i,1) = exp(model0.Mu(i) + 0.5 * model0.Sigma(i,i));
% 	for j=1:model.nbVar
% 		model.Sigma(i,j) = exp(model0.Mu(i)+model0.Mu(j)+0.5*(model0.Sigma(i,i)+model0.Sigma(j,j))) * (exp(model0.Sigma(i,j))-1);
% 	end
% end
model.Mu = exp(model0.Mu);
model.Sigma = exp(model0.Sigma);


%% Gaussian conditioning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DataIn0 = linspace(min(Data0(1,:)),max(Data0(1,:)),100);
DataOut0 = GMR(model0, DataIn0, 1, 2);

% DataIn = linspace(min(Data(1,:)),max(Data(1,:)),100);
% DataOut = GMR(model0, DataIn, 1, 2);
DataIn = exp(DataIn0);
DataOut = exp(DataOut0);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 2300 1200],'color',[1 1 1]);

%Normal distribution
subplot(1,2,1); hold on; grid on; title('Normal distribution');
plot(Data0(1,:), Data0(2,:), '.','markersize',8,'color',[.7 .7 .7]);
plot(DataIn0, DataOut0, '-','linewidth',2,'color',[.8 0 0]);
plotGMM(model0.Mu, model0.Sigma*2, [.8 0 0], .3);
axis equal; xlabel('x_1'); ylabel('x_2');
set(gca,'xtick',model0.Mu(1),'ytick',model0.Mu(2));

% nbDrawingSeg = 35;
% t = linspace(-pi, pi, nbDrawingSeg);
% [V,D] = eig(model0.Sigma*2);
% R = real(V*D.^.5);
% X = R * [cos(t); sin(t)] + repmat(model0.Mu, 1, nbDrawingSeg);
% gaussPDF(X, model0.Mu, model0.Sigma)

%Log-normal distribution
subplot(1,2,2); hold on; grid on; title('Log-normal distribution');
nbX = 50;
[X0,Y0] = meshgrid(linspace(min(Data(1,:)),max(Data(1,:)),nbX), linspace(min(Data(2,:)),max(Data(2,:)),nbX));
X = reshape(X0,1,nbX^2);
Y = reshape(Y0,1,nbX^2);
C = logGaussPDF([X;Y], model0.Mu, model0.Sigma)
pcolor(X0,Y0,reshape(C,nbX,nbX)); shading flat;

plot(Data(1,:), Data(2,:), '.','markersize',8,'color',[.7 .7 .7]);
plot(DataIn, DataOut, '-','linewidth',2,'color',[.8 0 0]);

%plotGMM(model.Mu, model.Sigma*2, [0 .8 0], .3);

nbDrawingSeg = 35;
t = linspace(-pi, pi, nbDrawingSeg);
[V,D] = eig(model0.Sigma*2);
R = real(V*D.^.5);
X = exp(R * [cos(t); sin(t)] + repmat(model0.Mu, 1, nbDrawingSeg));
patch(X(1,:), X(2,:), [.8 0 0], 'lineWidth', 1, 'EdgeColor', [.4 0 0], 'facealpha', .3,'edgealpha', .3);

plot(model.Mu(1,:), model.Mu(2,:), '.', 'markersize', 6, 'color', [.4 0 0]);
axis equal; axis([min(Data(1,:)), max(Data(1,:)), min(Data(2,:)), max(Data(2,:))]); 
xlabel('exp(x_1)'); ylabel('exp(x_2)');
set(gca,'xtick',exp(model0.Mu(1)),'ytick',exp(model0.Mu(2)));
%print('-dpng','graphs/demo_GMM_logNormal01.png');

pause;
close all;