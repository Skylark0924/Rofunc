function demo_LS_IRLS_logisticRegression02
% Logistic regression with multivariate inputs computed with iteratively reweighted least squares (IRLS) algorithm.
%
% If this code is useful for your research, please cite the related reference:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
% }
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarIn = 2; %Dimension of input vector
nbData = 80; %Number of datapoints
nbIter = 50; %Number of iterations for IRLS


%% Generate data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = [-8; 2; .5]; 
X =	[ones(nbData,1), rand(nbData,nbVarIn).*6]; 
p = 1 ./ (1 + exp(-X*a));
y = max(min(round(p+randn(nbData,1).*2E-1), 1), 0);


%% Iteratively reweighted least squares (IRLS)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = zeros(nbVarIn+1,1); %Parameters
for n=1:nbIter
	mu = 1 ./ (1 + exp(-X*a)); %Expected value of the Bernoulli distribution (logistic function)
	W = diag(mu .* (1 - mu)); %Diagonal weighting matrix	
	a = X' * W * X \ X' * (W * X * a + y - mu); %Update of parameters with weighted least squares
end


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbGrid = 20;
[xx, yy] = ndgrid(linspace(0,6,nbGrid), linspace(0,6,nbGrid));
X0 = [ones(nbGrid^2,1), xx(:), yy(:)];
p = 1 ./ (1 + exp(-X0*a)); %Logistic function (probability of passing the exam)

figure('PaperPosition',[0 0 8 6],'position',[10,10,1200,700]); hold on; grid on; rotate3d on;
plot3(X(:,2), X(:,3), y(:,1), '.','markersize',18,'color',[0 0 0]);

% plot3(X0(:,2), X0(:,3), p, '-','linewidth',2,'color',[.8 0 0]);
surf(xx, yy, reshape(p,nbGrid,nbGrid), 'facecolor',[.8 0 0],'edgecolor',[.8 0 0],'facealpha',0.2); %'facecolor',[1 .4 .4],'linewidth',2,'edgecolor',[.8 0 0],'facealpha',0.2
view(3); axis vis3d;

set(gca,'fontsize',12,'xtick',[],'ytick',[],'ztick',[0,.5,1]); 
xlabel('$x_1$','interpreter','latex','fontsize',28); 
ylabel('$x_2$','interpreter','latex','fontsize',28);
zlabel('$p(x)$','interpreter','latex','fontsize',28);
% print('-dpng','graphs/demo_LS_IRLS_logisticRegression02.png');

pause;
close all;