function demo_LS_IRLS_logisticRegression01
% Logistic regression computed with iteratively reweighted least squares (IRLS) algorithm,
% which is equivalent to minimizing the log-likelihood of a Bernoulli distributed process using Newton's method
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
nbVarIn = 1; %Dimension of input vector
nbData = 20; %Number of datapoints
nbIter = 50; %Number of iterations for IRLS


%% Generate data (example from https://en.wikipedia.org/wiki/Logistic_regression)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Time passed studying for an exam (explanatory variables)
X =	[ones(nbData,1), [0.50; 0.75; 1.00; 1.25; 1.50; 1.75 ;1.75; 2.00; 2.25; 2.50; 2.75; 3.00; 3.25; 3.50; 4.00; 4.25; 4.50; 4.75; 5.00; 5.50]]; 
%Failed/passed exam (binary response variables)
y =	[0; 0; 0; 0; 0; 0; 1; 0; 1; 0; 1; 0; 1; 0; 1; 1; 1; 1; 1; 1]; 


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
X0 = [ones(100,1), linspace(0,6,100)'];
p = 1 ./ (1 + exp(-X0*a)); %Logistic function (probability of passing the exam)

figure('PaperPosition',[0 0 8 4],'position',[10,10,1200,500]); hold on; grid on;
plot(X(:,2), y(:,1), '.','markersize',18,'color',[0 0 0]);
% set(gca,'fontsize',12,'xtick',0:6,'ytick',[0,.5,1]); 
% xlabel('$x$','interpreter','latex','fontsize',18); ylabel('$p(x)$','interpreter','latex','fontsize',18);
% print('-dpng','graphs/demo_LS_IRLS_logisticRegression01.png');
plot(X0(:,2), p, '-','linewidth',2,'color',[.8 0 0]);
set(gca,'fontsize',12,'xtick',sort([0:6,round(-a(1)/a(2),2)]),'ytick',[0,.5,1]); 
xlabel('$x$','interpreter','latex','fontsize',18); ylabel('$p(x)$','interpreter','latex','fontsize',18);
% print('-dpng','graphs/demo_LS_IRLS_logisticRegression02.png');

pause;
close all;