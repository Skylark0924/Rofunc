function demo_search01
% EM-based stochastic optimization 
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon13RAS,
% 	author="Calinon, S. and Kormushev, P. and Caldwell, D. G.",
% 	title="Compliant skills acquisition and multi-optima policy search with {EM}-based reinforcement learning",
% 	journal="Robotics and Autonomous Systems",
% 	year="2013",
% 	month="April",
% 	Volume="61",
% 	number="4",
% 	pages="369--379",
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 2; %Dimension of datapoints
nbEpisods = 50; %Number of exploration iterations
nbE = 5; %Number of initial points (for the first iteration)
nbPointsRegr = 5; %Number of points with highest rewards considered at each iteration (importance sampling)
minSigma = eye(nbVar) * 1E0; %Minimum exploration covariance matrix

Mu = [4; 8]; %Initial policy parameters
Sigma = eye(nbVar) * 5E0; %Initial exploration noise
p=[]; %Storing tested parameters (initialized as empty)
r=[]; %Storing associated rewards (initialized as empty)


%% EM-based stochastic optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbEpisods  
  %Generate noisy data with variable exploration noise
	[V,D] = eig(Sigma);
  pNoisy = repmat(Mu,1,nbE) + V*D.^.5 * randn(nbVar,nbE);
  
  nbE = 1; %nbE=1 for the next iterations
	
  %Compute associated rewards
  rNoisy = rewardEval(pNoisy);
  %Add new points to dataset
  p = [p, pNoisy]; 
  r = [r, rNoisy]; 
  %Keep the nbPointsRegr points with highest rewards
  [rSrt, idSrt] = sort(r,'descend');
  nbP = min(length(idSrt),nbPointsRegr);
  pTmp = p(:,idSrt(1:nbP));
  rTmp = rSrt(1:nbP);
  
  %Compute error term
  eTmp = pTmp - repmat(Mu,1,nbP);

  %Udpate of mean and covariance (exploration noise)
  Mu = pTmp * rTmp' / sum(rTmp);  
  Sigma0 = (eTmp * diag(rTmp) * eTmp') / sum(rTmp);
	
%   %Udpate of the policy with a form corresponding to gradient descent (giving same result)
%   Mu = Mu + eTmp * rTmp' / sum(rTmp);

%   %CEM update of mean and covariance (exploration noise)
%   Mu = mean(pTmp,2);
%   Sigma0 = (eTmp * eTmp') / nbP;
  
  %Add minimal exploration noise
  Sigma = Sigma0 + minSigma;
  
  %Log data (for display purpose)
  s(n).Mu = Mu; 
  s(n).Sigma0 = Sigma0; 
  s(n).Sigma = Sigma; 
  s(n).r = r(end);
  s(n).nbData = length(r);
  s(n).pTmp = pTmp;
end


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rgMax = 10; 
nbGrid = 200;
X = linspace(0,rgMax,nbGrid); Y = linspace(0,rgMax,nbGrid);
[xx,yy] = meshgrid(X,Y);
x = reshape(xx,1,nbGrid*nbGrid); y = reshape(yy,1,nbGrid*nbGrid);
z = rewardEval([x;y]);
zz = reshape(z,nbGrid,nbGrid);
nbContours = 12;
listContours = linspace(.005,0.043,nbContours);
colCont = repmat(linspace(.9,.6,nbContours),3,1);

figure('PaperPosition',[0 0 4 4],'position',[10 10 700 700]); 
axes('Position',[0 0 1 1]); hold on; box off;
axis([0 rgMax 0 rgMax]); axis square;
set(gca,'xtick',[],'ytick',[]);
for i=1:length(listContours) 
	v = [listContours(i), listContours(i)]; %This is to handle bug in the new version of the contour() function
  contour(xx,yy,zz,v,'color',colCont(:,i),'linewidth',1);
end
hg1=[]; hg3=[]; 
for n=1:nbEpisods
  delete(hg1);
  hg1 = plot(p(1,1:s(n).nbData),p(2,1:s(n).nbData),'.','markersize',10,'color',[0 0 0]); 
	pause(0.01); drawnow;
  delete(hg3); 
  hg2 = plot(s(n).pTmp(1,:),s(n).pTmp(2,:),'o','color',[0 .8 0],'linewidth',2);  
  hg3 = plotGMM(s(n).Mu, s(n).Sigma0,[1 0 0], .1);  
	pause(0.01); drawnow;
  delete(hg2);
end

%print('-dpng','graphs/demo_search01.png');
pause;
close all;


%% Compute reward function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r = rewardEval(p)
Mu(:,1) = [4.5; 6.5];
Sigma(:,:,1) = [0.8 0.3; 0.3 0.4] * 10;
Mu(:,2) = [5.5; 4.5];
Sigma(:,:,2) = [0.4 0.4; 0.4 0.8] * 6;
Mu(:,3) = [6.5; 3.5];
Sigma(:,:,3) = [0.5 0.0; 0.0 0.05] * 7;
Priors = [.45 .45 .1];

nbStates = size(Mu,2);
r = zeros(1,size(p,2));
for i=1:nbStates
  r = r + Priors(i) * gaussPDF(p, Mu(:,i), Sigma(:,:,i));
end