function demo_regularization01
% Regularization of GMM parameters with minimum admissible eigenvalue.
%
% If this code is useful for your research, please cite the related publication:
% @misc{pbdlib,
% 	title = {{PbDlib} robot programming by demonstration software library},
% 	howpublished = {\url{http://www.idiap.ch/software/pbdlib/}},
% 	note = {Accessed: 2019/04/18}
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
model.nbStates = 3; %Number of states in the GMM
model.nbVar = 2; %Number of variables [x1,x2]
nbData = 100; %Length of each trajectory
nbSamples = 5; %Number of demonstrations
minE = 1E0; %Minimum admissible eigenvalue


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/N.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; 
end


%% Parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_timeBased([repmat(1:nbData,1,nbSamples); Data], model);
model.Mu = model.Mu(2:end,:);
model.Sigma = model.Sigma(2:end,2:end,:);
model = EM_GMM(Data, model);


%% Regularization after parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:model.nbStates
	[V(:,:,i), D(:,:,i)] = eigs(model.Sigma(:,:,i)); %Eigendecomposition
	D2(:,:,i) = diag(max(diag(D(:,:,i)),minE)); %Apply threshold on small eigenvalues
	model.Sigma2(:,:,i) = V(:,:,i) * D2(:,:,i) * V(:,:,i)'; %Reconstruct covariance
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,700,500]); hold on; axis off;
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.5 .5 .5]);
plotGMM(model.Mu, model.Sigma, [.8 0 0],.5);
for i=1:model.nbStates
	for j=1:model.nbVar
		plot2DArrow(model.Mu(:,i), V(:,j,i) * D(j,j,i)^.5, [.4 0 0]); %Plot rescaled eigenvector
	end
end
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
%print('-dpng','graphs/demo_regularization01a.png');

plotGMM(model.Mu, model.Sigma2, [0 .8 0],.3);
for i=1:model.nbStates
	for j=1:model.nbVar
		plot2DArrow(model.Mu(:,i), V(:,j,i) * D2(j,j,i)^.5, [0 .4 0]); %Plot rescaled eigenvector
	end
end

%print('-dpng','graphs/demo_regularization01b.png');
pause;
close all;