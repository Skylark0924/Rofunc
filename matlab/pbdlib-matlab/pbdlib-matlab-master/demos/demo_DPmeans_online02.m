function demo_DPmeans_online02
% Online clustering with DP-Means algorithm, with stochastic samples
% 
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Bruno17AURO,
% 	author="Bruno, D. and Calinon, S. and Caldwell, D. G.",
% 	title="Learning Autonomous Behaviours for the Body of a Flexible Surgical Robot",
% 	journal="Autonomous Robots",
% 	year="2017",
% 	month="February",
% 	volume="41",
% 	number="2",
% 	pages="333--347",
% 	doi="10.1007/s10514-016-9544-6"
% }
% 
% Written by Danilo Bruno and Sylvain Calinon, 2015
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
minSigma = 1E-2;
lambda = 2;


%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S = load('data/data_train.mat')';
Data = S.data';
Data = spline(1:size(Data,2), Data, linspace(1,size(Data,2),100)); %Resampling (optional)


%% Online GMM parameters estimation (data fed one by one)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 700 700]); hold on; box on; axis off; 
axis([min(Data(1,:)), max(Data(1,:)), min(Data(2,:)), max(Data(2,:))]); axis equal;
h = [];
model = [];
N = 0;
for t=1:size(Data,2)
	[model,N] = OnlineEMDP(N, Data(:,t), minSigma, model, lambda);
	delete(h);
	h = plotGMM(model.Mu,model.Sigma,[1 0 0],.4);
	plot(Data(1,t),Data(2,t),'k.','markerSize',5);
	drawnow;
end

pause;
close all;