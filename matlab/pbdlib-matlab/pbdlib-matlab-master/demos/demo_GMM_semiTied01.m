function demo_GMM_semiTied01
% Semi-tied Gaussian Mixture Model by tying the covariance matrices of a
% Gaussian mixture model with a set of common basis vectors.
%
% If this code is useful for your research, please cite the related publication:
% @article{Tanwani16RAL,
%   author="Tanwani, A. K. and Calinon, S.",
%   title="Learning Robot Manipulation Tasks with Task-Parameterized Semi-Tied Hidden Semi-{M}arkov Model",
%   journal="{IEEE} Robotics and Automation Letters ({RA-L})",
%   year="2016",
%   month="January",
%   volume="1",
%   number="1",
%   pages="235--242",
% 	doi="10.1109/LRA.2016.2517825"
% }
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Ajay Tanwani
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
model.nbStates = 3; %Number of states in the GMM
model.nbVar = 3; %Number of variables [x1,x2,x3]
model.nbSamples = 5; %Number of demonstrations
model.params_Bsf = 5E-2; %Initial variance of B in semi-tied GMM
nbData = 300; %Length of each trajectory


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/Zshape3D.mat'); %Load 'Data' 
model = init_GMM_kbins(Data, model, nbData);
model = EM_semitiedGMM(Data, model);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('color',[1 1 1],'Position',[10 10 700 650]); hold on; axis off; box off;
clrmap = lines(model.nbVar);
for n=1:model.nbSamples
	plot3(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), Data(3,(n-1)*nbData+1:n*nbData),'-','linewidth',1.5,'color',[.7 .7 .7]);
end
for i=1:model.nbVar
	mArrow3(zeros(model.nbVar,1), model.H(:,i), 'color',clrmap(i,:),'stemWidth',0.75, 'tipWidth',1.0, 'facealpha',0.75);
end
plotGMM3D(model.Mu, model.Sigma+repmat(eye(model.nbVar)*2E0,[1,1,model.nbStates]), [0 .6 0], .4, 2);
for i=1:model.nbStates
	for j=1:model.nbVar
		w = model.SigmaDiag(j,j,i).^0.5;
		if w>5E-1
			mArrow3(model.Mu(:,i), model.Mu(:,i)+model.H(:,j)*w, 'color',clrmap(j,:),'stemWidth',0.75, 'tipWidth',1.25, 'facealpha',1);
		end
	end
end
view(-40,6); axis equal;

%print('-dpng','graphs/demo_GMM_semitiedGMM01.png');
pause;
close all;