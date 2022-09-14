function demo_trajDistrib_differencingMatrix01
% Conditioning on trajectory distribution constructed by differentiation matrix
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 3; %Number of derivatives
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.dt = 1E-1; %Time step (without rescaling, large values such as 1 has the advantage of creating clusers based on position information)
nbSamples = 1; %Number of demonstrations
nbData = 200; %Number of datapoints in a trajectory
nbRepros = 10; %Number of reproductions


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos = [];
load('data/2Dletters/S.mat');
Data = spline(1:size(demos{1}.pos,2), demos{1}.pos, linspace(1,size(demos{1}.pos,2),nbData)); %Resampling


%% Construct distribution from differentiation matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~, PHI1] = constructPHI(model, nbData, nbSamples); 
Mu = Data(:);
Sigma = inv(PHI1' * PHI1);


%% Conditioning on trajectory distribution (reconstruction from partial data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
in = 1:model.nbVarPos;
out = model.nbVarPos+1:model.nbVarPos*nbData;
Mu2 = zeros(model.nbVarPos*nbData, nbRepros);
for n=1:nbRepros
	%Input data
	MuIn = Mu(51:52) + (rand(model.nbVarPos,1)-0.5) .* 1E1;
	%Gaussian conditioning with trajectory distribution
	Mu2(in,n) = MuIn;
	Mu2(out,n) = Mu(out) + Sigma(out,in) / Sigma(in,in) * (MuIn - Mu(in));
end


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1650 1250]); hold on; axis off;
plot(Mu(1:model.nbVarPos:end), Mu(2:model.nbVarPos:end), '-','lineWidth',2,'color',[0 0 0]);
for n=1:nbRepros
	plot(Mu2(1:model.nbVarPos:end,n), Mu2(2:model.nbVarPos:end,n), '-','lineWidth',2,'color',[.8 0 0]);
end
axis equal; axis tight;

%Plot covariance
figure('position',[1670 10 800 800]); hold on; axis off;
colormap(flipud(gray));
pcolor(abs(Sigma)); 
set(gca,'xtick',[1,nbData*model.nbVarPos],'ytick',[1,nbData*model.nbVarPos]);
axis square; axis([1 nbData*model.nbVarPos 1 nbData*model.nbVarPos]); axis ij; shading flat;

%print('-dpng','graphs/demo_trajDistrib_differencingMatrix01.png');
pause;
close all;