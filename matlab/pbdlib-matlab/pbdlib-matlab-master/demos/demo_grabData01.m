function demo_grabData01
% Collect movement data from mouse cursor
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
% 	author="Calinon, S.",
% 	title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
% 	journal="Intelligent Service Robotics",
% 	publisher="Springer Berlin Heidelberg",
% 	year="2016",
% 	volume="9",
% 	number="1",
% 	pages="1--29",
% 	doi="10.1007/s11370-015-0187-9",
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
model.dt = 0.01; %Duration of time step
nbData = 200; %Length of each trajectory
nbSamples = 5;

%% Collect data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Use left mouse button to draw trajectories, use right mouse button to quit');
Data = grabDataFromCursor(nbData);
nbSamples = size(Data,2) / nbData;
for n=1:nbSamples
	demos{n}.pos = Data(:,(n-1)*nbData+1:n*nbData);
	demos{n}.vel = gradient(demos{n}.pos) / model.dt;
	demos{n}.acc = gradient(demos{n}.vel) / model.dt;
end
% save('data/2Dletters/tmp.mat','demos');

Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	Data = [Data s(n).Data]; 
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,500]); hold on; box on; 
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.7 .7 .7]);
axis equal; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

%print('-dpng','graphs/demo_grabData01.png');
pause;
close all;
