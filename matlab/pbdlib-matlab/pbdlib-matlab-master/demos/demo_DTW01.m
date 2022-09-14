function demo_DTW01
% Trajectory realignment through dynamic time warping (DTW).
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
nbData = 200; %Length of each trajectory
wMax = 50; %Warping time window 
nbSamples = 5; %Number of demonstrations
nbVar = 2; %Number of dimensions (max 2 for 2Dletters dataset)


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos(1:nbVar,:), linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
end


%% Dynamic time warping
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r(1).Data = s(1).Data;
for n=2:nbSamples
	[r(1).Data, r(n).Data, r(n-1).wPath] = DTW(r(1).Data, s(n).Data, wMax);
	%Realign previous trajectories
	p = r(n-1).wPath(1,:);
	for m=2:n-1
		DataTmp = r(m).Data(:,p);
		r(m).Data = spline(1:size(DataTmp,2), DataTmp, linspace(1,size(DataTmp,2),nbData)); %Resampling
	end
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1600,700]); 
for k=1:nbVar
	%Before DTW
	subplot(2,nbVar+2,(k-1)*4+1); hold on; if k==1 title('Before DTW'); end;
	for n=1:nbSamples
		plot(s(n).Data(k,:), '-','linewidth',1,'color',[.6 .6 .6]);
	end
	xlabel('t'); ylabel(['x_' num2str(k)]);
	axis tight; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
	%After DTW
	subplot(2,nbVar+2,(k-1)*4+2); hold on; if k==1 title('After DTW'); end;
	for n=1:nbSamples
		plot(r(n).Data(k,:), '-','linewidth',1,'color',[0 0 0]);
	end
	xlabel('t'); ylabel(['x_' num2str(k)]);
	axis tight; set(gca,'Xtick',[]); set(gca,'Ytick',[]);
end
%spatial graph
subplot(2,nbVar+2,[3,4,7,8]); hold on; 
for n=1:nbSamples
	plot(s(n).Data(1,:), s(n).Data(2,:), '-','linewidth',1,'color',[.6 .6 .6]);
	plot(s(n).Data(1,:), s(n).Data(2,:), '.','markersize',12,'color',[.6 .6 .6]);
	plot(r(n).Data(1,:), r(n).Data(2,:), '-','linewidth',1,'color',[0 0 0]);
	plot(r(n).Data(1,:), r(n).Data(2,:), '.','markersize',12,'color',[0 0 0]);
end
xlabel('t'); ylabel(['x_' num2str(k)]);
axis tight; set(gca,'Xtick',[]); set(gca,'Ytick',[]);

figure('position',[1620,10,700,700]); hold on;
for n=1:nbSamples-1
	plot(r(n).wPath(1,:),r(n).wPath(2,:),'-','color',[0 0 0]);
end
xlabel('w_1'); ylabel('w_2');

%print('-dpng','graphs/demo_DTW01.png');
pause;
close all;