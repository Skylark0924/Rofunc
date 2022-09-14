function demo_DPmeans_online01
% Online clustering with DP-Means algorithm.
% 
% If this code is useful for your research, please cite the related publication:
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
disp('Draw path with the left mouse button, and exit with the right mouse button.');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
minSigma = 1E-5;
exitFlag = 0;
runningFlag = 0;
N = 0;
lambda = 0.04;


%% Online GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('position',[10 10 700 700]); hold on; box on; axis off;
setappdata(fig,'exitFlag',exitFlag);
setappdata(fig,'runningFlag',runningFlag);
set(fig,'WindowButtonDownFcn',{@wbd});
disp('-Left mouse button to draw several trajectories');
disp('-Right mouse button when done');
axis([-0.1 0.1 -0.1 0.1]);

h = [];
model = [];
while exitFlag==0
	drawnow;
	if runningFlag==1
		cur_point = get(gca,'Currentpoint');
		P = cur_point(1,1:2)';
		[model,N] = OnlineEMDP(N,P,minSigma,model,lambda);
		plot(P(1),P(2),'k.','markerSize',5);
		delete(h);
		h = plotGMM(model.Mu,model.Sigma,[1 0 0],0.6);
	end
	runningFlag = getappdata(fig,'runningFlag');
	exitFlag = getappdata(fig,'exitFlag');
end

close all;
end

% -----------------------------------------------------------------------
function wbd(h,evd) % executes when the mouse button is pressed
	%disp('button down');
	muoseside = get(gcf,'SelectionType');
	if strcmp(muoseside,'alt')==1
		setappdata(gcf,'exitFlag',1);
		return;
	end
	%get the values and store them in the figure's appdata
	props.WindowButtonMotionFcn = get(h,'WindowButtonMotionFcn');
	props.WindowButtonUpFcn = get(h,'WindowButtonUpFcn');
	setappdata(h,'TestGuiCallbacks',props);
	set(h,'WindowButtonMotionFcn',{@wbm});
	set(h,'WindowButtonUpFcn',{@wbu});
	setappdata(gcf,'runningFlag',1);
end
% -----------------------------------------------------------------------
function wbm(h,evd) % executes while the mouse moves
	%disp('mouse moves');
end
% -----------------------------------------------------------------------
function wbu(h,evd) % executes when the mouse button is released
	%disp('button released');
	setappdata(gcf,'runningFlag',0);
	%get the properties and restore them
	props = getappdata(h,'TestGuiCallbacks');
	set(h,props);
end