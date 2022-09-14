function demo_HSMM_online01
% Online HSMM 
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
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Ioannis Havoutis and Sylvain Calinon
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
lambda = 0.02;
addDemo = 0;
sampleModel = 0;


%% Online GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('position',[10 10 700 700]); hold on; box on; axis on;
setappdata(fig,'exitFlag',exitFlag);
setappdata(fig,'runningFlag',runningFlag);
setappdata(fig,'sampleModel',sampleModel);
set(fig,'WindowButtonDownFcn',{@wbd});
disp('-Left mouse button to draw several trajectories');
disp('-Middle mouse button to sample model from mouse position');
disp('-Right mouse button when done');
axis([-0.1 0.1 -0.1 0.1]);
title('\lambda = 0.02, minSigma = 1E-5');

h = [];
model = [];
demo = [];
while exitFlag==0
	drawnow;
	if runningFlag==1
        addDemo = 1;
		cur_point = get(gca,'Currentpoint');
		P = cur_point(1,1:2)';
        demo = [demo, P];
		[model,N] = OnlineEMDP(N,P,minSigma,model,lambda);
		plot(P(1),P(2),'k.','markerSize',5);
        delete(h);
        h = plotGMM(model.Mu,model.Sigma, [0.5 0.5 0.5], .3);%[1 0 0],0.6);
    end
    if runningFlag==0 && addDemo==1
        addDemo = 0;
        model = OnlineHSMM(model, demo, 1);
        demo = [];
    end
    if sampleModel == 1
        cur_point = get(gca,'Currentpoint'); % get current point
        P = cur_point(1,1:2)';
        plot(P(1),P(2),'ro','markerSize',5);
        closeness = zeros(size(model.Sigma,3),1); % find "closest" state
        for i=1:size(model.Sigma,3)
            closeness(i,:)=gaussPDF(P, model.Mu(:,i), model.Sigma(:,:,i));
        end
        [~,start_st] = max(closeness);
        pred = sample_hsmm_lqr(model, start_st, 250, P); % 250 is here arbitary
        plot(pred(1,:),pred(2,:),'-r','LineWidth',1.5);
        setappdata(gcf,'sampleModel',0);
    end
    sampleModel = getappdata(fig,'sampleModel');
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
    elseif strcmp(muoseside,'extend')==1
        disp('Sample starting from the "nearest" state.');
   		setappdata(gcf,'sampleModel',1);
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
    setappdata(gcf,'sampleModel',0);
	%get the properties and restore them
	props = getappdata(h,'TestGuiCallbacks');
	set(h,props);
end