function [Data, nbSamples] = grabDataFromCursor02(nbData)
% This version stacks the Data with varying nbData

nbSamples = 0;
exitFlag = 0;
runningFlag = 0;
if nargin<1
	nbData = 100;
end

fig = figure('position',[10,10,700,700]); hold on;
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
setappdata(gcf,'motion',[]);
setappdata(gcf,'data',[]);
setappdata(gcf,'nbSamples',nbSamples);
setappdata(gcf,'exitFlag',exitFlag);
setappdata(gcf,'runningFlag',runningFlag);
set(fig,'WindowButtonDownFcn',{@wbd});
axis([-10 10 -10 10]);
plot(0,0,'k+');

motion=[];
Data=[];
while exitFlag==0
	drawnow;
	if runningFlag==1 
		cur_point = get(gca,'Currentpoint');
		motion = [motion cur_point(1,1:2)'];
		plot(motion(1,end),motion(2,end),'k.','markerSize',1);
		runningFlag = getappdata(gcf,'runningFlag');
		if(runningFlag==0)
			duration = getappdata(gcf,'duration');
			%Resampling
			nbDataTmp = size(motion,2);
			xx = 1:nbDataTmp; %linspace(1,nbDataTmp,nbData);
			motion = spline(1:nbDataTmp, motion, xx);
			%motion = interp1(1:nbDataTmp, motion', xx)';
			motion_smooth = motion;
			for n=1:1
				motion_smooth(1,:) = smooth(motion_smooth(1,:),10);
				motion_smooth(2,:) = smooth(motion_smooth(2,:),10);
			end
			plot(motion_smooth(1,:),motion_smooth(2,:), 'r', 'lineWidth', 1);
			nbSamples = nbSamples + 1;
			Data = [Data, [xx; motion_smooth]]; %With time entries
			%Data = [Data, motion_smooth]; %Without time entries
			motion = [];
		end
	end
	runningFlag = getappdata(gcf,'runningFlag');
	exitFlag = getappdata(gcf,'exitFlag');
end

close all


% -----------------------------------------------------------------------
function wbd(h,evd) % executes when the mouse button is pressed
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
tic
% -----------------------------------------------------------------------
function wbm(h,evd) % executes while the mouse moves
% -----------------------------------------------------------------------
function wbu(h,evd) % executes when the mouse button is released
setappdata(gcf,'runningFlag',0);
duration = toc;
setappdata(gcf,'duration',duration);
%get the properties and restore them
props = getappdata(h,'TestGuiCallbacks');
set(h,props);
