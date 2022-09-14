function Data = getRobotDataFromCursor(nbData)

nbSamples = 0;
exitFlag = 0;
runningFlag = 0;
if nargin<1
	nbData = 100;
end


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kp = 100; %Tracking gain
dt = 0.01; %Time step
dxMax = 10; %Maximum speed allowed
nbDOFs = 3; %Nb of articulations
armLength = 1.0; %Length of an articulation
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robot = SerialLink(repmat(L1,nbDOFs,1));
q = ones(nbDOFs,1) * -pi/nbDOFs; %Initial pose
q(1) = q(1) + pi;


%% Set up application
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('position',[10,10,700,700],'name','Move the robot with the mouse and wheel mouse'); hold on; box on;
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
setappdata(gcf,'motion',[]);
setappdata(gcf,'data',[]);
setappdata(gcf,'nbSamples',nbSamples);
setappdata(gcf,'exitFlag',exitFlag);
setappdata(gcf,'runningFlag',runningFlag);
setappdata(gcf,'mw',0);
set(fig,'WindowButtonMotionFcn',{@wbm});
set(fig,'WindowButtonDownFcn',{@wbd});
set(fig,'WindowButtonUpFcn',{@wbu});
set(fig,'WindowScrollWheelFcn',{@wsw});
axis([-1 1 -.1 1.9]);
plot(0,0,'k+');
hp = plotArm(q, ones(nbDOFs,1)*armLength, [0;0;0], .02, [0 0 0]);

motion = [];
Data = [];
oh = 0;
while exitFlag==0
	mw = getappdata(gcf,'mw');
	oh = oh + mw * 0.1;
% 		if oh > pi
% 			oh = oh - mw * 0.1;
% 		elseif oh < -pi
% 			oh = oh + mw * 0.1;
% 		end
	setappdata(gcf,'mw',0);
	cur_point = get(gca,'Currentpoint');
	xh = [cur_point(1,1:2)'; oh];

	%IK
	Htmp = robot.fkine(q);
	o = tr2eul(Htmp)';
	x = [Htmp.t(1:2,end); o(3)]; %x,y,e_z
	J = robot.jacob0(q);
	J = J([1:2,end],:); %dx,dy,wz	
	dxh = kp * (xh-x);
	if norm(dxh) > dxMax
		dxh = dxMax * dxh / norm(dxh);
	end
	dq = J \ dxh;
	q = q + dq*dt;

	%Plot
	delete(hp);
	hp = plotArm(q, ones(nbDOFs,1)*armLength, [0;0;0], .02, [0 0 0]);
	drawnow;

	if runningFlag==1 
		%Log data
		motion = [motion, [x;q]];
		
		plot(motion(1,end),motion(2,end),'k.','markerSize',1);
		
		runningFlag = getappdata(gcf,'runningFlag');
		if(runningFlag==0)
			duration = getappdata(gcf,'duration');
			%Resampling
			nbDataTmp = size(motion,2);
			xx = linspace(1,nbDataTmp,nbData);
			motion = spline(1:nbDataTmp, motion, xx);
			%motion = interp1(1:nbDataTmp, motion', xx)';
			motion_smooth = motion;
			for n=1:1
				for i=1:size(motion_smooth,1)
					motion_smooth(i,:) = smooth(motion_smooth(i,:),5);
				end
			end
			plot(motion_smooth(1,:),motion_smooth(2,:), 'r', 'lineWidth', 1);
			nbSamples = nbSamples + 1;
			%Data = [Data, [1:nbData; motion_smooth]]; %With time entries
			Data = [Data, motion_smooth]; %Without time entries
			motion = [];
		end
	end
	runningFlag = getappdata(gcf,'runningFlag');
	exitFlag = getappdata(gcf,'exitFlag');
end

close all;


%% Mouse button down
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wbd(h,evd) % executes when the mouse button is pressed
muoseside = get(gcf,'SelectionType');
if strcmp(muoseside,'alt')==1
	setappdata(gcf,'exitFlag',1);
	return;
end
setappdata(gcf,'runningFlag',1);
tic


%% Mouse move
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wbm(h,evd) % executes while the mouse moves

	
%% Mouse scroll wheel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wsw(h,evd) % executes while the mouse moves
setappdata(gcf,'mw',evd.VerticalScrollCount);
	

%% Mouse button up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wbu(h,evd) % executes when the mouse button is released
setappdata(gcf,'runningFlag',0);
duration = toc;
setappdata(gcf,'duration',duration);

