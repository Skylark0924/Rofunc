function demo_TPGMM_teleoperation01
% Time-invariant task-parameterized GMM applied to a teleoperation task (position and orientation).
%
% Octave users should use: pkg load image, pkg load statistics
%
% If this code is useful for your research, please cite the related publication:
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%		publisher="Springer Berlin Heidelberg",
%		doi="10.1007/s11370-015-0187-9",
%		year="2016",
%		volume="9",
%		number="1",
%		pages="1--29"
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
disp('Use the mouse wheel to change the orientation, and move close or far from the line to see the change of behavior.');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbData = 100;
model.nbStates = 2; %Number of Gaussians in the GMM
model.nbFrames = 1; %Number of candidate frames of reference
model.nbVarIn = 2; %Input dimension (position of object)
model.nbVarOut = 1; %Output dimension (orientation of robot end-effector)
model.nbVar = model.nbVarIn + model.nbVarOut;
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.dt = 1E-2; %Time step duration
model.rfactor = 5E-4;	%Control cost in LQR
model.tfactor = 1E-2;	%Teleoperator cost
R = eye(model.nbVarOut) * model.rfactor; %Control cost matrix

imgVis = 0; %Visualization option
if imgVis==1
	global img alpha
	[img, ~, alpha] = imread('data/drill06.png');
	% img = imresize(img, .2);
	% alpha = imresize(alpha, .2);
end


%% Discrete dynamical System settings 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1d = zeros(model.nbDeriv);
for i=0:model.nbDeriv-1
	A1d = A1d + diag(ones(model.nbDeriv-i,1),i) * model.dt^i * 1/factorial(i); %Discrete 1D
end
B1d = zeros(model.nbDeriv,1); 
for i=1:model.nbDeriv
	B1d(model.nbDeriv-i+1) = model.dt^i * 1/factorial(i); %Discrete 1D
end
A = kron(A1d, eye(model.nbVarOut)); %Discrete nD
B = kron(B1d, eye(model.nbVarOut)); %Discrete nD


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posObj = [zeros(1,nbData); ones(1,nbData)*-1500; mvnrnd(0,eye(model.nbVarOut)*1E-4,nbData)']; %Object position and orientation
Data0 = [repmat([1000;-1100],1,nbData/2)+diag([1E3,4E2])*randn(2,nbData/2), repmat([1000;1000],1,nbData/2)+randn(2,nbData/2)*4E2]; %Path of robot/teleoperator (first half close to plane, second half far from plane)
Data0 = [Data0; [posObj(end,1:nbData/2)-pi/2+randn(1,nbData/2)*1E-5, (rand(1,nbData/2)-0.5)*2*pi]]; %Concatenation of orientation data (first half close to plane, second half far from plane)

%Set task parameters
for t=1:nbData
	model.p(1,t).b = posObj(:,t); %param1 (object)
	for m=1:model.nbFrames
		model.p(m,t).A = [cos(posObj(end,t)), -sin(posObj(end,t)), 0; sin(posObj(end,t)) cos(posObj(end,t)) 0; 0 0 1];
		model.p(m,t).invA = inv(model.p(m,t).A); %Precomputation of inverse
	end
end

%Observation of data from the perspective of the frames
for m=1:model.nbFrames
	for t=1:nbData
		Data(:,m,t) = model.p(m,t).invA * (Data0(:,t) - model.p(m,t).b); 
	end
end


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of TP-GMM with EM');
model = init_tensorGMM_kmeans(Data, model); 
model = EM_tensorGMM(Data, model);


% %% Simulate reproduction
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('Reproduction\n');
% 
% %Simulate object and teleoperator trajectories
% rPosObj = [zeros(1,nbData); ones(1,nbData)*-200; ones(1,nbData).*-0.2]; %Object position and orientation
%
% aTmp = linspace(0,4*pi,nbData);
% rPosTel = [ones(1,nbData).*aTmp*(0.02/4*pi)+sin(aTmp)*0.05+0.1; cos(aTmp)*0.4; ones(1,nbData).*0.2]; %Teleoperator position and orientation 
% 
% rPosTel(1:2,:) = rPosTel(1:2,:) * 1000;
%
% x = rPosTel(model.nbVarIn+1:model.nbVarIn+model.nbVarOut, 1);
% dx = zeros(model.nbVarOut,1);
% for t=1:nbData
% 	%Frame1 (object)
% 	%pTmp(1).A = eye(model.nbVar);
% 	pTmp(1).A = [cos(rPosObj(end,t)), -sin(rPosObj(end,t)), 0; sin(rPosObj(end,t)) cos(rPosObj(end,t)) 0; 0 0 1];
% 	pTmp(1).b = rPosObj(:,t);
% 
% 	%GMR with GMM adapted to the current situation
% 	mtmp.nbStates = model.nbStates;
% 	mtmp.Priors = model.Priors;
% 	for i=1:mtmp.nbStates
% 		mtmp.Mu(:,i) = pTmp(1).A * squeeze(model.Mu(:,1,i)) + pTmp(1).b;
% 		mtmp.Sigma(:,:,i) = pTmp(1).A * squeeze(model.Sigma(:,:,1,i)) * pTmp(1).A';
% 	end
% 	[MuOut(:,1), SigmaOut(:,:,1)] = GMR(mtmp, rPosTel(1:model.nbVarIn,t), 1:model.nbVarIn, model.nbVarIn+1:model.nbVarIn+model.nbVarOut);
% 	
% 	%Second Gaussian as teleoperator
% 	MuOut(:,2) = rPosTel(model.nbVarIn+1:model.nbVarIn+model.nbVarOut,t);
% 	SigmaOut(:,:,2) = eye(model.nbVarOut) * model.tfactor; 
% 	
% 	%Product of Gaussians
% 	SigmaTmp = zeros(model.nbVarOut);
% 	MuTmp = zeros(model.nbVarOut,1);
% 	for m=1:2
% 		SigmaTmp = SigmaTmp + inv(SigmaOut(:,:,m));
% 		MuTmp = MuTmp + SigmaOut(:,:,m) \ MuOut(:,m);
% 	end
% 	rMu(:,t) = SigmaTmp \ MuTmp;
% 	rSigma(:,:,t) = inv(SigmaTmp);
% 	
% 	%Linear quadratic tracking (infinite horizon)
% 	Q = zeros(model.nbVarOut*2);
% 	Q(1:model.nbVarOut,1:model.nbVarOut) = SigmaTmp;
% 	P = solveAlgebraicRiccati_eig(A, B/R*B', (Q+Q')/2); 
% 	L = R\B'*P; %Feedback term
% 	ddx = L * ([rMu(:,t); zeros(model.nbVarOut,1)] - [x; dx]); %Compute acceleration (with only feedback terms)
% 	dx = dx + ddx * model.dt;
% 	x = x + dx * model.dt;
% 	rData(:,t) = [rPosTel(1:model.nbVarIn,t); x];
% end


% %% Plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[20,50,2300,900]);
% clrmap = lines(model.nbStates);
% ttl={'Frame 1 (in)','Frame 1 (out)'};
% 
% %DEMOS
% subplot(1,4,1); hold on; box on; title('Demonstrations');
% plotPlane(posObj, [0 0 0]);
% plotEE(Data0, [.7 0 .7], 0);
% %legend('object','robot','teleoperator');
% axis equal; %set(gca,'xtick',[],'ytick',[]);
% 
% %FRAME IN 
% subplot(1,4,2); hold on; grid on; box on; title(ttl{1});
% plot(squeeze(Data(1,1,:)), squeeze(Data(2,1,:)), '.','markersize',15,'color',[.7 0 .7]);
% for i=1:model.nbStates
% 	plotGMM(squeeze(model.Mu(1:2,1,i)), squeeze(model.Sigma(1:2,1:2,1,i)), clrmap(i,:), .4);
% end
% axis equal; %set(gca,'xtick',[0],'ytick',[0]);
% 
% %FRAME OUT 
% % subplot(1,4,3); hold on; grid on; box on; title(ttl{2});
% for i=1:model.nbStates
% 	subplot(1,4,2+i); hold on; grid on; box on; title(ttl{2});
% 	mtmp.nbStates = 1;
% 	mtmp.Priors = model.Priors(i);
% 	mtmp.Mu = model.Mu(3,1,i);
% 	mtmp.Sigma = model.Sigma(3,3,1,i);
% 	plotGMM1D(mtmp, [-pi,pi,0,1], clrmap(i,:), .3, 50);	
% 	axis([-pi,pi,-.05,1.2]); 
% 	set(gca,'ytick',[0,1],'xtick',[-pi,-pi/2,0,pi/2,pi],'xticklabel',{'-\pi','-\pi/2','0','\pi/2','\pi'});
% end
% 
% %print('-dpng','graphs/demo_TPGMM_teleoperation_model01.png');
% pause;
% close all;
% return


% for t=1:5:nbData
% 	mtmp.nbStates = 1;
% 	mtmp.Priors = 1;
% 	mtmp.Mu = rMu(:,t);
% 	mtmp.Sigma = rSigma(:,:,t);
% 	plotGMM1D(mtmp, [.2 .2 .2], [-1,0,5,1], .3, 50);
% end
% %REPRO
% subplot(1,4,4); hold on; grid on; box on; title('Reproduction');
% axis equal; 
% %axis([min(rPosObj(1,:)) max(rPosObj(1,:)) min(rPosObj(2,:)) max(rPosObj(2,:))]); set(gca,'xtick',[0],'ytick',[0]);
% plotPlane(rPosObj, [0 0 0]);
% plotEE(rPosTel, [.5 .5 .5]);
% plotEE([rPosTel(1:model.nbVarIn,:); rMu], [0 .7 0]);
% plotEE(rData, [.7 0 .7]);
% 
% %print('-dpng','graphs/demo_TPGMM_teleoperation02.png');
% pause;
% % close all;


%% Interactive plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure('position',[10,10,700,1300],'name','Move the robot with the mouse and wheel mouse'); hold on; box on;
set(fig,'WindowButtonMotionFcn',{@wbm});
set(fig,'WindowButtonDownFcn',{@wbd});
set(fig,'WindowScrollWheelFcn',{@wsw});
set(fig,'CloseRequestFcn',{@crq});
H = uicontrol('Style','PushButton','String','Exit','Callback','delete(gcbf)');
setappdata(gcf,'mw',0);
set(gca,'Xtick',[]); set(gca,'Ytick',[]);
axis equal; axis([0 2000 -1500 1500]); 

rpo = [0; -1000; -0.2]; %Object position and orientation
xt = [zeros(model.nbVarIn,1); -pi];
x = [-pi/2; 0];
hp = [];
id = 1; idf = 1;
while (ishandle(H)) 
	cur_point = get(gca,'Currentpoint');
	xt(1:2) = cur_point(1,1:2)';
	xt(3) = xt(3) + getappdata(gcf,'mw') * 0.1;
	setappdata(gcf,'mw',0);
	
	
	%Frame1 (object)
	offtmp = getappdata(gcf,'mb');
	if offtmp~=0
		rpo(end) = rpo(end) + offtmp * 0.3;
		setappdata(gcf,'mb',0);
	end
	pTmp(1).A = [cos(rpo(end)), -sin(rpo(end)), 0; sin(rpo(end)) cos(rpo(end)) 0; 0 0 1];
	pTmp(1).b = rpo;
	
	
	%GMM adapted to the current situation
	mtmp = [];
	mtmp.nbStates = model.nbStates;
	mtmp.Priors = model.Priors;
	for i=1:mtmp.nbStates
		mtmp.Mu(:,i) = pTmp(1).A * squeeze(model.Mu(:,1,i)) + pTmp(1).b;
		mtmp.Sigma(:,:,i) = pTmp(1).A * squeeze(model.Sigma(:,:,1,i)) * pTmp(1).A';
	end
	
	
	%GMR 
	in = 1:model.nbVarIn;
	out = model.nbVarIn+1:model.nbVar;
	MuOut = zeros(model.nbVarOut, 1);
	MuTmp = zeros(model.nbVarOut, model.nbStates);
	SigmaOut = zeros(model.nbVarOut, model.nbVarOut, 1);
	SigmaYX = zeros(model.nbVarOut, model.nbVarOut, model.nbStates);
	for i=1:model.nbStates
		SigmaYX(:,:,i) = mtmp.Sigma(out,out,i); % - mtmp.Sigma(out,in,i) / mtmp.Sigma(in,in,i) * mtmp.Sigma(in,out,i);
	end
	%Compute activation weight
	for i=1:model.nbStates
		h(i) = gaussPDF(xt(in), mtmp.Mu(in,i), mtmp.Sigma(in,in,i)); 
	end
	h = h ./ (sum(h)+realmin);
	%Compute conditional means
	for i=1:model.nbStates
		MuTmp(:,i) = mtmp.Mu(out,i); % + mtmp.Sigma(out,in,i) / mtmp.Sigma(in,in,i) * (xt(in) - mtmp.Mu(in,i));
		MuOut = MuOut + h(i) * MuTmp(:,i);
	end
	%Compute conditional covariances
	for i=1:model.nbStates
		SigmaOut = SigmaOut + h(i) * SigmaYX(:,:,i);
% 		SigmaOut = SigmaOut + h(i) * (SigmaYX(:,:,i) + MuTmp(:,i) * MuTmp(:,i)');
	end
% 	SigmaOut = SigmaOut - MuOut*MuOut' + eye(model.nbVarOut) * model.params_diagRegFact; 



	%Set second Gaussian as teleoperator
	MuOut(:,2) = xt(model.nbVarIn+1:model.nbVarIn+model.nbVarOut);
	SigmaOut(:,:,2) = eye(model.nbVarOut) * model.tfactor; 
	
	%Product of Gaussians
	SigmaTmp = zeros(model.nbVarOut);
	MuTmp = zeros(model.nbVarOut,1);
	for m=1:2
		SigmaTmp = SigmaTmp + inv(SigmaOut(:,:,m));
		MuTmp = MuTmp + SigmaOut(:,:,m) \ MuOut(:,m);
	end
	rMu = SigmaTmp \ MuTmp;
	
	
	%Linear quadratic tracking (infinite horizon, discrete version)
	Q = blkdiag(SigmaTmp, zeros(model.nbVarOut));
	P = solveAlgebraicRiccati_eig_discrete(A, B*(R\B'), Q);
	L = (B' * P * B + R) \ B' * P * A; %Feedback gain (discrete version)
	u = L * ([rMu; zeros(model.nbVarOut,1)] - x); %Compute acceleration commands (with only feedback terms)
	x = A * x + B * u;
	xr = [xt(1:model.nbVarIn); x];
% 	xh = [xt(1:model.nbVarIn); rMu];
% 	xh2 = [xt(1:model.nbVarIn); MuOut(:,1)];
	
	
	%Plot
	delete(hp); hp=[];
% 	for i=1:model.nbStates
% 		hp = [hp, plotGMM(mtmp.Mu(1:model.nbVarIn,i), mtmp.Sigma(1:model.nbVarIn,1:model.nbVarIn,i), [1 .7 .7], .3)];
% 	end
	hp = [hp, plotPlane(rpo,[0 0 0])];
	hp = [hp, plotEE(xt, [.8 .8 1], imgVis)];
	hp = [hp, plotEE(xr, [0 0 .8], imgVis)];
% 	hp = [hp plotEE(xh, [0 .7 0], 0)];
% 	hp = [hp plotEE(xh2, [.8 0 0], 0)];
	drawnow;
	
% % 	print('-dpng',['graphs/anim/teleop_drill' num2str(id,'%.3d') '.png']);
% 	id = id+1;
% 	if mod(id,10)==0
% 		print('-dpng',['graphs/anim/teleop_drill' num2str(idf,'%.3d') '.png']);
% 		idf = idf+1;
% 	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotEE(x, col, imgVis)
global img alpha
if imgVis==1
	h = [];
	if col(1)==0 
		img2 = imrotate(img, rad2deg(-x(3,1)));
		alpha2 = imrotate(alpha, rad2deg(-x(3,1)));
		h = [h, image(x(1,1)-size(img2,1)/2, x(2,1)-size(img2,2)/2, img2, 'AlphaData', alpha2)];
	else
		img2 = imrotate(img, rad2deg(-x(3,1)));
		alpha2 = imrotate(alpha, rad2deg(-x(3,1)));
		h = [h, image(x(1,1)-size(img2,1)/2, x(2,1)-size(img2,2)/2, img2, 'AlphaData', alpha2.*.3)];
	end
else
	h = plot(x(1,:), x(2,:), '.','markersize',20,'color',col);
	for t=1:size(x,2)
		msh = [x(1:2,t), x(1:2,t) + [cos(x(3,t)); sin(x(3,t))] .* 150]; 
		h = [h, plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',col)];
	end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = plotPlane(x,col)
h = []; 
%plot(x(1,:), x(2,:), '.','markersize',20,'color',col);
for t=1:size(x,2)
	msh = [x(1:2,t), x(1:2,t) + [cos(x(3,t)); sin(x(3,t))] .* 2100]; 
	h = [h plot(msh(1,:), msh(2,:), '-','linewidth',3,'color',col)];
end
if size(x,2)==1
	msh = [msh, msh(:,2)-[0;100], msh(:,1)-[0;100]];
	h = [h patch(msh(1,:), msh(2,:), [.7 .7 .7],'linestyle','none')];
	h = [h plot2DArrow(mean(msh,2), [cos(x(3,1)); sin(x(3,1))] .* 250, [.8 0 0], 4, 5E1)];
	h = [h plot2DArrow(mean(msh,2), [-sin(x(3,1)); cos(x(3,1))] .* 250, [.8 0 0], 4, 5E1)];
end


%% Mouse move
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wbm(h,evd)


%% Mouse scroll wheel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wsw(h,evd) 
setappdata(gcf,'mw',evd.VerticalScrollCount);


%% Mouse button down
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function wbd(h,evd) % executes when the mouse button is pressed
muoseside = get(gcf,'SelectionType');
if strcmp(muoseside,'normal')==1
	setappdata(gcf,'mb',-1);
end
if strcmp(muoseside,'alt')==1
	setappdata(gcf,'mb',1);
end
