function demo_IK_pointing_TPGMM01
% Task-parameterized GMM to encode pointing direction by considering nullspace constraint (4 frames) 
% (example with two objects and robot frame, starting from the same initial pose (nullspace constraint), 
% by using a single Euler orientation angle and 3 DOFs robot).
%
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
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
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
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
disp('This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 6; %Number of states in the GMM
model.nbFrames = 4; %Three candidate frames are defined: 1 in joint space and 2 in task space (2 objects)
model.nbVars = [4,2,2,2]; %[[t,q],[t,e1],[t,e2]], where q are joint angles and e1,e2 are orientation offsets
model.nbVar = model.nbVars(1); %Dimension for the product of Gaussians
model.nbQ = model.nbVars(1)-1; %Number of articulations of the robot
model.nbObj = 2; %Number of objects in the workspace
model.dt = 0.01; %Time step
nbSamples = 6; %Number of demonstrations
nbRepros = 6; %Number of reproduction attempts
nbData = 300; %Length of each trajectory
nbDataRepro = nbData;
eMax = 1; %Maximum error norm for stable Jacobian computation
Kp = 0.15; %Amplification gain for error computation 
KpQ = 0.15; %Amplification gain for joint angle error computation 

needsData = 1;
needsModel = 1;
needsRepro = 1;


%% Create robot (requires the Robotics Toolbox)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
armLength = 0.2; %length of each segment
for i=1:model.nbQ
	Lrob(i) = Link('d', 0, 'a', armLength, 'alpha', 0);
end
robot = SerialLink(Lrob);


%% Demonstrations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if needsData==1
disp('Generate data...');

q0 = [0; pi/2; zeros(model.nbQ-2,1)];

for n=1:nbSamples
	
	%Set initial pose
	s(n).q(:,1) = q0;

	%Set object 1 position
	oTmp = rand(2,1) .* [armLength*2; 2*pi/3] + [armLength*model.nbQ; pi/6];
	s(n).obj(:,1,:) = repmat([oTmp(1)*cos(oTmp(2)); oTmp(1)*sin(oTmp(2))], 1, nbData); %rand(2,1).*[3;.5]+[-1;.6]
	
	%Set object 2 position
	oTmp = rand(2,1) .* [armLength*2; 2*pi/3] + [armLength*model.nbQ; pi/6];
	s(n).obj(:,2,:) = repmat([oTmp(1)*cos(oTmp(2)); oTmp(1)*sin(oTmp(2))], 1, nbData);
	
	%Motion loop
	for t=1:nbData
		%Computation of error terms for the two objects
		Htmp = robot.fkine(s(n).q(:,t));
		Etmp = tr2eul(Htmp);
		s(n).x(:,t) = Etmp(3);
		for j=1:model.nbObj
			dir = s(n).obj(:,j,t) - Htmp.t(1:2,end);
			xh = atan2(dir(2),dir(1));
			e = xh - s(n).x(:,t);
			if norm(e)>eMax
				e = eMax * e / norm(e);
			end
			s(n).e(:,j,t) = e;
		end

		%Update of robot pose (through Jacobian)
		Jtmp = robot.jacob0(s(n).q(:,t),'rot');
		s(n).J(:,:,t) = Jtmp(3,:);
		J = s(n).J(:,:,t);
		pinvJ = pinv(J);
		%pinvJ = (J'*J + eye(model.nbQ)*1E-8) \ J'; %Damped pseudoinverse
		%W = diag([1,1,1]);
		%pinvJ = (J'*W*J + eye(model.nbQ)*1E-8) \ J'*W; %Damped weighted pseudoinverse
		
		if t<nbData/3 
			s(n).dq(:,t) = pinvJ * Kp * s(n).e(:,1,t)/model.dt;
			%s(n).dq(2,t) = s(n).dq(2,t) * 2E-1; %Simulate weak articulation
		elseif t<2*nbData/3 
			s(n).dq(:,t) = pinvJ * Kp * s(n).e(:,2,t)/model.dt;
			%s(n).dq(2,t) = s(n).dq(2,t) * 2E-1; %Simulate weak articulation
		else
			%s(n).dq(:,t) = zeros(model.nbQ,1);
			s(n).dq(:,t) = KpQ * (q0 - s(n).q(:,t))/model.dt;
		end
		
		%Nullspace control
		N = eye(model.nbQ) - pinvJ*J;
		s(n).dq(:,t) = s(n).dq(:,t) + N * KpQ * (q0 - s(n).q(:,t))/model.dt;
		s(n).q(:,t+1) = s(n).q(:,t) + s(n).dq(:,t) * model.dt;
	end
end

%Generate dataset
Data = zeros(model.nbVar, model.nbFrames, nbData*nbSamples);
for n=1:nbSamples
	for t=1:nbData
		Data(1:model.nbVars(1),1,(n-1)*nbData+t) = [t*model.dt; s(n).q(:,t+1) + randn(model.nbVars(1)-1,1)*1E-6];
		Data(1:model.nbVars(2),2,(n-1)*nbData+t) = [t*model.dt; s(n).e(:,1,t) + randn(model.nbVars(2)-1,1)*1E-6];
		Data(1:model.nbVars(3),3,(n-1)*nbData+t) = [t*model.dt; s(n).e(:,2,t) + randn(model.nbVars(3)-1,1)*1E-6];
		Data(1:model.nbVars(4),4,(n-1)*nbData+t) = [t*model.dt; s(n).x(:,t) + randn(model.nbVars(4)-1,1)*1E-6]; 
	end
end

save('data/TPGMMpointing_nullspace_data02.mat','s','Data');
end %needsData
load('data/TPGMMpointing_nullspace_data02.mat');


%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if needsModel==1
fprintf('Parameters estimation of TP-GMM with EM:');
model = init_TPGMM_timeBased(Data, model); %Initialization
%model = init_TPGMM_kmeans(Data, model); %Initialization
model = EM_TPGMM(Data, model);

save('data/TPGMMpointing_nullspace_model02.mat','model');
end %needsModel
load('data/TPGMMpointing_nullspace_model02.mat');


%% Reproduction with GMR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if needsRepro==1
disp('Reproduction with GMR...');
rr.Priors = model.Priors;
rr.nbStates = model.nbStates;

for n=1:nbRepros
		
	%Set object 1 position
	oTmp = rand(2,1) .* [armLength*2; 2*pi/3] + [armLength*model.nbQ; pi/6];
	r(n).obj(:,1,:) = repmat([oTmp(1)*cos(oTmp(2)); oTmp(1)*sin(oTmp(2))], 1, nbDataRepro); 
	
	%Set object 2 position
	oTmp = rand(2,1) .* [armLength*2; 2*pi/3] + [armLength*model.nbQ; pi/6];
	r(n).obj(:,2,:) = repmat([oTmp(1)*cos(oTmp(2)); oTmp(1)*sin(oTmp(2))], 1, nbDataRepro);
	
	%Initial pose of robot
	%r(n).q(:,1) = rand(model.nbQ,1)*pi/4;
	r(n).q(:,1) = s(n).q(:,1);
	
	%Retrieval of motion
	for t=1:nbDataRepro
		
		%Compute relative orientation error
		J = robot.jacob0(r(n).q(:,t),'rot');
		J = J(3,:);
		pinvJ = pinv(J);
		Htmp = robot.fkine(r(n).q(:,t));
		Etmp = tr2eul(Htmp);
		r(n).x(:,t) = Etmp(3);
		for j=1:model.nbObj
			dir = r(n).obj(:,j,t) - Htmp.t(1:2,end);
			xh = atan2(dir(2),dir(1));
			e(:,j) = xh - r(n).x(:,t);
			if norm(e(:,j))>eMax
				e(:,j) = eMax * e(:,j) / norm(e(:,j));
			end
			r(n).e(:,j,t) = e(:,j);
		end
		
		%Update Frame 1 (null space)
		N = eye(model.nbQ) - pinvJ*J;
		pTmp(1).A = [1 zeros(1,model.nbVars(1)-1); zeros(model.nbQ,1) N*KpQ];
		pTmp(1).b = [0; r(n).q(:,t)-N*KpQ*r(n).q(:,t)];
		%pTmp(1).b = [0; pinv(Jtmp)*Jtmp*r(n).q(:,t)]; %Correct only for KpQ=1
		
		%Update Frame 2 (task space)
		pTmp(2).A = [1 zeros(1,model.nbVars(2)-1); zeros(model.nbQ,1) pinvJ*Kp];
		pTmp(2).b = [0; r(n).q(:,t)+pinvJ*Kp*r(n).e(:,1,t)];
		
		%Update Frame 3 (task space)
		pTmp(3).A = [1 zeros(1,model.nbVars(3)-1); zeros(model.nbQ,1) pinvJ*Kp];
		pTmp(3).b = [0; r(n).q(:,t)+pinvJ*Kp*r(n).e(:,2,t)];
		
		%Update Frame 4 (task space)
		pTmp(4).A = [1 zeros(1,model.nbVars(4)-1); zeros(model.nbQ,1) pinvJ];
		pTmp(4).b = [0; r(n).q(:,t)-pinvJ*r(n).x(:,t)];
		
		%TP-GMR
		[rr.Mu, rr.Sigma] = productTPGMM(model, pTmp);
		r(n).q(:,t+1) = GMR(rr, t*model.dt, 1, 2:model.nbVars(1));
	end
end

save('data/TPGMMpointing_nullspace_repro02.mat','r');
end %needsRepro
load('data/TPGMMpointing_nullspace_repro02.mat');


%% Plot timelines
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,600,680]);
fTmp = [1 3 5; 2 0 0; 4 0 0; 6 0 0]; 
for m=1:model.nbFrames
	for k=1:model.nbVars(m)-1
		subplot(model.nbVar-1, 2, fTmp(m,k)); hold on; 
		plotGMM(squeeze(model.Mu([1,k+1],m,:)), squeeze(model.Sigma([1,k+1],[1,k+1],m,:))+repmat(eye(2)*1E-4,[1 1 model.nbStates]), [0 .7 0]);
		for n=1:nbSamples
			plot(squeeze(Data(1,m,(n-1)*nbData+1:n*nbData)), squeeze(Data(k+1,m,(n-1)*nbData+1:n*nbData)), '-','color',[.3 .3 .3]);
		end
		if m==1
			for n=1:nbRepros
				plot(squeeze(Data(1,m,1:nbData)), r(n).q(k,1:nbData), '-','color',[.8 0 0],'linewidth',1.5);
			end
		elseif m==2 || m==3
			for n=1:nbRepros
				plot(squeeze(Data(1,m,1:nbData)), squeeze(r(n).e(k,m-1,:)), '-','color',[.8 0 0],'linewidth',1.5);
			end
		else
			for n=1:nbRepros
				plot(squeeze(Data(1,m,1:nbData)), squeeze(r(n).x(1,:)), '-','color',[.8 0 0],'linewidth',1.5);
			end
		end
		xlabel('$t$','interpreter','latex','fontsize',14); 
		set(gca,'xtick', [model.dt, nbData*model.dt], 'xticklabel',{'0','1'});
		ylabel(['$X^{(' num2str(m) ')}_' num2str(k) '$'],'interpreter','latex','fontsize',16); 
		if k==1
			if m==1
				title('Frame 1 (preferred pose)','fontsize',10);
			elseif m==2
				title('Frame 2 (red object)','fontsize',10);
			elseif m==3
				title('Frame 3 (blue object)','fontsize',10);
			elseif m==4
				title('Frame 4 (robot frame)','fontsize',10);
			end
		end
	end
end

 
% %% Plots 2D anim
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[10,10,1200,600],'color',[1,1,1]);
% hold on; axis off; axis equal;
% h=[];
% for n=1:nbRepros
% 	for t=round(linspace(1,nbDataRepro,nbDataRepro/4))
% 		delete(h);
% 		h = plotArm(r(n).q(:,t), [ones(model.nbQ-1,1)*armLength; armLength*5], [0; 0; -n*2+(t/nbDataRepro)], .002, [1 .7 .7],[1 .7 .7]);
% 		h = [h, plotArm(r(n).q(:,t), ones(model.nbQ,1)*armLength, [0; 0; 0], .05, [.7 .7 1])]; 
% 		h = [h, plot(r(n).obj(1,1,t), r(n).obj(2,1,t), '.','markersize',20,'color',[.8 0 0])];
% 		h = [h, plot(r(n).obj(1,2,t), r(n).obj(2,2,t), '.','markersize',20,'color',[0 0 .8])];
% 		axis([-1 2 -.2 1.1]); 
% 		%pause(0.02);
% 		drawnow;
% 		if t<3
% 			pause;
% 		end
% 	end
% end


%% Plots 2D demos static
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tList = [1, nbData/6, nbData/3+nbData/6, nbData];
figure('PaperPosition',[0 0 12 4.5],'position',[10,10,1200,450],'color',[1,1,1]);
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
for n=1:nbSamples
	for j=2:length(tList)
		subplot(length(tList)-1,nbSamples, (j-2)*nbSamples+n); hold on; axis off;
		if j==2
			title(['Demonstration ' num2str(n)],'fontsize',10);
		end
		plotArm(s(n).q(:,tList(j)), [ones(model.nbQ-1,1)*armLength; armLength*4], [0; 0; -1], .002, [.7 .7 .7],[.7 .7 .7]);
		ql = ones(model.nbQ,1) * 999;
		for t=tList(j-1):tList(j) 
			if norm(ql-s(n).q(:,t))>0.08 || t==tList(j)
				colTmp = [.3 .3 .3] + [.5 .5 .5] * (tList(j)-t)/(tList(j)-tList(j-1));
				plotArm(s(n).q(:,t), ones(model.nbQ,1)*armLength, [0; 0; t/2], .05, colTmp); 
				plot(s(n).obj(1,1,t), s(n).obj(2,1,t), '.','markersize',20,'color',[.8 0 0]);
				plot(s(n).obj(1,2,t), s(n).obj(2,2,t), '.','markersize',20,'color',[0 0 .8]);
				ql = s(n).q(:,t);
			end
		end
		text(0.1, -0.2, ['t=' num2str(tList(j)/100,'%.1f')]);
		axis equal; axis([-1 1 -.2 1.1]); 
	end
end


%% Plots 2D repros static
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 12 4.5],'position',[10,10,1200,450],'color',[1,1,1]);
set(0,'DefaultAxesLooseInset',[0,0,0,0]);
for n=1:nbRepros
	for j=2:length(tList)
		subplot(length(tList)-1,nbRepros, (j-2)*nbRepros+n); hold on; axis off;
		if j==2
			title(['Reproduction ' num2str(n)],'fontsize',10);
		end
		plotArm(r(n).q(:,tList(j)), [ones(model.nbQ-1,1)*armLength; armLength*4], [0; 0; -1], .002, [.7 .7 .7],[.7 .7 .7]);
		ql = ones(model.nbQ,1) * 999;
		for t=tList(j-1):tList(j) 
			if norm(ql-r(n).q(:,t))>0.08 || t==tList(j)
				colTmp = [.3 .3 .7] + [.5 .5 .3] * (tList(j)-t)/(tList(j)-tList(j-1));
				plotArm(r(n).q(:,t), ones(model.nbQ,1)*armLength, [0; 0; t/2], .05, colTmp); 
				plot(r(n).obj(1,1,t), r(n).obj(2,1,t), '.','markersize',20,'color',[.8 0 0]);
				plot(r(n).obj(1,2,t), r(n).obj(2,2,t), '.','markersize',20,'color',[0 0 .8]);
				ql = r(n).q(:,t);
			end
		end
		text(0.1, -0.2, ['t=' num2str(tList(j)/100,'%.1f')]);
		axis equal; axis([-1 1 -.2 1.1]); 
	end
end

%print('-dpng','graphs/demoIK_pointing_TPGMM01.png');
%pause;
%close all;