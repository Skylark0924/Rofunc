function demo_IK_nullspace_TPGMM01
% Inverse kinematics with nullspace treated with task-parameterized GMM (bimanual tracking task, version with 4 frames).
% 
% This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).
% First run 'startup_rvc' from the robotics toolbox.
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
disp('This demo requires the robotics toolbox RTB10 (http://petercorke.com/wordpress/toolboxes/robotics-toolbox).');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 1; %Number of states
model.nbFrames = 4; %Number of frames
model.nbVars = [2,2,2,2]; %[xl],[xr2],[xl2],[xr]
model.nbVar = max(model.nbVars);
model.nbQ = 5; %Number of variables in configuration space (joint angles)
model.nbX = 2; %Number of variables in operational space (end-effector position)
model.nbVarOut = model.nbQ; %[q]
model.dt = 0.01; %Time step
nbSamples = 1; %Number of demonstration
nbRepros = 1; %Number of reproduction
nbData = 200; %Number of datapoints in a demonstration
pinvDampCoeff = 1e-8; %Coefficient for damped pseudoinverse

needsModel = 1;


%% Create robot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
armLength = .5;
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
arm = SerialLink(repmat(L1,3,1));


%% Generate demonstrations 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if needsModel==1
disp('Demonstration...');
for n=1:nbSamples
	q = [pi/2 pi/2 pi/3 -pi/2 -pi/3]'; %Initial pose
	for t=1:nbData
		s(n).q(:,t) = q; %Log data
		%Forward kinematics
		Htmp = arm.fkine(q(1:3));
		s(n).lx(:,t) = Htmp.t(1:2,end);
		Htmp = arm.fkine(q([1,4:5]));
		s(n).rx(:,t) = Htmp.t(1:2,end);
		%Reference trajectory
		if t==1 
			%Objects moving on a line
			s(n).lxh = [linspace(s(n).lx(1,1),s(n).lx(1,1)-.6*armLength,nbData); linspace(s(n).lx(2,1),s(n).lx(2,1)+2*armLength,nbData)];
			s(n).rxh = [linspace(s(n).rx(1,1),s(n).rx(1,1)+.6*armLength,nbData); linspace(s(n).rx(2,1),s(n).rx(2,1)+2*armLength,nbData)];
% 			%Objects moving on a curve
% 			s(n).lxh = [-sin(linspace(0,pi,nbData))*0.4+s(n).lx(1,1); linspace(s(n).lx(2,1),s(n).lx(2,1)+2*armLength,nbData)];
% 			s(n).rxh = [sin(linspace(0,pi,nbData))*0.4+s(n).rx(1,1); linspace(s(n).rx(2,1),s(n).rx(2,1)+2*armLength,nbData)];
		end
		%Build Jacobians
		lJ = arm.jacob0(q(1:3),'trans');
		lJ = lJ(1:2,:);
		rJ = arm.jacob0(q([1,4:5]),'trans');
		rJ = rJ(1:2,:);
		J = lJ; 
		J(3:4,[1,4:5]) = rJ;
		Ja = J(1:2,:);
		Jb = J(3:4,:);
		pinvJ = (J'*J+eye(model.nbQ)*pinvDampCoeff) \ J'; %damped pseudoinverse
		pinvJa = (Ja'*Ja+eye(model.nbQ)*pinvDampCoeff) \ Ja'; %damped pseudoinverse
		pinvJb = (Jb'*Jb+eye(model.nbQ)*pinvDampCoeff) \ Jb'; %damped pseudoinverse
		
% 		Na = eye(model.nbQ) - pinvJa*Ja; %Nullspace projection matrix
% 		Nb = eye(model.nbQ) - pinvJb*Jb; %Nullspace projection matrix
		
		%An alternative way of computing the nullspace projection matrix is given by
		%http://math.stackexchange.com/questions/421813/projection-matrix-onto-null-space
		[U,S,V] = svd(pinvJa);
		S2 = zeros(model.nbQ);
		S2(model.nbX+1:end,model.nbX+1:end) = eye(model.nbQ-model.nbX);
		Na = U * S2 * U';
% 		%pinvNa = U*(eye(5)-S2)*U';
% 		pinvNa = U*pinv(S2)*U';
		
		[U,S,V] = svd(pinvJb);
		S2 = zeros(model.nbQ);
		S2(model.nbX+1:end,model.nbX+1:end) = eye(model.nbQ-model.nbX);
		Nb = U * S2 * U';
% 		%pinvNb = U*(eye(5)-S2)*U';
% 		pinvNb = U*pinv(S2)*U';
		
		%IK controller
		ldx = (s(n).lxh(:,t) - s(n).lx(:,t)) / model.dt;
		rdx = (s(n).rxh(:,t) - s(n).rx(:,t)) / model.dt;
		
		%Generate artificial dataset
		%dq =  pinvJ * [ldx; rdx]; %Equal priority between arms
% 		dq =  pinvJa * ldx + Na * pinvJb * rdx;	%Priority on left arm
		dq =  pinvJb * rdx + Nb * pinvJa * ldx; %Priority on right arm

		% Projection on local frames
		% Projecting end-effector position x on a frame (A,b) is performed by inv(A)*(x-b)
% 		s(n).fr(1).Data(:,t) = ldx * model.dt;
% 		s(n).fr(2).Data(:,t) = Jb*Na*pinvJb*rdx * model.dt;
% % 		s(n).fr(2).Data(:,t) = rdx * model.dt;
% 		s(n).fr(3).Data(:,t) = Ja*Nb*pinvJa*ldx * model.dt;
% % 		s(n).fr(3).Data(:,t) = ldx * model.dt;
% 		s(n).fr(4).Data(:,t) = rdx * model.dt;
		
		% Projecting end-effector position x on a frame (A,b) performed by inv(A)*(x-b)
 		s(n).fr(1).Data(:,t) = s(n).lx(:,t)-s(n).lxh(:,t);
		s(n).fr(2).Data(:,t) = Jb*Na*pinvJb*rdx * model.dt;
%  		s(n).fr(2).Data(:,t) = s(n).rx(:,t)-s(n).rxh(:,t);
		s(n).fr(3).Data(:,t) = Ja*Nb*pinvJa*ldx * model.dt;
% 		s(n).fr(3).Data(:,t) = s(n).lx(:,t)-s(n).lxh(:,t);
 		s(n).fr(4).Data(:,t) = s(n).rx(:,t)-s(n).rxh(:,t);

		q = q + dq * model.dt;
	end
end

%% Create dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data = zeros(model.nbVar, model.nbFrames, nbData*nbSamples);
for n=1:nbSamples
	s(n).nbData = nbData;
	for m=1:model.nbFrames
		Data(1:model.nbVars(m),m,(n-1)*nbData+1:n*nbData) = s(n).fr(m).Data; 
	end
end

%% TP-GMM learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Parameters estimation of TP-GMM with EM:');
model = init_TPGMM_timeBased(Data, model); %Initialization
%model = init_TPGMM_kmeans(Data, model); %Initialization
model = EM_TPGMM(Data, model);

% for m=1:model.nbFrames
% 	figure, hold on;
% 	title(['Frame ' num2str(m)]);
% 	plot(s(1).fr(m).Data(1,:),s(1).fr(m).Data(2,:));
%  	plotGMM(squeeze(model.Mu(1:2,m,:)),squeeze(model.Sigma(1:2,1:2,m,:)),[0 1.0 0]);
% end

model.nbVar = model.nbQ; %Update of nbVar to later use productTPGMM()

%% Reproduction 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Reproduction...');
rr.Priors = model.Priors;
rr.nbStates = model.nbStates;
% figure
for n=1:nbRepros
	r(n).q(:,1) = [pi/2 pi/2 pi/3 -pi/2 -pi/3]; %Initial pose
	for t=1:nbData
		%Forward kinematics
		Htmp = arm.fkine(r(n).q(1:3,t));
		r(n).lx(:,t) = Htmp.t(1:2,end);
		Htmp = arm.fkine(r(n).q([1,4:5],t));
		r(n).rx(:,t) = Htmp.t(1:2,end);
		%Reference trajectory
		if t==1
			%Objects moving on a line
			r(n).lxh = [linspace(r(n).lx(1,1),r(n).lx(1,1)-.6*armLength,nbData); linspace(r(n).lx(2,1),r(n).lx(2,1)+2*armLength,nbData)];
			r(n).rxh = [linspace(r(n).rx(1,1),r(n).rx(1,1)+.6*armLength,nbData); linspace(r(n).rx(2,1),r(n).rx(2,1)+2*armLength,nbData)];
% 			%Objects moving on a curve
% 			r(n).lxh = [-sin(linspace(0,pi,nbData))*0.3+r(n).lx(1,1); linspace(r(n).lx(2,1),r(n).lx(2,1)+2*armLength,nbData)];
% 			r(n).rxh = [sin(linspace(0,pi,nbData))*0.3+r(n).rx(1,1); linspace(r(n).rx(2,1),r(n).rx(2,1)+2*armLength,nbData)];
		end
		%IK controller
		ldx = (r(n).lxh(:,t) - r(n).lx(:,t)) / model.dt;
		rdx = (r(n).rxh(:,t) - r(n).rx(:,t)) / model.dt;
		%Build Jacobians
		lJ = arm.jacob0(r(n).q(1:3,t),'trans');
		lJ = lJ(1:2,:);
		rJ = arm.jacob0(r(n).q([1,4:5],t),'trans');
		rJ = rJ(1:2,:);
		J = lJ; 
		J(3:4,[1,4:5]) = rJ;
		Ja = J(1:2,:);
		Jb = J(3:4,:);
		pinvJa = (Ja'*Ja+eye(model.nbQ)*pinvDampCoeff) \ Ja'; %damped pseudoinverse
		pinvJb = (Jb'*Jb+eye(model.nbQ)*pinvDampCoeff) \ Jb'; %damped pseudoinverse
				
% 		Na = eye(model.nbQ) - pinvJa*Ja; %Nullspace projection matrix
% 		Nb = eye(model.nbQ) - pinvJb*Jb; %Nullspace projection matrix
		
		%An alternative way of computing the nullspace projection matrix is given by
		%http://math.stackexchange.com/questions/421813/projection-matrix-onto-null-space
		[U,S,V] = svd(pinvJa);
		S2 = zeros(model.nbQ);
		S2(model.nbX+1:end,model.nbX+1:end) = eye(model.nbQ-model.nbX);
		Na = U * S2 * U';
% 		%pinvNa = U*(eye(5)-S2)*U';
% 		pinvNa = U*pinv(S2)*U';
		
		[U,S,V] = svd(pinvJb);
		S2 = zeros(model.nbQ);
		S2(model.nbX+1:end,model.nbX+1:end) = eye(model.nbQ-model.nbX);
		Nb = U * S2 * U';
% 		%pinvNb = U*(eye(5)-S2)*U';
% 		pinvNb = U*pinv(S2)*U';
		
		%Update frames
		%Priority on left arm (dq =  pinvJa * ldx + Na * pinvJb * rdx)
		%left
		pTmp(1).A = pinvJa;
		pTmp(1).b = r(n).q(:,t) + pinvJa * ldx * model.dt;
		%right
		pTmp(2).A = Na * pinvJb; 
		pTmp(2).b = r(n).q(:,t) + Na * pinvJb * rdx * model.dt; 
		%Priority on right arm (dq =  pinvJb * rdx + Nb * pinvJa * ldx)
		%left
		pTmp(3).A = Nb * pinvJa; 
		pTmp(3).b = r(n).q(:,t) + Nb * pinvJa * ldx * model.dt; 
		%right
		pTmp(4).A = pinvJb;
		pTmp(4).b = r(n).q(:,t) + pinvJb * rdx * model.dt;
		
		%Reproduction with TPGMM
		[rr.Mu, rr.Sigma] = productTPGMM(model, pTmp);
% 		for m=1:model.nbFrames
% 			plotGMM(model.Mu(:,m),model.Sigma(:,:,m),[0 1.0 0]);
% 			hold on
% 		end
		r(n).q(:,t+1) = rr.Mu; 
% 		plotGMM(rr.Mu(1:2),rr.Sigma(1:2,1:2),[1.0 0 0]);
% 		pause
% 		cla
	end
end

save('data/TPGMMtmp.mat','s','r','model');
end %needsModel
load('data/TPGMMtmp.mat');


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1000,450],'color',[1 1 1]); 
colTmp = repmat(linspace(.8,.2,nbData),3,1)';

%DEMOS
subplot(1,2,1); hold on; axis off; title('Demonstration');
for n=1:nbSamples
	for t=round(linspace(1,nbData,10))	
		plotArm(s(n).q(1:3,t), ones(3,1)*armLength, [0;0;t/nbData], .02, colTmp(t,:)); %left arm
		plotArm(s(n).q([1,4:5],t), ones(3,1)*armLength, [0;0;t/nbData], .02, colTmp(t,:)); %right arm
	end
end
for n=1:nbSamples
	plot3(s(n).rxh(1,:), s(n).rxh(2,:), ones(1,nbData)*2, 'r-','linewidth',2);
	plot3(s(n).lxh(1,:), s(n).lxh(2,:), ones(1,nbData)*2, 'r-','linewidth',2);
end
set(gca,'xtick',[],'ytick',[]); axis equal; %axis([-1.1 1.1 -.1 1.2]); 

%REPROS
subplot(1,2,2); hold on; axis off; title('Reproduction');
for n=1:nbRepros
	for t=round(linspace(1,nbData,10))
		plotArm(r(n).q(1:3,t), ones(3,1)*armLength, [0;0;t/nbData], .02, colTmp(t,:)); %left arm
		plotArm(r(n).q([1,4:5],t), ones(3,1)*armLength, [0;0;t/nbData], .02, colTmp(t,:)); %right arm
	end
end
for n=1:nbRepros
	plot3(r(n).rxh(1,:), r(n).rxh(2,:), ones(1,nbData)*2, 'r-','linewidth',2);
	plot3(r(n).lxh(1,:), r(n).lxh(2,:), ones(1,nbData)*2, 'r-','linewidth',2);
end
set(gca,'xtick',[],'ytick',[]); axis equal; 

%print('-dpng','graphs/demoIK_nullspace_TPGMM01.png');
pause;
close all;