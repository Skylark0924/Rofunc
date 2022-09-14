function demo_ILC01
% Iterative correction of errors for a recurring movement with ILC
% (Lagrange form (quadratic optimal design, Q-ILC), notation as in Koc, Maeda, Neumann and Peters (2015) 
% "Optimizing robot striking movement primitives with iterative learning control") 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	pages="1261--1312",
% 	doi="10.1007/978-94-007-6046-2_68"
% }
%
% Copyright (c) 2019 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon and Hakan Girgin
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
nbData = 200; %Length of a trajectory
nbCycles = 3; %Number of demonstrations

model.dt = 0.01; %Time step
model.kP = 10; %Stiffness 
model.kV = (2*model.kP)^.5; %Damping
model.kPlow = 1; %Low stiffness 
model.kVlow = (2*model.kP)^.5; %Low damping
model.nbVarPos = 2; %Dimension of position data (here: x1,x2)
model.nbDeriv = 2; %Number of static & dynamic features (D=2 for [x,dx])
model.nbVar = model.nbVarPos * model.nbDeriv; %Dimension of state vector
model.qfactor = 1E1;	%Error cost in LQR
model.rfactor = 1E-5;	%Control cost in LQR

%Dynamical system variables (discrete version)
A = kron([1, model.dt; 0, 1], eye(model.nbVarPos));
B = kron([0; model.dt], eye(model.nbVarPos));

%Feedback gains
K = [eye(model.nbVarPos)*model.kP, eye(model.nbVarPos)*model.kV];
Klow = [eye(model.nbVarPos)*model.kPlow, eye(model.nbVarPos)*model.kVlow];

%Cost matrices
Q = eye(model.nbVar) * model.qfactor;
Q = kron(eye(nbData), Q);
R = eye(model.nbVarPos) * model.rfactor;
R = kron(eye(nbData), R);

%ILC variables
F = zeros(nbData*model.nbVar, nbData*model.nbVarPos); %Input-to-output matrix, see Eq. (8) in Koc et al
for i=1:nbData
	for j=1:nbData
		ii = [1:model.nbVar] + (i-1)*model.nbVar;
		jj = [1:model.nbVarPos] + (j-1)*model.nbVarPos;
		if j<i
			F(ii,jj) = A^((i-1)-j) * B; %Eq. (8) in Koc et al
		elseif j==i
			F(ii,jj) = B; %Eq. (8) in Koc et al
		else
			F(ii,jj) = 0; %Eq. (8) in Koc et al
		end	
	end
end
G = (F'*Q*F + R) \ (F'*Q*F); %Filtering matrix, see Eq. (10) in Koc et al
L = (F'*Q*F) \ (F'*Q); %Learning matrix, see Eq. (10) in Koc et al


%% Generate/load position and velocity profiles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tlist = linspace(0,2*pi,nbData);
% xpos = [cos(tlist); sin(2*tlist)];
% xh = [xpos; gradient(xpos)/model.dt];

Data = csvread('data/b1_contour.csv')';
xpos = spline(1:size(Data,2), Data(1:2,:), linspace(1,size(Data,2),nbData)); %Resample data
xh = [xpos; gradient(xpos)/model.dt]; %pos+vel


%% Iterative learning control (ILC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n=1:nbCycles
	%x = [1; 1; 0; 0]; %pos+vel 
	x = xh(:,1); %pos+vel 
	if n==1 %Initial controller
		U = [];
		E = [];
		for t=1:nbData
			r(n).x(:,t) = x; %Log data
			e = xh(:,t) - x; %Tracking error
			u = K*e + Klow*e; %acceleration command
			%Simulation of system dynamics (with simulated noise)
			x = A * x + B * u + [randn(2,1)*1E-1; randn(2,1)*1E-1];
			%Log data
			U = [U; u];
			E = [E; e];
		end
	else %Iterative correction of the controller with ILC
		U = G * (U+L*E); %Eq. (6) in Koc et al
		E = [];
		for t=1:nbData
			r(n).x(:,t) = x; %Log data
			e = xh(:,t) - x; %Tracking error
			%Simulation of robot dynamics (with simulated noise)
			id = [1:model.nbVarPos] + (t-1)*model.nbVarPos;
			x = A * x + B * (U(id) + Klow*e) + [randn(2,1)*1E-1; randn(2,1)*1E-1];
			%Log data
			E = [E; e];
		end
	end
end


% %% Plot 2d (animation)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('position',[20,10,1000,750]); hold on; axis off;
% plot(xh(1,:), xh(2,:), '-', 'color', [1 .7 .7])
% axis equal; axis([min(xh(1,:))-100 max(xh(1,:))+100 min(xh(2,:))-100 max(xh(2,:))+100]); 
% %set(gca,'xtick',[],'ytick',[]);
% h = [];
% for n=1:nbCycles
% 	for t=round(linspace(1,nbData,50))
% 		delete(h);
% 		h=[];
% 		h = [h plot(xh(1,t), xh(2,t), '.', 'markersize', 25, 'color', [1 .7 .7])];
% 		h = [h plot(r(n).x(1,t), r(n).x(2,t), '.', 'markersize', 25, 'color', [.7 .7 .7])];
% 		pause(0.1);
% 		if t==1
% 			pause
% 		end
% 	end
% end


%% Plot profiles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,700]); hold on; 
for i=1:model.nbVarPos*2
	for n=1:nbCycles
		%subplot(model.nbVarPos, nbCycles, (i-1)*nbCycles+n); title(['n=' num2str(n)]); hold on;
		subplot(nbCycles, model.nbVarPos*2, (n-1)*model.nbVarPos*2+i); title(['n=' num2str(n)]); hold on;
		plot(xh(i,:), '-', 'linewidth', 2, 'color', [1 .7 .7]);
		plot(r(n).x(i,:), '-', 'linewidth', 2, 'color', [.7 .7 .7]);
		xlabel('t'); 
		if i>2
			ylabel(['dx_' num2str(i-2)]);
		else
			ylabel(['x_' num2str(i)]);
		end
		set(gca,'xtick',[],'ytick',[]);
		axis([1 nbData min(xh(i,:))-100 max(xh(i,:))+100]);
	end
end

%print('-dpng','graphs/demo_ILC01.png');
pause;
close all;