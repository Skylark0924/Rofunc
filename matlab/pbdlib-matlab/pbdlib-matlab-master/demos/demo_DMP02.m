function demo_DMP02
% Generalization of dynamical movement primitive (DMP) with polynomial fitting using locally weighted regression (LWR). 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19chapter,
% 	author="Calinon, S. and Lee, D.",
% 	title="Learning Control",
% 	booktitle="Humanoid Robotics: a Reference",
% 	publisher="Springer",
% 	editor="Vadakkepat, P. and Goswami, A.", 
% 	year="2019",
% 	doi="10.1007/978-94-007-7194-9_68-1",
% 	pages="1--52"
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
model.nbStates = 8; %Number of activation functions (i.e., number of states in the GMM)
model.nbVar = 1; %Number of variables for the radial basis functions [s] (decay term)
model.nbVarPos = 2; %Number of motion variables [x1,x2] 
model.kP = 50; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.alpha = 1.0; %Decay factor
model.dt = 0.01; %Duration of time step
model.polDeg = 3; %Degree of polynomial fit
nbData = 200; %Length of each trajectory
nbSamples = 5; %Number of demonstrations
L = [eye(model.nbVarPos)*model.kP, eye(model.nbVarPos)*model.kV]; %Feedback term


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posId=[1:model.nbVarPos]; velId=[model.nbVarPos+1:2*model.nbVarPos]; accId=[2*model.nbVarPos+1:3*model.nbVarPos]; 
demos=[];
load('data/2Dletters/G.mat');
sIn(1) = 1; %Initialization of decay term
for t=2:nbData
	sIn(t) = sIn(t-1) - model.alpha * sIn(t-1) * model.dt; %Update of decay term (ds/dt=-alpha s)
end
xTar = demos{1}.pos(:,end);
Data=[];
DataDMP=[];
for n=1:nbSamples
	%Demonstration data as [x;dx;ddx]
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).Data = [s(n).Data; gradient(s(n).Data)/model.dt]; %Velocity computation	
	s(n).Data = [s(n).Data; gradient(s(n).Data(end-model.nbVarPos+1:end,:))/model.dt]; %Acceleration computation
	Data = [Data s(n).Data]; %Concatenation of the multiple demonstrations
	%Nonlinear forcing term
	DataDMP = [DataDMP, (s(n).Data(accId,:) - ...
		(repmat(xTar,1,nbData)-s(n).Data(posId,:))*model.kP + s(n).Data(velId,:)*model.kV) ./ repmat(sIn,model.nbVarPos,1)];
end


%% Setting of the basis functions and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(sIn, model);
%model = init_GMM_logBased(sIn, model); %Log-spread in s <-> equal spread in t
%model = init_GMM_kmeans(sIn, model);

%Set Sigma 
for i=1:model.nbStates
	model.Sigma(:,:,i) = 1E-3; %Heuristic setting of covariance
end

%Compute activation
H = zeros(model.nbStates,nbData);
for i=1:model.nbStates
	H(i,:) = gaussPDF(sIn, model.Mu(:,i), model.Sigma(:,:,i));
end
H = H ./ repmat(sum(H,1),model.nbStates,1);
H2 = repmat(H,1,nbSamples);

% %Nonlinear force profile retrieval (as in standard DMP)
% MuF = X * pinv(H2);
% currF = MuF * H; 


% %Nonlinear force profile retrieval for polynomial of degree 0
% X = ones(nbSamples*nbData,1); 
% Xr = ones(nbData,1); 
% Y = DataDMP';
% for i=1:model.nbStates
% 	W = diag(H2(i,:));
% 	MuF(:,i) = X'*W*X \ X'*W * Y; %Weighted least squares
% end
% Yr = (MuF * H)'; 


%Nonlinear force profile retrieval (as in Ijspeert'13, but without (g-y0) modulation)
X = repmat(sIn',nbSamples,1);
Y = DataDMP';
for i=1:model.nbStates
	W = diag(H2(i,:));
	MuF(:,i) = X'*W*X \ X'*W * Y; %Weighted least squares
end
Xr = sIn';
Yr = zeros(nbData,model.nbVarPos);
for i=1:model.nbStates
	Yr = Yr + diag(H(i,:)) * Xr * MuF(:,i)';
end
% for t=1:nbData
% 	for i=1:model.nbStates
% 		Yr(t,:) = Yr(t,:) + H(i,t) * Xr(t,:) * MuF(:,i)';
% 	end
% end


% %Nonlinear force profile retrieval (as in Ijspeert'13, including (g-y0) modulation)
% G = xTar - Data(1:model.nbVarPos,1);
% Yr = zeros(nbData,model.nbVarPos);
% for j=1:model.nbVarPos
% 	X = repmat(sIn',nbSamples,1) .* repmat(G(j),1,nbData*nbSamples)';
% 	Y = DataDMP(j,:)';
% 	for i=1:model.nbStates
% 		W = diag(H2(i,:));
% 		MuF(j,i) = X'*W*X \ X'*W * Y; %Weighted least squares
% 	end
% 	Xr = sIn' .* repmat(G(j),1,nbData)';
% 	for i=1:model.nbStates
% 		Yr(:,j) = Yr(:,j) + diag(H(i,:)) * Xr * MuF(j,i)';
% 	end
% end


% %Nonlinear force profile retrieval - WLS version 1 (with polynomial)
% X = [];
% Xr = [];
% for d=0:model.polDeg 
% 	X = [X, repmat(sIn.^d,1,nbSamples)'];
% 	Xr = [Xr, sIn.^d'];
% end
% Y = DataDMP';
% for i=1:model.nbStates
% 	W = diag(H2(i,:));
% 	MuF(:,:,i) = X'*W*X \ X'*W * Y; %Weighted least squares
% end
% Yr = zeros(nbData,model.nbVarPos);
% for t=1:nbData
% 	for i=1:model.nbStates
% 		Yr(t,:) = Yr(t,:) + H(i,t) * Xr(t,:) * MuF(:,:,i);
% 	end
% end


% %Nonlinear force profile retrieval - WLS version 2 (with polynomial)
% for i=1:model.nbStates
% 	st(i).X = [];
% 	st(i).Xr = [];
% 	for d=0:model.polDeg 
% 		st(i).X = [st(i).X, repmat((sIn-model.Mu(1,i)).^d,1,nbSamples)'];
% 		st(i).Xr = [st(i).Xr, (sIn-model.Mu(1,i)).^d'];
% 	end
% end
% Y = DataDMP';
% for i=1:model.nbStates
% 	W = diag(H2(i,:));
% 	MuF(:,:,i) = st(i).X' * W * st(i).X \ st(i).X' * W * Y; %Weighted least squares
% end
% Yr = zeros(nbData,model.nbVarPos);
% for t=1:nbData
% 	for i=1:model.nbStates
% 		Yr(t,:) = Yr(t,:) + H(i,t) * st(i).Xr(t,:) * MuF(:,:,i);
% 	end
% end


%Motion retrieval with DMP
x = Data(1:model.nbVarPos,1);
dx = zeros(model.nbVarPos,1);
for t=1:nbData
	%Compute acceleration, velocity and position	
	ddx = L * [xTar-x; -dx] + Yr(t,:)' * sIn(t); 
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;
	r(1).Data(:,t) = x;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 16 4],'position',[10,10,1300,500],'color',[1 1 1]); 
xx = round(linspace(1,64,model.nbStates));
clrmap = colormap('jet')*0.5;
clrmap = min(clrmap(xx,:),.9);

%Spatial plot
axes('Position',[0 0 .2 1]); hold on; axis off;
plot(Data(1,:),Data(2,:),'.','markersize',8,'color',[.7 .7 .7]);
plot(r(1).Data(1,:),r(1).Data(2,:),'-','linewidth',3,'color',[.8 0 0]);
axis equal; axis square;  

%Timeline plot of the nonlinear perturbing force
axes('Position',[.25 .58 .7 .4]); hold on; 
for n=1:nbSamples
	plot(sIn, DataDMP(1,(n-1)*nbData+1:n*nbData), '-','linewidth',2,'color',[.7 .7 .7]);
end
[~,id] = max(H,[],1);
for i=1:model.nbStates
	plot(sIn(id==i), Xr(id==i)*MuF(1,i), '-','linewidth',6,'color',min(clrmap(i,:)+0.5,1));
% 	Xr = [];
% 	for d=0:model.polDeg 
% 		Xr = [Xr, sIn(id==i).^d']; %Version 1
% 		%Xr = [Xr, (sIn(id==i)-model.Mu(1,i)).^d']; %Version 2
% 	end
% 	plot(sIn(id==i), Xr*MuF(:,1,i), '-','linewidth',6,'color',min(clrmap(i,:)+0.5,1));
end
plot(sIn, Yr(:,1), '-','linewidth',2,'color',[.8 0 0]);
axis([min(sIn) max(sIn) min(DataDMP(1,:)) max(DataDMP(1,:))]);
ylabel('$F_1$','fontsize',16,'interpreter','latex');
view(180,-90);

%Timeline plot of the basis functions activation
axes('Position',[.25 .12 .7 .4]); hold on; 
for i=1:model.nbStates
	patch([sIn(1), sIn, sIn(end)], [0, H(i,:), 0], min(clrmap(i,:)+0.5,1), 'EdgeColor', 'none', 'facealpha', .4);
	plot(sIn, H(i,:), 'linewidth', 2, 'color', min(clrmap(i,:)+0.2,1));
end
axis([min(sIn) max(sIn) 0 1]);
xlabel('$s$','fontsize',16,'interpreter','latex'); 
ylabel('$h$','fontsize',16,'interpreter','latex');
view(180,-90);

%print('-dpng','-r100','graphs/demo_DMP02.png');
pause;
close all;