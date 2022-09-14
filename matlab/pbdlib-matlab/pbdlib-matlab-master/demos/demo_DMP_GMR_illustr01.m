function demo_DMP_GMR_illustr01
% Illustration of DMP with GMR to regenerate the nonlinear force profile. 
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
model.nbStates = 5; %Number of states in the GMM
model.nbVar = 3; %Number of variables [s,F1,F2] (decay term and perturbing force)
model.nbVarPos = model.nbVar-1; %Dimension of spatial variables
model.kP = 50; %Stiffness gain
model.kV = (2*model.kP)^.5; %Damping gain (with ideal underdamped damping ratio)
model.alpha = 1.0; %Decay factor
model.dt = 0.01; %Duration of time step
nbData = 200; %Length of each trajectory
nbSamples = 5; %Number of demonstrations
L = [eye(model.nbVarPos)*model.kP, eye(model.nbVarPos)*model.kV]; %Feedback term


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
posId=[1:model.nbVar-1]; velId=[model.nbVar:2*(model.nbVar-1)]; accId=[2*(model.nbVar-1)+1:3*(model.nbVar-1)]; 
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
	%Training data as [s;F]
	DataDMP = [DataDMP [sIn; ...
		(s(n).Data(accId,:) - (repmat(xTar,1,nbData)-s(n).Data(posId,:))*model.kP + s(n).Data(velId,:)*model.kV) ./ repmat(sIn,model.nbVarPos,1)]];
end


%% Learning and reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = init_GMM_timeBased(DataDMP, model);
%model = init_GMM_logBased(DataDMP, model); %Log-spread in s <-> equal spread in t
%model = init_GMM_kmeans(DataDMP, model);
model = EM_GMM(DataDMP, model);
%Nonlinear force profile retrieval
currF = GMR(model, sIn, 1, 2:model.nbVar);
%Motion retrieval with DMP
x = Data(1:model.nbVarPos,1) + [-1;5];
dx = zeros(model.nbVarPos,1);
for t=1:nbData
	%Compute acceleration, velocity and position	
	ddx = L * [xTar-x; -dx] + currF(:,t) * sIn(t); 
	dx = dx + ddx * model.dt;
	x = x + dx * model.dt;
	r(1).Data(:,t) = x;
end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 10 4.0],'position',[10,10,2300,900],'color',[1 1 1]); 
clrmap = lines(model.nbStates);

%Activation of the basis functions
for i=1:model.nbStates
	h(i,:) = model.Priors(i) * gaussPDF(sIn, model.Mu(1,i), model.Sigma(1,1,i));
end
h = h ./ repmat(sum(h,1)+realmin, model.nbStates, 1);

%Spatial plot
subplot(3,3,[1,4,7]); hold on; axis off;
for n=1:nbSamples
	plot(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.4 .4 .4]);
end
%plot(r(1).Data(1,[1,end]),r(1).Data(2,[1,end]),':','linewidth',2,'color',[0 0 0]);
displaySpring(r(1).Data(:,[1,end]), [.6 0 0]);
plot(r(1).Data(1,:),r(1).Data(2,:),'-','linewidth',2,'color',[.8 0 0]);
axis([min(Data(1,:))-0.1 max(Data(1,:))+0.1 min(Data(2,:))-0.1 max(Data(2,:))+4.5]); axis equal; 

%Timeline plot of the nonlinear perturbing force
for j=1:2
subplot(3,3,[2,3]+(j-1)*3); hold on;
for n=1:nbSamples
	plot(sIn, DataDMP(j+1,(n-1)*nbData+1:n*nbData), '-','linewidth',1,'color',[.4 .4 .4]);
end
for i=1:model.nbStates
	plotGMM(model.Mu([1,j+1],i), model.Sigma([1,j+1],[1,j+1],i)+diag([1E-4 2E4]), clrmap(i,:), .7);
end
plot(sIn, currF(j,:), '-','linewidth',2,'color',[.8 0 0]);
axis([sIn(end) sIn(1) min(DataDMP(j+1,:)) max(DataDMP(j+1,:))]);
set(gca,'xtick',[],'ytick',[]);
%set(gca,'xtick',[sIn(end),sIn(1)],'xticklabel',{'1','0'},'ytick',[]);
ylabel(['$f_' num2str(j) '$'],'fontsize',16,'interpreter','latex');
view(180,-90);
end

%Timeline plot of the basis functions activation
subplot(3,3,[8,9]); hold on;
for i=1:model.nbStates
	patch([sIn(1), sIn, sIn(end)], [0, h(i,:), 0], clrmap(i,:), ...
		'linewidth', 1.5, 'EdgeColor', max(clrmap(i,:)-0.2,0), 'facealpha', .7, 'edgealpha', .7);
end
axis([sIn(end) sIn(1) 0 1]);
set(gca,'xtick',[sIn(end),sIn(1)],'xticklabel',{'0','1'},'ytick',[],'fontsize',12);
xlabel('$s$','fontsize',16,'interpreter','latex'); 
ylabel('$h_i$','fontsize',16,'interpreter','latex');
view(180,-90);

%Plot decay term
axes('Position',[0.06 0.78 .15 .2]); hold on; 
plot(sIn,'-','lineWidth',1,'color',[.8 0 0]);
axis([1 nbData sIn(end) sIn(1)]);
set(gca,'xtick',[],'ytick',[sIn(end),sIn(1)],'yticklabel',{'0','1'});
xlabel('$t$','fontsize',16,'interpreter','latex'); 
ylabel('$s$','fontsize',16,'interpreter','latex'); 

%print('-dpng','graphs/DMP_GMR01.png');
pause;
close all;
end


%% Spring display function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hg = displaySpring(x,colTmp)
  sc = 0.35;
  x1 = x(:,1);
  x2 = x(:,2);
  vdiff = norm(x2-x1)/(2*sc);
  h = 1.6;
  msh0 = [-1.5 -h; -4 -h; -4 0; -vdiff 0; -4 0; ...
          -4 h; -3 h; -3 h-1; -2 h+1; -2 h-1; -1 h+1; -1 h-1; 0 h+1; 0 h-1; 1 h+1; 1 h-1; 2 h+1; 2 h-1; 3 h+1; 3 h; 4 h; 4 0; ...
          vdiff 0; 4 0; 4 -h; ...
          3 -h; 3 -h+1; -3 -h+1; -1.5 -h+1; -1.5 -h-1; -3 -h-1; 3 -h-1; 3 -h]' * sc;
  e1 = (x2-x1)/norm(x2-x1);
  e2 = [e1(2); -e1(1)];
  R = [e1 e2];
  if norm(x2-x1)>1.2
    msh = R * msh0 + repmat(x1+(x2-x1)*0.5,1,size(msh0,2));
  else
    msh = [x1 x2];
  end
  
  hg = plot(msh(1,:),msh(2,:),'-','linewidth',1,'color',colTmp); %colTmp
  %plot(msh(1,6:21),msh(2,6:21),'-','linewidth',2,'color',[.8 0 0]);
  %plot(msh(1,25:end),msh(2,25:end),'-','linewidth',2,'color',[.8 .8 0]);

  %h = [h draw2DArrow(msh(:,21)+[0.5;-0.5], msh(:,6)-msh(:,21)+[-0.8;1.2], [.7 0 0])];
  %h = [h draw2DArrow(msh(:,25)+[0.5;-0.5], [1.2;1.1], [.7 0 0])];
  hg = [hg plot(x(1,1),x(2,1),'.','markersize',15,'color',colTmp)];
  hg = [hg plot(x(1,2),x(2,2),'.','markersize',15,'color',colTmp)];
end