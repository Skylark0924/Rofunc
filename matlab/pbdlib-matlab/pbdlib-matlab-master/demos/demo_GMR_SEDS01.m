function demo_GMR_SEDS01
% Continuous autonomous dynamical system with state-space encoding using GMM, with GMR used for 
% reproduction by using a constrained optimization similar to Khansari-Zadeh and Billard's SEDS approach. 
%
% If this code is useful for your research, please cite the related publications:
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
% @article{Khansari11TRO,
%   author="Khansari-Zadeh, S. M. and Billard, A.",
%   title="Learning Stable Non-Linear Dynamical Systems with {G}aussian Mixture Models",
%   journal="{IEEE} Trans. on Robotics",
%   year="2011",
%   volume="27",
%   number="5",
%   pages="943--957"
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
model.nbStates = 6; %Number of states in the GMM
model.nbVar = 4; %Number of variables [x1,x2,dx1,dx2]
model.dt = 0.01; %Time step duration
nbData = 100; %Length of each trajectory
nbSamples = 5; %Number of demonstrations


%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/Snake.mat');
%load('data/2Dletters/S.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).Data = s(n).Data - repmat(s(n).Data(:,end),1,nbData); %Center at 0
	Data0(:,n) = s(n).Data(:,1);
	s(n).Data = [s(n).Data; gradient(s(n).Data)/model.dt]; %Velocity computation	
	Data = [Data, s(n).Data]; 
end
Data0 = mean(Data0,2);


%% Learning 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);
model = EM_GMM(Data, model);


%% Constrained optimization (Regression AB*X=Y -> [A,b]*[x;1]=dx)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Refinement through constrained optimization...');
in=[1:2]; out=[3:4]; 

%Weight based on GMM
for i=1:model.nbStates
  H(:,i) = model.Priors(i) * gaussPDF(Data(in,:),model.Mu(in,i),model.Sigma(in,in,i));
end
H = H ./ repmat(sum(H,2),1,model.nbStates);    

options = optimset('Algorithm','active-set','display','notify'); %'large-scale' could be used instead 'active-set'
X = Data(in,:);
Y = Data(out,:);
for i=1:model.nbStates
  %Initialization through weighted least-squares regression (dx=A*x)
  A0 = [(X * diag(H(:,i).^2) * X') \ X * diag(H(:,i).^2) * Y']';
	%A0 = model.Sigma(out,in,i) / model.Sigma(in,in,i);
  %Refined solution through constrained optimization
  model.A(:,:,i) = fmincon(@(A)myfun(A,X,Y,diag(H(:,i))),A0,[],[],[],[],[],[],@(A)mycon(A),options);
	%[V,D] = eigs(model.A(:,:,i));
  %real(diag(D))
end


%% Reproduction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Instable reproduction
rData0 = zeros(2,nbData);
rData0(:,1) = Data0;
for t=2:nbData
	dx = GMR(model, rData0(:,t-1), [1:2], [3:4]); 
	rData0(:,t) = rData0(:,t-1) + dx * model.dt;
end

%Stable reproduction
rData = zeros(2,nbData);
rData(:,1) = Data0;
for t=2:nbData
  %Weight based on GMM
  for i=1:model.nbStates
    h(i,1) = model.Priors(i) * gaussPDF(rData(:,t-1), model.Mu(1:2,i), model.Sigma(1:2,1:2,i));
  end
  h = h / sum(h);
  %Compute velocity command through GMR
	dx = 0;
  for i=1:model.nbStates
    dx = dx + h(i) * model.A(:,:,i) * rData(:,t-1); 
  end
  %Update position
  rData(:,t) = rData(:,t-1) + dx * model.dt;
end


%% Plot 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 20 24],'position',[10,10,1000,650],'color',[1 1 1]); hold on; axis off; 
for i=1:nbSamples
	plot(Data(1,(i-1)*nbData+1:i*nbData), Data(2,(i-1)*nbData+1:i*nbData), '-','lineWidth',1,'color',[.4 .4 .4]);
end
for c=1:25
	plotGMM(zeros(2,1), eye(2)*exp(-.4*c)*1E4, [.8 .8 .8]);
end
for i=1:model.nbStates
	plotGMM(model.Mu(1:2,i), model.Sigma(1:2,1:2,i), [.6 .6 .6], .8);
end
hl(1)=plot(rData0(1,:), rData0(2,:), '-','lineWidth',2,'color',[1 0 0]);
hl(2)=plot(rData(1,:), rData(2,:), '-','lineWidth',2,'color',[0 .8 0]);
legend(hl,'Instable GMR reproduction','Stable SEDS reproduction');
axis equal; axis([-5 55 -10 30]);

%print('-dpng','graphs/demo_SEDS01.png');
pause;
close all;


%% Optimization function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function f = myfun(A,X,Y,W)
fTmp = (A*X-Y)*W;
f = norm(reshape(fTmp,size(Y,1)*size(Y,2),1));


%% Constraint function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [c,ceq] = mycon(A)
%E = -inv(A)*b;
Atmp = (A+A')*.5;

[~,D] = eig(Atmp);
for j=1:size(A,1)
  %polesR(j) = real(D(j,j)) + 1E-1; %margin on "negativity"
	polesR(j) = real(D(j,j)) + 1E0; %margin on "negativity"
end
%Force the poles to be in the left plane
c = [polesR]; 
ceq = 0; %nonlinear equalities at x.

% for k=1:size(A,1)
%   c(k) = det(Atmp(1:end-k,1:end-k));
%   if mod(k,2)==0 %even
%     c(k) = -c(k);
%   end
% end
% ceq=0;