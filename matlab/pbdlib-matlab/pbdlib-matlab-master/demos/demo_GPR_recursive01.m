function demo_GPR_recursive01
% Recursive computation of Gaussian process regression (GPR) 
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVar = 3; %Dimension of datapoint (t,x1,x2)
nbData = 20; %Number of datapoints
nbDataRepro = 100; %Number of datapoints for reproduction
nbSamples = 1; %Number of demonstrations
p(1)=1E0; p(2)=1E1; p(3)=1E-2; %GPR parameters


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/G.mat');
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	tt = [1:nbData/2,3*nbData/4:nbData];  %Simulate missing data
	s(n).Data = [tt; s(n).Data(:,tt)];
	Data = [Data s(n).Data]; 
end
xIn = Data(1,:);
xOut = Data(2:end,:);
xInHat = linspace(1,nbData,nbDataRepro);


%% Standard GPR computation (requires O(N^3) operations and O(N^2) storage)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = p(1) .* exp(-p(2).^-1 .* pdist2(xIn', xIn').^2) + p(3) .* eye(size(xIn,2)); 
Kd = p(1) .* exp(-p(2).^-1 .* pdist2(xInHat', xIn').^2);
r(1).Data = [xInHat; (Kd * (K \ xOut'))']; 


%% GPR computation through recursive least squares estimate (does not require matrix inversion)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tList = 1:size(xIn,2);
A = zeros(size(xIn,2), size(xOut,1)); %Initial estimate of A
iB = eye(size(xIn,2)) * 1E10; %Initial estimate of iB
for t=1:size(xIn,2)
% 	V = K(t,:); %New input data
	V = p(1) .* exp(-p(2).^-1 .* pdist2(xIn(:,t)', xIn').^2) + p(3) .* (tList==t); %New input data
	C = xOut(:,t)'; %New output data 
	L =  iB*V' ./ (1 + V*iB*V'); %Kalman gain
	A = A + L * (C-V*A); %Update A
	iB = iB - iB*V' ./ (1+V*iB*V')*V*iB; %Update iB
end
r(2).Data = [xInHat; (Kd * A)']; 


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1300 1300]);
%Plot 1D
for m=2:nbVar
	limAxes = [1, nbData, min(Data(m,:))-1E0, max(Data(m,:))+1E0];
	subplot(nbVar-1,1,m-1); hold on;
	h(1) = plot(r(1).Data(1,:), r(1).Data(m,:), '-','lineWidth',3.5,'color',[.8 .8 .8]);
	h(2) = plot(r(2).Data(1,:), r(2).Data(m,:), ':','lineWidth',3.5,'color',[.8 0 0]);
	for n=1:nbSamples
		plot(s(n).Data(1,:), s(n).Data(m,:), '.','markersize',20,'color',[.2 .2 .2]);
	end
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	ylabel(['$x_' num2str(m-1) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
	legend(h,{'Standard GPR','Recursive GPR'});
end

%print('-dpng','graphs/demo_GPR_recursive01.png');
pause;
close all;