function demo_GPR05
% Gaussian process regression (GPR) for motion generation with new targets 
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
nbData = 200; %Number of datapoints
nbDataRepro = 200; %Number of datapoints for reproduction
nbSamples = 4; %Number of demonstrations
p(1)=1E-3; p(2)=1E2; p(3)=1E-8; %GPR parameters


%% Load motion data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/Data02.mat');
xIn = [];
xOut = [];
for n=1:nbSamples
	xIn = [xIn, [s(n).Data(1,:).*1E2; repmat(s(n).Data(2:end,end),1,nbData)]];
	xOut = [xOut, s(n).Data(2:end,:)];
end
xInHat = [linspace(min(xIn(1,:)), max(xIn(1,:)), nbDataRepro); repmat((xIn(2:end,1)+xIn(2:end,end))./2,1,nbDataRepro)];
% xInHat2 = [linspace(min(xIn(1,:)), max(xIn(1,:)), nbDataRepro); repmat(xIn(2:end,1),1,nbDataRepro)];


%% GPR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = p(1) .* exp(-p(2).^-1 .* pdist2(xIn', xIn').^2) + p(3) .* eye(size(xIn,2)); 
Kd = p(1) .* exp(-p(2).^-1 .* pdist2(xInHat', xIn').^2);
xOutHat = [xInHat(1,:); (Kd * (K \ xOut'))'];
%Covariance computation
Kdd = p(1) * exp(-p(2).^-1 .* pdist2(xInHat', xInHat').^2);
S = Kdd - Kd * (K \ Kd');
SigmaOutHat = zeros(nbVar-1,nbVar-1,nbData);
for t=1:nbDataRepro
	SigmaOutHat(:,:,t) = eye(nbVar-1) * S(t,t); 
end

% Kd2 = p(1) .* exp(-p(2).^-1 .* pdist2(xInHat2', xIn').^2);
% xOutHat2 = [xInHat2(1,:); (Kd2 * (K \ xOut'))'];
% %Covariance computation
% Kdd2 = p(1) * exp(-p(2).^-1 .* pdist2(xInHat2', xInHat2').^2);
% S = Kdd2 - Kd2 * (K \ Kd2');
% SigmaOutHat2 = zeros(nbVar-1,nbVar-1,nbData);
% for t=1:nbDataRepro
% 	SigmaOutHat2(:,:,t) = eye(nbVar-1) * S(t,t); 
% end


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10 10 1300 600]);
%Plot 1D
for m=2:nbVar
	limAxes = [xIn(1,1), xIn(1,end), min(xOut(m-1,:))-1E-1, max(xOut(m-1,:))+1E-1];
	subplot(nbVar-1,2,(m-2)*2+1); hold on;
	patch([xOutHat(1,:), xOutHat(1,end:-1:1)], ...
		[xOutHat(m,:)+squeeze(SigmaOutHat(m-1,m-1,:).^.5)'*1E3, xOutHat(m,end:-1:1)-squeeze(SigmaOutHat(m-1,m-1,end:-1:1).^.5)'*1E3], ...
		[1 .8 .8],'edgecolor','none');
% 	patch([xOutHat2(1,:), xOutHat2(1,end:-1:1)], ...
% 		[xOutHat2(m,:)+squeeze(SigmaOutHat2(m-1,m-1,:).^.5)'*1E3, xOutHat2(m,end:-1:1)-squeeze(SigmaOutHat2(m-1,m-1,end:-1:1).^.5)'*1E3], ...
% 		[.8 1 .8],'edgecolor','none');
	for n=1:nbSamples
		plot(xIn(1,(n-1)*nbData+1:n*nbData), xOut(m-1,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]);
	end
	plot(xOutHat(1,:), xOutHat(m,:), '-','lineWidth',2,'color',[.8 0 0]);
	set(gca,'xtick',[],'ytick',[]);
	xlabel('$t$','interpreter','latex','fontsize',18);
	ylabel(['$x_' num2str(m-1) '$'],'interpreter','latex','fontsize',18);
	axis(limAxes);
end
%Plot 2D
subplot(nbVar-1,2,[2:2:(nbVar-1)*2]); hold on;
% plotGMM(xOutHat(2:3,:), SigmaOutHat*1E5, [1 .2 .2],.2);
% plotGMM(xOutHat2(2:3,:), SigmaOutHat2*1E5, [.2 1 .2],.2);
plot(xOutHat(2,:), xOutHat(3,:), '-','lineWidth',2,'color',[.8 0 0]);
plot(xInHat(2,1), xInHat(3,1), '.','markersize',22,'color',[.8 0 0]);
for n=1:nbSamples
	plot(xOut(1,(n-1)*nbData+1:n*nbData), xOut(2,(n-1)*nbData+1:n*nbData), '-','lineWidth',1,'color',[.2 .2 .2]);
	plot(xOut(1,n*nbData), xOut(2,n*nbData), '.','markersize',18,'color',[.2 .2 .2]);
end
set(gca,'xtick',[],'ytick',[]); axis equal; axis square;
xlabel(['$x_1$'],'interpreter','latex','fontsize',18);
ylabel(['$x_2$'],'interpreter','latex','fontsize',18);

%print('-dpng','graphs/demo_GPR05.png');
pause;
close all;