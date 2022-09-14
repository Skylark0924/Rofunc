function demo_LS_polFit01
% Polynomial fitting with least squares 
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%nbVarIn = 2; %Dimension of input vector
%nbVarOut = 1; %Dimension of output vector
%nbData = 100; %Number of observations


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A0 = [-.2; -.2; .2; 1]; %Polynomimal relationship between input and output (to be estimated)
% x = linspace(-1,1,nbData)';
% X = [x.^3, x.^2, x, ones(nbData,1)]; %Input data
% Y = X * A0 + randn(nbData,nbVarOut)*2E-2; %Output data (with noise)

load('data/DataLS01.mat');
nbData = size(x,1);

figure('PaperPosition',[0 0 8 4],'position',[10,10,1300,700]);

for nbVarIn=1:6

X = [];
for i=0:nbVarIn-1
	X = [X, x.^i]; %-> X=[x.^3, x.^2, x, 1]
end

%Array used to display more points
xr = linspace(min(x),max(x),200)';
Xr = [];
for i=0:nbVarIn-1
	Xr = [Xr, xr.^i];
end


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = X' * X \ X' * Y;

%Compute fitting error
e = norm(Y - X * A);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
j = 1; %Dimension to display for output data
subplot(2,3,nbVarIn); hold on; title(['Degree ' num2str(nbVarIn-1) ' (e=' num2str(e,'%.2f') ')'],'fontsize',12);
for t=1:nbData
	plot([x(t) x(t)], [Y(t,j) X(t,:)*A(:,j)], '-','linewidth',1,'color',[0 0 0]);
	plot(x(t), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot(xr, Xr*A(:,j), 'r-','linewidth',2);
xlabel('x','fontsize',12);
ylabel('y','fontsize',12);
axis([min(x) max(x) min(Y(:,j))-0.1 max(Y(:,j))+0.1]);
%set(gca,'xtick',[],'ytick',[]);

end %nbVarIn

% print('-dpng','-r300','graphs/demo_LS_polFit01.png');
pause;
close all;