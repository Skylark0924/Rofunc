function demo_LS01
% Multivariate ordinary least squares 
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarIn = 2; %Dimension of input vector
nbVarOut = 1; %Dimension of output vector
nbData = 40; %Number of datapoints


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A0 = rand(nbVarIn,nbVarOut)-0.5; %Linear relation between input and output (to be estimated)
A0 = [3; 2]; %Linear relation between input and output (to be estimated)
X = rand(nbData,nbVarIn); %Input data
Y = X * A0 + randn(nbData,nbVarOut)*5E-1; %Output data (with noise)


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if nbData > nbVarIn
% 	A = (X'*X)\X' * Y; 
% else
% 	A = X'/(X*X') * Y; 
% end
A = pinv(X) * Y;

%Compute error
%norm(Y-X*A)^2
(Y-X*A)' * (Y-X*A)
e = 0;
for t=1:nbData
	%e = e + norm(Y(t,:)-X(t,:)*A)^2;
	e = e + (Y(t,:)-X(t,:)*A)' * (Y(t,:)-X(t,:)*A);
end
e


% %% 2D Plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure('PaperPosition',[0 0 8 4],'position',[10,10,1300,600]); 
% for i=1:nbVarIn
% 	for j=1:nbVarOut	
% 		subplot(nbVarOut,nbVarIn,(j-1)*nbVarIn+i); hold on;
% 		for t=1:nbData
% 			plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',[.7 .7 .7]);
% 			plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
% 		end
% 		plot([0 1], [0 A(i,j)], '-','linewidth',2,'color',[.8 0 0]);
% 		xlabel(['x_' num2str(i)],'fontsize',18); ylabel(['y_' num2str(j)],'fontsize',18);
% 		%axis([0 1 -1 6]);
% 	end
% end
% %print('-dpng','-r600','graphs/demo_LS_2D01.png');


%% 3D Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 6],'position',[10,10,800,700]); hold on; box on;
for t=1:nbData
	plot3([X(t,1) X(t,1)], [X(t,2) X(t,2)], [Y(t,1) X(t,:)*A], '-','linewidth',2,'color',[.7 .7 .7]);
	plot3(X(t,1), X(t,2), Y(t,1), '.','markersize',14,'color',[0 0 0]);
end
patch([0 0 1 1 0], [0 1 1 0 0], [0 [0,1]*A [1,1]*A [1,0]*A 0], [1 .4 .4],'linewidth',2,'edgecolor',[.8 0 0],'facealpha',0.2);
view(3); xlabel('x_1','fontsize',18); ylabel('x_2','fontsize',18); zlabel('y_1','fontsize',18);
axis([0 1 0 1 -1 6]);
% print('-dpng','-r600','graphs/demo_LS_3D01.png');

% figure('PaperPosition',[0 0 8 6],'position',[10,10,1300,700]); 
% subplot(1,2,1); hold on; 
% for t=1:nbData
% 	plot3([X(t,1) X(t,1)], [X(t,2) X(t,2)], [Y(t,1) X(t,:)*A], '-','linewidth',2,'color',[.7 .7 .7]);
% 	plot3(X(t,1), X(t,2), Y(t,1), '.','markersize',14,'color',[0 0 0]);
% end
% patch([0 0 1 1 0], [0 1 1 0 0], [0 [0,1]*A [1,1]*A [1,0]*A 0], [1 .4 .4],'linewidth',2,'edgecolor',[.8 0 0],'facealpha',0.2);
% view(0,0); xlabel('x_1','fontsize',18); ylabel('x_2','fontsize',18); zlabel('y_1','fontsize',18);
% axis([0 1 0 1 -1 6]);
% subplot(1,2,2); hold on; 
% for t=1:nbData
% 	plot3([X(t,1) X(t,1)], [X(t,2) X(t,2)], [Y(t,1) X(t,:)*A], '-','linewidth',2,'color',[.7 .7 .7]);
% 	plot3(X(t,1), X(t,2), Y(t,1), '.','markersize',14,'color',[0 0 0]);
% end
% patch([0 0 1 1 0], [0 1 1 0 0], [0 [0,1]*A [1,1]*A [1,0]*A 0], [1 .4 .4],'linewidth',2,'edgecolor',[.8 0 0],'facealpha',0.2);
% view(90,0); xlabel('x_1','fontsize',18); ylabel('x_2','fontsize',18); zlabel('y_1','fontsize',18);
% axis([0 1 0 1 -1 6]);
% % print('-dpng','-r600','graphs/demo_LS_2D02.png');

pause;
close all;