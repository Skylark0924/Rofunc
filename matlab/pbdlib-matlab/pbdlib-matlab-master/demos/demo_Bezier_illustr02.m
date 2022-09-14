function demo_Bezier_illustr02
% Illustration of linear, quadratic and cubic Bezier curves 
%
% If this code is useful for your research, please cite the related publication:
% @incollection{Calinon19MM,
%   author="Calinon, S.",
%   title="Mixture Models for the Analysis, Edition, and Synthesis of Continuous Time Series",
%   booktitle="Mixture Models and Applications",
%   publisher="Springer",
%   editor="Bouguila, N. and Fan, W.", 
%   year="2019"
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


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbVar = 2; %Dimension of datapoint
% nbDeg = 3; %Degree of the Bezier curve (1=linear, 2=quadratic, 3=cubic)
nbData = 100; %Number of datapoints in a trajectory
tl = linspace(0,1,nbData);


% %% General Bezier curve
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P = rand(nbVar,nbDeg+1); %Control points
% B = zeros(nbDeg,nbData);
% for i=0:nbDeg
% 	B(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-t).^(nbDeg-i) .* t.^i; %Bernstein basis functions
% end
% x = P * B; 


% %% Linear Bezier curve
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % P = rand(nbVar,nbDeg+1); %Control points
% P = [[0;0],[5;4]]; %Control points
% for t=1:nbData
% 	Bp1p2(:,t) = P(:,1) .* (1-tl(t)) + P(:,2) .* tl(t);
% end
% x = Bp1p2; %Linear Bezier curve
% 
% %Plot
% figure; hold on; axis off;
% plot(x(1,:), x(2,:), '-','linewidth',2,'color',[0 0 0]);
% plot(P(1,:), P(2,:), '.','markersize',20,'color',[.8 0 0]);
% 
% t=31;
% plot(x(1,t), x(2,t), '.','markersize',20,'color',[0 0 0]);
% 
% axis equal;
% % print('-dpng','graphs/demo_Bezier_linear01.png');


% %% Quadratic Bezier curve
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % P = rand(nbVar,nbDeg+1); %Control points
% P = [[0;0],[2;4],[5;1]]; %Control points
% for t=1:nbData
% 	Bp1p2(:,t) = P(:,1) .* (1-tl(t)) + P(:,2) .* tl(t);
% 	Bp2p3(:,t) = P(:,2) .* (1-tl(t)) + P(:,3) .* tl(t);
% 	Bp1p2p3(:,t) = Bp1p2(:,t) .* (1-tl(t)) + Bp2p3(:,t) .* tl(t);
% end
% x = Bp1p2p3; %Quadratic Bezier curve
% 
% %Plot
% figure; hold on; axis off;
% for t=1:10:nbData
% 	msh = [Bp1p2(:,t), Bp2p3(:,t)];
% 	plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.7 .7 .7]);
% end
% plot(P(1,:), P(2,:), '-','linewidth',1,'color',[0 0 0]);
% plot(x(1,:), x(2,:), '-','linewidth',3,'color',[0 0 0]);
% plot(P(1,:), P(2,:), '.','markersize',20,'color',[.8 0 0]);
% 
% t=31;
% msh = [Bp1p2(:,t), Bp2p3(:,t)];
% plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 .6 0]);
% plot(msh(1,:), msh(2,:), '.','markersize',20,'color',[0 .6 0]);
% plot(x(1,t), x(2,t), '.','markersize',20,'color',[0 0 0]);
% 
% axis equal;
% % print('-dpng','graphs/demo_Bezier_quadratic01.png');


%% Cubic Bezier curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P = rand(nbVar,nbDeg+1); %Control points
P = [[0;1],[1;4],[5;3],[7;0]]; %Control points
for t=1:nbData
	Bp1p2(:,t) = P(:,1) .* (1-tl(t)) + P(:,2) .* tl(t);
	Bp2p3(:,t) = P(:,2) .* (1-tl(t)) + P(:,3) .* tl(t);
	Bp3p4(:,t) = P(:,3) .* (1-tl(t)) + P(:,4) .* tl(t);
	Bp1p2p3(:,t) = Bp1p2(:,t) .* (1-tl(t)) + Bp2p3(:,t) .* tl(t);
	Bp2p3p4(:,t) = Bp2p3(:,t) .* (1-tl(t)) + Bp3p4(:,t) .* tl(t);
	Bp1p2p3p4(:,t) = Bp1p2p3(:,t) .* (1-tl(t)) + Bp2p3p4(:,t) .* tl(t);
end
x = Bp1p2p3p4; %Cubic Bezier curve

%Plot
figure; hold on; axis off;
for t=1:10:nbData
	msh = [Bp1p2(:,t), Bp2p3(:,t)];
	plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.7 .7 .7]);
	msh = [Bp2p3(:,t), Bp3p4(:,t)];
	plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.7 .7 .7]);
	msh = [Bp1p2p3(:,t), Bp2p3p4(:,t)];
	plot(msh(1,:), msh(2,:), '-','linewidth',1,'color',[.8 .8 .8]);
end
plot(P(1,1:2), P(2,1:2), '-','linewidth',1,'color',[0 0 0]);
plot(P(1,3:4), P(2,3:4), '-','linewidth',1,'color',[0 0 0]);
plot(x(1,:), x(2,:), '-','linewidth',3,'color',[0 0 0]);
plot(P(1,:), P(2,:), '.','markersize',20,'color',[.8 0 0]);

t=31;
msh = [Bp1p2(:,t), Bp2p3(:,t)];
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 .6 0]);
plot(msh(1,:), msh(2,:), '.','markersize',20,'color',[0 .6 0]);
msh = [Bp2p3(:,t), Bp3p4(:,t)];
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 .6 0]);
plot(msh(1,:), msh(2,:), '.','markersize',20,'color',[0 .6 0]);
msh = [Bp1p2p3(:,t), Bp2p3p4(:,t)];
plot(msh(1,:), msh(2,:), '-','linewidth',2,'color',[0 0 .8]);	
plot(msh(1,:), msh(2,:), '.','markersize',20,'color',[0 0 .8]);
plot(x(1,t), x(2,t), '.','markersize',20,'color',[0 0 0]);

axis equal;
% print('-dpng','graphs/demo_Bezier_cubic01.png');


% %% Additional plot
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nbDeg=10;
% B = zeros(nbDeg,nbData);
% for i=0:nbDeg
% 	B(i+1,:) = factorial(nbDeg) ./ (factorial(i) .* factorial(nbDeg-i)) .* (1-tl).^(nbDeg-i) .* tl.^i; %Bernstein basis functions
% end
% figure; hold on;
% for i=0:nbDeg
% 	plot(tl, B(i+1,:), 'linewidth',2);
% end
% set(gca,'xtick',[],'ytick',[],'linewidth',2);
% xlabel('t','fontsize',22); ylabel('b_i','fontsize',22);
% % print('-dpng','graphs/demo_Bezier_linear02.png');
% % print('-dpng','graphs/demo_Bezier_quadratic02.png');
% % print('-dpng','graphs/demo_Bezier_cubic02.png');
% % print('-dpng','graphs/demo_Bezier_highOrder02.png');

pause;
close all;